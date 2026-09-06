"""
Shared utilities for comparing the DQN agent (rl_algs/DQN.py) against the
rule-based expert baseline (rl_algs/rl_dqn_expert.py) on the same NPK
fertilization environment.

Provides:
    - configure_npk_args / build_env / build_normalized_env: build the exact
      same environment (state variables, discrete dose set, custom reward)
      for both policies.
    - rollout: run one episode of any `Agent`-compatible policy and collect a
      per-day trace of the variables needed for the comparison plots.
    - train_dqn_quick / load_dqn_policy: get a DQN policy to compare, either
      by training a short demo run in-notebook or loading a checkpoint
      produced by `train_agent.py --agent-type DQN`.
    - plot_yield_and_fertilization / plot_cumulative_reward /
      plot_nitrogen_dynamics: the three publication-style comparison figures.

Written by: Aliou Garga, 2026
"""

import glob
import os
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import utils
from pcse_gym.args import NPK_Args, WOFOST_Args, Agro_Args
from pcse_gym import wrappers
from rl_algs.rl_utils import Agent


# ----------------------------------------------------------------------------
# Environment construction
# ----------------------------------------------------------------------------

STATE_OUTPUT_VARS = ["FIN", "DVS", "LAI", "WSO", "NAVAIL", "NUPTAKETOTAL"]
"""FIN is required internally to detect episode termination. NAVAIL (total
mineral N available in the soil pool) is used as the TNSOIL proxy and
NUPTAKETOTAL (total crop N uptake) as the NUPTT proxy requested in the
research state definition: WOFOST-Gym does not expose variables literally
named TNSOIL/NUPTT."""

STATE_WEATHER_VARS = ["RAIN", "TMIN", "TMAX"]

DEFAULT_DOSE_KG_HA = [0, 40, 70, 80, 100]


def configure_npk_args(args: utils.Args, dose_kg_ha: list = DEFAULT_DOSE_KG_HA) -> utils.Args:
    """Mutate `args` in place so it builds a `Limited_N_Env` (`ln-v0`) exposing
    exactly `dose_kg_ha` as the discrete Nitrogen action set, with the crop,
    soil-N and weather variables needed by `rollout()`.

    Args:
        args: a `utils.Args` (or subclass, e.g. an agent's `AgentArgs`)
        dose_kg_ha: allowed N doses in kg/ha, must include 0
    """
    if args.npk is None:
        args.npk = NPK_Args(wf=WOFOST_Args(), ag=Agro_Args())

    assert isinstance(args.save_folder, str), "`args.save_folder` must be set"
    if not args.save_folder.endswith("/"):
        args.save_folder += "/"
    os.makedirs(args.save_folder, exist_ok=True)

    args.npk.num_fert = int(max(dose_kg_ha))
    args.npk.fert_amount = 1.0
    args.npk.output_vars = list(STATE_OUTPUT_VARS)
    args.npk.weather_vars = list(STATE_WEATHER_VARS)
    args.npk.intvn_interval = 1
    args.dose_kg_ha = list(dose_kg_ha)

    return args


def build_env(args: utils.Args) -> gym.Env:
    """Build the raw (non-normalized) environment: base env -> reward wrapper
    -> dose-restricted action wrapper. `args` must already have been passed
    through `configure_npk_args`.
    """
    env = utils.make_gym_env(args)
    env = utils.wrap_env_reward(env, args)
    if getattr(args, "dose_kg_ha", None):
        env = wrappers.NPKDoseActionWrapper(env, dose_kg_ha=list(args.dose_kg_ha))
    return env


def build_normalized_env(args: utils.Args) -> gym.Env:
    """Same as `build_env`, plus observation/reward normalization. This is
    the wrapper stack a DQN agent is trained on (see `rl_algs/rl_utils.py:make_env`),
    so it must be used to evaluate a trained DQN checkpoint.
    """
    env = build_env(args)
    env = wrappers.NormalizeObservation(env)
    env = wrappers.NormalizeReward(env)
    return env


# ----------------------------------------------------------------------------
# Episode rollout
# ----------------------------------------------------------------------------

ROLLOUT_VARS = ("WSO", "DVS", "LAI", "NAVAIL", "NUPTAKETOTAL", "RAIN", "TMIN", "TMAX")


def rollout(env: gym.Env, policy: Agent, max_steps: int = 400) -> dict:
    """Roll out one episode of `policy` on `env` and collect a per-day trace.

    Values are recorded on their real physical scale even if `env` normalizes
    observations/rewards internally (detected via `unnormalize_obs`/`unnormalize`,
    exposed by `NormalizeObservation`/`NormalizeReward`), so that a DQN agent's
    rollout and the expert's rollout are directly comparable.

    Args:
        env: environment built with `build_env`/`build_normalized_env`
        policy: any object with `get_action(obs) -> int` (e.g.
            `ExpertFertilizationPolicy` or `TorchDQNPolicy`)
        max_steps: safety cap on episode length (days)
    """
    col_names = list(env.unwrapped.output_vars) + list(env.unwrapped.weather_vars) + ["DAYS"]
    idx = {name: i for i, name in enumerate(col_names)}
    for required in ROLLOUT_VARS:
        assert required in idx, (
            f"`{required}` missing from the environment's output/weather vars. "
            f"Build the environment with `configure_npk_args`/`build_env`."
        )

    def to_real(obs: np.ndarray) -> np.ndarray:
        if hasattr(env, "unnormalize_obs"):
            return np.asarray(env.unnormalize_obs(np.asarray(obs)))
        return np.asarray(obs)

    def reward_to_real(reward) -> float:
        reward = float(np.asarray(reward).reshape(-1)[0])
        if hasattr(env, "unnormalize"):
            return float(env.unnormalize(reward))
        return reward

    obs, _ = env.reset()
    data = {k: [] for k in ("day", "dose", "reward", "cum_reward") + ROLLOUT_VARS}

    cum_reward = 0.0
    term, trunc, step = False, False, 0
    while not (term or trunc) and step < max_steps:
        action = int(np.asarray(policy.get_action(obs)).reshape(-1)[0])
        dose = float(env.dose_kg_ha[action])

        real_obs = to_real(obs)
        day = float(real_obs[idx["DAYS"]])

        next_obs, reward, term, trunc, _ = env.step(action)
        reward = reward_to_real(reward)
        cum_reward += reward

        real_next_obs = to_real(next_obs)

        data["day"].append(day)
        data["dose"].append(dose)
        data["reward"].append(reward)
        data["cum_reward"].append(cum_reward)
        for v in ROLLOUT_VARS:
            data[v].append(float(real_next_obs[idx[v]]))

        obs = next_obs
        step += 1

    return {k: np.array(v) for k, v in data.items()}


def summarize_episode(name: str, episode: dict) -> None:
    """Print a one-line summary of a rollout produced by `rollout()`."""
    n_events = int(np.sum(episode["dose"] > 0))
    total_n = float(np.sum(episode["dose"]))
    print(
        f"[{name}] {len(episode['day'])} jours simules | "
        f"WSO final = {episode['WSO'][-1]:.1f} kg/ha | "
        f"N applique = {total_n:.0f} kg/ha en {n_events} apport(s) | "
        f"Recompense cumulee = {episode['cum_reward'][-1]:.2f}"
    )


# ----------------------------------------------------------------------------
# DQN policy: quick in-notebook training or loading a CLI-trained checkpoint
# ----------------------------------------------------------------------------


class TorchDQNPolicy(Agent):
    """Adapts a trained `rl_algs.DQN.DQN` network to the same
    `get_action(obs) -> int` interface used by `ExpertFertilizationPolicy`,
    so both policies can be run through the same `rollout()` function.
    """

    def __init__(self, q_network, device: str = "cpu") -> None:
        self.q_network = q_network
        self.device = device

    def get_action(self, obs: np.ndarray) -> int:
        import torch

        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.q_network(x), dim=-1).item())


def load_dqn_policy(args: utils.Args, agent_path: str, device: str = "cpu"):
    """Load a DQN checkpoint (`agent.pt`, produced either by `train_dqn_quick`
    or by `python3 train_agent.py --agent-type DQN ...`) and return
    `(env, policy)` ready for `rollout()`.
    """
    import torch
    from rl_algs.DQN import DQN as DQNNetwork

    envs = gym.vector.SyncVectorEnv([lambda: build_normalized_env(args)])
    q_network = DQNNetwork(envs, state_fpath=agent_path).to(device)
    q_network.eval()

    return envs.envs[0], TorchDQNPolicy(q_network, device=device)


def train_dqn_quick(args: utils.Args, total_timesteps: int = 20000, **dqn_kwargs) -> str:
    """Train a DQN agent directly with `rl_algs.DQN.train()` for a short,
    notebook-friendly number of timesteps, so the comparison notebook runs
    end-to-end without requiring a pre-trained checkpoint.

    For a properly trained research agent, prefer the CLI:
        python3 train_agent.py --agent-type DQN --env-id ln-v0 \\
            --env-reward RewardCustomFertilization --dose-kg-ha 0 40 70 80 100 \\
            --save-folder <logs/> --alg.total-timesteps 1000000
    and load the resulting `agent.pt` with `load_dqn_policy()` instead.

    Returns the path to the trained `agent.pt` checkpoint.
    """
    from rl_algs import DQN as DQN_module

    args.alg = DQN_module.Args(total_timesteps=total_timesteps, **dqn_kwargs)
    args.agent_type = "DQN"
    if not hasattr(args, "track"):
        args.track = False

    before = time.time()
    DQN_module.train(args)

    run_dirs = [d for d in glob.glob(f"{args.save_folder}DQN/*") if os.path.isdir(d) and os.path.getmtime(d) >= before - 1]
    assert run_dirs, f"Could not locate the training run directory under `{args.save_folder}DQN/`"
    latest = max(run_dirs, key=os.path.getmtime)

    return os.path.join(latest, "agent.pt")


# ----------------------------------------------------------------------------
# Publication-style comparison plots
# ----------------------------------------------------------------------------

CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e1e0d9"
TEXT_PRIMARY = "#0b0b0b"


def _color_for(name: str, index: int) -> str:
    """Pick a stable, contrasted color per policy name. `Expert` and `DQN`
    always get the same two (validated, colorblind-safe) leading hues so
    repeated calls across the three plots stay visually consistent."""
    preferred = {"Expert": CATEGORICAL_COLORS[0], "DQN": CATEGORICAL_COLORS[1]}
    return preferred.get(name, CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)])


def set_paper_style() -> None:
    """Apply a clean, high-resolution matplotlib style suitable for
    publication figures (readable fonts, sober contrasted colors, no
    chartjunk)."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#898781",
            "axes.labelcolor": TEXT_PRIMARY,
            "text.color": TEXT_PRIMARY,
            "xtick.color": MUTED_COLOR,
            "ytick.color": MUTED_COLOR,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _strip_spines(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_yield_and_fertilization(episodes: dict, save_path: Optional[str] = None) -> None:
    """Plot 1: WSO (biomass/yield) over time, with applied N doses overlaid
    as bars, one panel per policy, sharing the same y-axis scales."""
    set_paper_style()

    names = list(episodes.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    wso_max = max(ep["WSO"].max() for ep in episodes.values()) * 1.1
    dose_max = max(ep["dose"].max() for ep in episodes.values())
    dose_max = dose_max * 1.15 if dose_max > 0 else 1.0

    for i, (ax, name) in enumerate(zip(axes, names)):
        ep = episodes[name]
        color = _color_for(name, i)

        (line,) = ax.plot(ep["day"], ep["WSO"], color=color, linewidth=2.2, label="WSO (biomasse)", zorder=3)
        ax.set_xlabel("Jours après semis")
        ax.set_ylabel("WSO (kg MS / ha)")
        ax.set_ylim(0, wso_max)
        ax.set_title(name)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
        _strip_spines(ax)

        tax = ax.twinx()
        applied = ep["dose"] > 0
        bars = tax.bar(
            ep["day"][applied],
            ep["dose"][applied],
            width=3.0,
            color=MUTED_COLOR,
            alpha=0.55,
            edgecolor="none",
            label="Dose d'azote appliquée",
            zorder=2,
        )
        tax.set_ylabel("Dose d'azote (kg N / ha)", color=MUTED_COLOR)
        tax.set_ylim(0, dose_max)
        tax.tick_params(axis="y", colors=MUTED_COLOR)
        _strip_spines(tax)

        ax.legend(handles=[line, bars], labels=[line.get_label(), bars.get_label()], loc="upper left", frameon=False)

    fig.suptitle("Rendement (WSO) et apports d'azote au cours de la saison", fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_cumulative_reward(episodes: dict, save_path: Optional[str] = None) -> None:
    """Plot 2: cumulative reward over the episode, one line per policy."""
    set_paper_style()

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for i, (name, ep) in enumerate(episodes.items()):
        ax.plot(ep["day"], ep["cum_reward"], color=_color_for(name, i), linewidth=2.2, label=name)

    ax.set_xlabel("Jours après semis")
    ax.set_ylabel("Récompense cumulée")
    ax.set_title("Récompense cumulée au cours de l'épisode", fontweight="bold")
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    _strip_spines(ax)
    ax.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_nitrogen_dynamics(episodes: dict, save_path: Optional[str] = None) -> None:
    """Plot 3: soil mineral Nitrogen (NAVAIL, proxy for TNSOIL) and crop N
    uptake (NUPTAKETOTAL, proxy for NUPTT) over time, one color per policy
    and one line style per variable, on a single shared axis (same units)."""
    set_paper_style()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    handles = []
    for i, (name, ep) in enumerate(episodes.items()):
        color = _color_for(name, i)
        (h1,) = ax.plot(ep["day"], ep["NAVAIL"], color=color, linewidth=2.2, linestyle="-",
                         label=f"{name} — N disponible dans le sol")
        (h2,) = ax.plot(ep["day"], ep["NUPTAKETOTAL"], color=color, linewidth=2.2, linestyle="--",
                         label=f"{name} — N absorbé par la culture")
        handles += [h1, h2]

    ax.set_xlabel("Jours après semis")
    ax.set_ylabel("Azote (kg N / ha)")
    ax.set_title("Dynamique de l'azote : sol (NAVAIL) vs absorption (NUPTAKETOTAL)", fontweight="bold")
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    _strip_spines(ax)
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="best")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_all(episodes: dict, save_dir: Optional[str] = None) -> None:
    """Convenience wrapper generating the three comparison figures in order."""
    paths = (None, None, None)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        paths = (
            os.path.join(save_dir, "yield_fertilization.png"),
            os.path.join(save_dir, "cumulative_reward.png"),
            os.path.join(save_dir, "nitrogen_dynamics.png"),
        )
    plot_yield_and_fertilization(episodes, save_path=paths[0])
    plot_cumulative_reward(episodes, save_path=paths[1])
    plot_nitrogen_dynamics(episodes, save_path=paths[2])
