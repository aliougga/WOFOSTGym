"""
Rule-based "expert" fertilization policy, used as a baseline to compare
against the DQN agent (see rl_algs/DQN.py).

The expert applies a single, fixed Nitrogen dose on a given day after sowing
(default: day 21) and does nothing for the rest of the season. It implements
the same `Agent` interface as the DQN agent (`get_action(obs) -> action`) and
is meant to be run on the *exact same* environment/wrapper stack as the DQN
agent (same state variables, same discrete dose set, same reward function) so
that the two policies are directly comparable.

Written by: Aliou Garga, 2026

To run a standalone demo episode:
    python3 -m rl_algs.rl_dqn_expert --save-folder logs/expert/ --agro-file wheat_agro.yaml
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tyro

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from rl_algs.rl_utils import Agent


class ExpertFertilizationPolicy(Agent):
    """Simplified expert policy: one fertilization event at `application_day`
    days after sowing, at a fixed dose, and no other intervention.
    """

    def __init__(
        self,
        dose_kg_ha: list = [0, 40, 70, 80, 100],
        application_dose: float = 80,
        application_day: int = 21,
    ) -> None:
        """Initialize the :class:`ExpertFertilizationPolicy`.

        Args:
            dose_kg_ha: the discrete dose set exposed by `NPKDoseActionWrapper`.
                Must match the wrapper's configuration for the environment
                being controlled.
            application_dose: the single dose (kg N/ha) applied on
                `application_day`. Must be a member of `dose_kg_ha`.
            application_day: number of days after sowing (i.e. `DAYS` in the
                observation) on which fertilization is applied.
        """
        assert application_dose in dose_kg_ha, (
            f"`application_dose` ({application_dose}) must be one of the doses exposed by the "
            f"environment's `NPKDoseActionWrapper`: {dose_kg_ha}"
        )
        self.dose_kg_ha = list(dose_kg_ha)
        self.application_day = application_day
        self.application_action = self.dose_kg_ha.index(application_dose)
        self.null_action = self.dose_kg_ha.index(0)

    def get_action(self, obs: np.ndarray) -> int:
        """Return the fertilization action for the current observation.

        `DAYS` (days elapsed since sowing) is always the last entry of the
        raw observation vector produced by the WOFOST Gym environment,
        regardless of which `output_vars`/`weather_vars` are configured.

        Args:
            obs: the environment observation (flat array)
        """
        day = float(np.asarray(obs).reshape(-1)[-1])
        if day == self.application_day:
            return self.application_action
        return self.null_action

    def __str__(self) -> str:
        dose = self.dose_kg_ha[self.application_action]
        return f"Expert(day={self.application_day}, dose={dose}kg N/ha)"


@dataclass
class ExpertArgs(utils.Args):
    """Args for running a standalone expert demo episode"""

    env_id: str = "ln-v0"
    env_reward: Optional[str] = "RewardCustomFertilization"

    """Discrete Nitrogen dose set (kg/ha), must include 0"""
    dose_kg_ha: list = field(default_factory=lambda: [0, 40, 70, 80, 100])
    """Single dose applied on `application_day` (must be in `dose_kg_ha`)"""
    application_dose: float = 80
    """Day after sowing on which the expert fertilizes"""
    application_day: int = 21


if __name__ == "__main__":
    # Local import to avoid a hard dependency on npk_comparison_utils for
    # library users who only need `ExpertFertilizationPolicy`.
    from rl_algs.npk_comparison_utils import configure_npk_args, build_env, rollout, summarize_episode

    args = tyro.cli(ExpertArgs)
    configure_npk_args(args, dose_kg_ha=args.dose_kg_ha)
    env = build_env(args)

    expert = ExpertFertilizationPolicy(
        dose_kg_ha=args.dose_kg_ha,
        application_dose=args.application_dose,
        application_day=args.application_day,
    )

    episode = rollout(env, expert)
    summarize_episode("Expert", episode)
