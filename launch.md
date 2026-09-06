# Lancer WOFOSTGym sur Google Colab — Simulation DQN pour le Cameroun

Ce guide explique, étape par étape, comment lancer ce projet (WOFOSTGym) sur Google Colab,
entraîner un agent **DQN** et obtenir des résultats de simulation dans le contexte du Cameroun
(région de l'Adamaoua / Ngaoundéré), en s'appuyant sur le profil de sol déjà présent dans
`env_config/soil/cameroun.yaml` (variation `village_cameroun_soil`).

> Le dépôt est encore en développement actif (voir le disclaimer du `README.md` principal).
> Certaines commandes indiquées dans les autres README du projet sont légèrement obsolètes ;
> ce guide a été vérifié directement contre le code (`utils.py`, `pcse_gym/pcse_gym/args.py`,
> `rl_algs/DQN.py`, `data_generation/gen_data.py`).

## 0. Prérequis

- Un compte Google (pour Google Colab).
- L'URL de votre dépôt GitHub, par ex. `https://github.com/aliougga/WOFOSTGym.git`
  (adaptez si vous utilisez un autre fork/branche).
- Une connexion internet (obligatoire : la météo historique est téléchargée depuis la
  base NASA POWER lors du premier accès à une localisation/année donnée).

---

## 1. Ouvrir un notebook Google Colab

1. Aller sur https://colab.research.google.com et créer un nouveau notebook.
2. (Optionnel mais recommandé pour accélérer l'entraînement) `Exécution > Modifier le type
   d'exécution > Accélérateur matériel > GPU (T4)`. Le réseau DQN utilisé ici est petit
   (2 couches denses), donc le CPU suffit largement pour une démonstration — le GPU n'aide
   surtout que si vous augmentez fortement `total_timesteps`.
3. Vérifier la version de Python dans une cellule :

```python
!python3 --version
```

Le projet exige **Python >= 3.12**. Si Colab vous donne une version antérieure, installez
3.12 via `deadsnakes` puis utilisez `python3.12` dans toutes les commandes qui suivent :

```bash
!sudo apt-get install -y python3.12 python3.12-venv python3.12-distutils
!curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
```

---

## 2. Cloner le dépôt

```bash
%cd /content
!git clone https://github.com/aliougga/WOFOSTGym.git
%cd /content/WOFOSTGym
```

Gardez toujours `/content/WOFOSTGym` comme répertoire de travail courant : tous les chemins
(`env_config/...`, `save_folder`, etc.) du projet sont relatifs à ce répertoire.

---

## 3. Installer les dépendances

```bash
!pip install -e pcse -e pcse_gym
!pip install tyro torch omegaconf wandb tensorboard
```

Pour les use-cases d'imitation learning (GAIL/BC — non nécessaires pour DQN) :

```bash
!pip install -e imitation -e stable-baselines3
!pip install tqdm huggingface_sb3
```

---

## 4. Vérifier l'installation

```bash
!python3 test_wofost.py --save-folder test/ --data-file test
```

Le premier lancement peut prendre quelques secondes de plus : le projet configure le
répertoire de cache météo (NASA POWER). Si cette commande se termine sans erreur,
l'installation est correcte.

Vous pouvez aussi inspecter les options disponibles directement dans votre environnement
Colab (utile car les noms d'options générés par `tyro` peuvent varier légèrement selon la
version installée) :

```bash
!python3 train_agent.py --help
!python3 data_generation/gen_data.py --help
```

---

## 5. Le contexte Cameroun : sol et cultures déjà disponibles

Le dépôt contient déjà un profil de sol pour le Cameroun :
`env_config/soil/cameroun.yaml`, enregistré sous le nom `cameroun` avec la variation
`village_cameroun_soil` (données pour un sol type de Ngaoundéré / Adamaoua — à ajuster
si vous disposez de vraies données pédologiques).

Des cultures pertinentes pour le Cameroun sont déjà configurées dans `env_config/crop/` et
`env_config/agro/` : `maize` (maïs), `millet` (mil), `sorghum` (sorgho), ainsi que
`cassava`, `cowpea`, `groundnut` dans `env_config/crop/crops.yaml`.

Tous les paramètres de localisation/sol/culture (`latitude`, `longitude`, `year`,
`soil_name`, `soil_variation`, `crop_name`, `crop_variety`, ...) peuvent être surchargés
en ligne de commande via le préfixe `--npk.ag.*`, sans avoir à créer de nouveau fichier
YAML — ils remplacent alors les valeurs du fichier `agro_file` de base
(`pcse_gym/pcse_gym/utils.py::set_agro_params`). C'est l'approche utilisée ci-dessous en
partant de `maize_agro.yaml` (dates de semis déjà proches de la saison des pluies
d'avril-septembre à l'Adamaoua).

Coordonnées utilisées pour Ngaoundéré : latitude `7.32`, longitude `13.58`. L'année doit
être comprise entre 1984 et 2018 (2015 utilisé ci-dessous, sans données manquantes).

---

## 6. Entraîner l'agent DQN sur la configuration Cameroun

```bash
!python3 train_agent.py \
  --agent-type DQN \
  --save-folder logs/dqn_cameroun/ \
  --agro-file maize_agro.yaml \
  --npk.ag.latitude 7.32 \
  --npk.ag.longitude 13.58 \
  --npk.ag.year 2015 \
  --npk.ag.soil-name cameroun \
  --npk.ag.soil-variation village_cameroun_soil
```

Notes :

- `--save-folder` doit se terminer par `/`.
- Par défaut, `rl_algs/DQN.py` entraîne pendant `total_timesteps = 1_000_000`, ce qui est
  long pour une simple démonstration Colab. Pour un run rapide (quelques minutes),
  réduisez cette valeur avant de lancer l'entraînement, par exemple en éditant
  temporairement `rl_algs/DQN.py` :

```python
!sed -i 's/total_timesteps: int = 1000000/total_timesteps: int = 20000/' rl_algs/DQN.py
```

(remettez `1000000`, ou une valeur plus réaliste, pour un entraînement complet donnant
de meilleurs résultats).

- Pour suivre l'expérience avec Weights & Biases, ajoutez `--track` (nécessite
  `!wandb login` au préalable). Ce n'est pas obligatoire : les métriques sont de toute
  façon écrites en local via TensorBoard (voir étape suivante).
- L'entraînement crée un dossier `logs/dqn_cameroun/DQN/<run_name>/` contenant
  `agent.pt` (poids du réseau, sauvegardés périodiquement) et les logs TensorBoard, ainsi
  qu'un `config.yaml` à la racine de `logs/dqn_cameroun/` qui fige la configuration
  complète (sol, culture, localisation) utilisée pour cet entraînement.

---

## 7. Suivre l'entraînement avec TensorBoard (dans Colab)

Dans une cellule séparée, pendant ou après l'entraînement :

```python
%load_ext tensorboard
%tensorboard --logdir logs/dqn_cameroun/DQN
```

Vous y verrez `charts/episodic_return`, `charts/average_reward`, `losses/td_loss`, etc.

---

## 8. Générer une simulation avec l'agent DQN entraîné

Une fois l'entraînement terminé (ou interrompu après quelques checkpoints — `agent.pt`
est sauvegardé régulièrement), utilisez `data_generation/gen_data.py` pour rejouer
l'agent sur l'environnement et enregistrer sa trajectoire (observations, actions,
récompenses) au format `.npz`.

Repérez d'abord le chemin exact de `agent.pt` :

```bash
!find logs/dqn_cameroun -name "agent.pt"
```

Puis lancez la génération de données, en pointant `--config-fpath` vers le
`config.yaml` sauvegardé à l'entraînement (il contient déjà le sol et la culture du
Cameroun) et en restreignant la boucle année/latitude/longitude à un seul point
(Ngaoundéré, année 2015) :

```bash
!python3 data_generation/gen_data.py \
  --save-folder results/dqn_cameroun/ \
  --data-file simulation_cameroun \
  --file-type npz \
  --agent-type DQN \
  --agent-path logs/dqn_cameroun/DQN/<run_name>/agent.pt \
  --config-fpath logs/dqn_cameroun/config.yaml \
  --year-low 2015 --year-high 2015 \
  --lat-low 7.32 --lat-high 7.32 \
  --lon-low 13.58 --lon-high 13.58
```

Remplacez `<run_name>` par le nom réel du dossier trouvé à l'étape précédente (il contient
un timestamp, par ex. `lnpkw-v0__DQN__1__1732400000`).

Ceci produit `results/dqn_cameroun/simulation_cameroun.npz`.

---

## 9. Visualiser les résultats de la simulation

> **Important : lancez la visualisation dans une cellule Python du notebook, pas via
> `!python3 ...`.** Une commande précédée de `!` s'exécute dans un **sous-processus**
> séparé du noyau (kernel) du notebook : `plt.show()` n'y affiche rien à l'écran, la
> figure est seulement écrite en `.png` sur disque — d'où l'impression que les
> graphiques ne sont que "téléchargeables" et jamais visibles directement. Pour un
> affichage inline réel dans Colab, appelez les fonctions Python directement dans une
> cellule (import, pas `!python3`) : c'est le même noyau que celui qui écrit dans le
> notebook, donc chaque figure s'affiche automatiquement sous la cellule.

Pour les résultats liés à l'article (comparaison DQN vs politique experte, et surtout
la **validation de la fonction de récompense**), utilisez directement
`rl_algs/npk_comparison_utils.py`, qui produit des figures "publication" stylées et
valide numériquement l'implémentation de $R_t$ contre l'équation de l'article :

```python
import utils
from rl_algs import npk_comparison_utils as ncu

args = utils.Args(npk=None, env_id="ln-v0", save_folder="results/dqn_cameroun/",
                   agro_file="maize_agro.yaml", env_reward="RewardCustomFertilization")
ncu.configure_npk_args(args, dose_kg_ha=[0, 40, 70, 80, 100])  # remplit args.npk
args.npk.ag.latitude, args.npk.ag.longitude, args.npk.ag.year = 7.32, 13.58, 2015
args.npk.ag.soil_name, args.npk.ag.soil_variation = "cameroun", "village_cameroun_soil"

env, policy = ncu.load_dqn_policy(args, "logs/dqn_cameroun/DQN/<run_name>/agent.pt")
episode = ncu.rollout(env, policy)
ncu.summarize_episode("DQN", episode)

# Les 3 figures de comparaison + la figure de validation de la récompense (terme
# par terme, avec l'erreur max entre l'implémentation et l'équation de l'article)
# s'affichent directement sous la cellule, ET sont enregistrées en .png si
# `save_dir` est fourni :
ncu.plot_all({"DQN": episode}, args, save_dir="figures/dqn_cameroun/")
```

Pour une inspection rapide et générique de n'importe quelle variable d'un `.npz`
(sans passer par le pipeline article), la même règle "cellule Python, pas `!python3`"
s'applique à `data_plotting.vis_data` :

```python
import utils
import data_plotting.vis_data as vis_data

args = ...  # le PlotArgs utilisé pour la simulation, ou reconstruit comme au step 6
obs, actions, rewards, next_obs, dones, output_vars = utils.load_data_file(
    "results/dqn_cameroun/simulation_cameroun.npz"
)
vis_data.plot_output(args, output_vars=output_vars, obs=obs, rewards=rewards, save=True)
```

Ou, sans dépendance au reste du projet, en chargeant le `.npz` à la main :

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load("results/dqn_cameroun/simulation_cameroun.npz", allow_pickle=True)
obs, output_vars = data["obs"], list(data["output_vars"])

plt.figure(figsize=(8, 5))
plt.plot(obs[:, output_vars.index("WSO")])
plt.xlabel("Jours écoulés")
plt.ylabel("WSO (biomasse des organes de réserve, kg/ha)")
plt.title("Simulation DQN — maïs, Ngaoundéré, Cameroun (2015)")
plt.show()
```

Dans tous les cas, les figures restent aussi enregistrées en `.png` (paramètres
`save_dir`/`fig_folder`/`save`) pour l'article — l'affichage inline et
l'enregistrement sur disque ne s'excluent pas.

---

## 10. Récupérer les résultats en dehors de Colab

```python
from google.colab import files
!zip -r resultats_cameroun.zip results/dqn_cameroun figures/dqn_cameroun logs/dqn_cameroun/config.yaml
files.download("resultats_cameroun.zip")
```

(Alternative : monter Google Drive avec `from google.colab import drive; drive.mount('/content/drive')`
puis copier les dossiers avec `!cp -r ...` pour conserver les résultats entre les sessions.)

---

## Dépannage rapide

- **`AssertionError` sur `save_folder`** : vérifiez que tous les `--save-folder` /
  `--fig-folder` se terminent bien par `/`.
- **Erreur de latitude/longitude/année hors bornes** : l'année doit être dans
  `[1984, 2019]` (voire `2023` selon la classe d'environnement) et la latitude/longitude
  dans des bornes physiques classiques `(-90,90)` / `(-180,180)`.
- **Flags `--npk.ag.*` non reconnus** : la génération des options CLI dépend de la version
  de `tyro` installée ; lancez `!python3 train_agent.py --help` pour confirmer l'orthographe
  exacte des options avant de relancer l'entraînement.
- **Session Colab déconnectée pendant un entraînement long** : réduisez
  `total_timesteps` (étape 6) ou sauvegardez régulièrement sur Google Drive ; les
  checkpoints `agent.pt` permettent de reprendre une génération de données même si
  l'entraînement complet n'est pas allé à son terme.