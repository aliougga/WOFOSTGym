# 📘 Explication complète du code : Entraînement d’un agent DQN

Ce code implémente un agent de **Deep Q-Network (DQN)**, un algorithme d’apprentissage par renforcement (Reinforcement Learning). Il permet à un agent d’apprendre à prendre des décisions optimales dans un environnement.

---

# 🧠 1. Notions clés utilisées

## 🔹 Reinforcement Learning (RL)
Le RL est un cadre où un agent apprend en interagissant avec un environnement :
- **Observation (state)** : ce que voit l’agent
- **Action** : ce que fait l’agent
- **Reward (récompense)** : feedback de l’environnement
- **Policy** : stratégie de décision

👉 Utilité : apprendre des comportements optimaux sans supervision.

---

## 🔹 Deep Q-Network (DQN)
Le DQN combine :
- **Q-Learning** (table de valeurs Q)
- **Réseaux de neurones** (approximation des Q-values)

👉 Il apprend une fonction :

Q(s, a)

qui estime la qualité d’une action dans un état.

👉 Utilité : gérer des espaces d’état complexes (images, vecteurs…).

---

## 🔹 Replay Buffer
Mémoire contenant les expériences passées :

(state, action, reward, next_state, done)

👉 Utilité :
- casse la corrélation des données
- améliore la stabilité de l’apprentissage

---

## 🔹 Target Network
Un second réseau utilisé pour stabiliser l’apprentissage.

👉 Utilité :
- évite que la cible change trop vite
- réduit les oscillations

---

## 🔹 Exploration vs Exploitation
- **Exploration** : essayer des actions aléatoires
- **Exploitation** : utiliser ce qu’on a appris

👉 Implémenté via **epsilon-greedy**

---

# ⚙️ 2. Importations

```python
import random
from argparse import Namespace
import time
from dataclasses import dataclass
import wandb
```

- `random` : exploration aléatoire
- `Namespace` : gestion des paramètres
- `time` : timing
- `dataclass` : structurer les arguments
- `wandb` : suivi des expériences

---

```python
import gymnasium as gym
import numpy as np
import torch
```

- `gymnasium` : environnement RL
- `numpy` : calcul numérique
- `torch` : deep learning

---

# 🧾 3. Classe Args (Hyperparamètres)

```python
@dataclass
class Args(RL_Args):
```

Cette classe contient les hyperparamètres du modèle.

## Paramètres importants :

- `total_timesteps` : durée d'entraînement
- `learning_rate` : vitesse d’apprentissage
- `buffer_size` : taille mémoire
- `gamma` : facteur de discount (importance du futur)
- `batch_size` : taille du batch
- `start_e`, `end_e` : epsilon initial/final
- `learning_starts` : début apprentissage
- `train_frequency` : fréquence d’entraînement

👉 Utilité : contrôler le comportement de l’algorithme.

---

# 🧠 4. Classe DQN (le modèle)

```python
class DQN(nn.Module, Agent):
```

C’est le réseau de neurones.

## Architecture :

```python
nn.Linear(input, 120)
ReLU
nn.Linear(120, 84)
ReLU
nn.Linear(84, nb_actions)
```

👉 C’est un **MLP (Multi-Layer Perceptron)**

## Forward pass :

```python
def forward(self, x):
    return self.network(x)
```

👉 Transforme un état en Q-values

## Action :

```python
torch.argmax(...)
```

👉 choisit l’action avec la meilleure Q-value

---

# 📉 5. Epsilon Schedule

```python
def linear_schedule(...)
```

Diminue progressivement epsilon :

epsilon_t = max(start + slope * t, end)

👉 Utilité :
- beaucoup explorer au début
- exploiter à la fin

---

# 🚀 6. Fonction principale : train()

## 🔹 Initialisation

```python
q_network = DQN(envs)
target_network = DQN(envs)
```

- `q_network` : réseau principal
- `target_network` : réseau stable

---

```python
optimizer = optim.Adam(...)
```

👉 Optimiseur pour ajuster les poids

---

```python
rb = ReplayBuffer(...)
```

👉 Création de la mémoire

---

# 🔁 7. Boucle d'entraînement

```python
for global_step in range(args.total_timesteps):
```

Chaque itération = 1 interaction avec l’environnement

---

## 🎲 7.1 Choix d’action

```python
if random.random() < epsilon:
```

- exploration aléatoire

```python
else:
    q_values = q_network(...)
```

- exploitation via réseau

---

## 🌍 7.2 Interaction environnement

```python
next_obs, rewards, terminations, truncations, infos = envs.step(actions)
```

👉 L’agent agit et reçoit :
- nouvel état
- récompense
- fin d’épisode

---

## 💾 7.3 Stockage

```python
rb.add(...)
```

👉 sauvegarde de l’expérience

---

## 🧠 7.4 Apprentissage

```python
if global_step > args.learning_starts:
```

On attend avant de commencer à apprendre

---

### 🔹 Sampling

```python
data = rb.sample(batch_size)
```

👉 échantillon aléatoire

---

### 🔹 Target Q-value

```python
target_max = target_network(...).max()
```

y = r + gamma * max Q_target(s', a')

---

### 🔹 Loss

```python
loss = MSE(td_target, old_val)
```

👉 minimise l’erreur entre :
- prédiction
- cible

---

### 🔹 Backpropagation

```python
loss.backward()
optimizer.step()
```

👉 mise à jour des poids

---

## 🔁 7.5 Mise à jour du target network

```python
target_network_param.data.copy_(...)
```

👉 Soft update :

theta_target = tau * theta + (1 - tau) * theta_target

---

# 📊 8. Logging et monitoring

```python
writer.add_scalar(...)
wandb.save(...)
```

👉 Permet de suivre :
- reward
- loss
- performance

---

# 💾 9. Checkpoints

```python
torch.save(...)
```

👉 sauvegarde du modèle régulièrement

---

# 🧪 10. Évaluation

```python
eval_policy(...)
```

👉 teste les performances du modèle

---

# 🧩 11. Points importants à retenir

## ✅ Avantages du DQN
- fonctionne avec grands espaces d’état
- stable grâce à :
  - replay buffer
  - target network

## ⚠️ Limites
- sensible aux hyperparamètres
- instable sans bonnes pratiques
- inefficace pour actions continues

---

# 🎯 Conclusion

Ce code implémente un pipeline complet de DQN :

1. Initialisation
2. Interaction avec l’environnement
3. Stockage des expériences
4. Apprentissage via mini-batch
5. Mise à jour du réseau cible
6. Logging et sauvegarde

👉 Il suit les bonnes pratiques modernes du RL.

---

# 🚀 Bonus : Résumé rapide

- DQN = Q-learning + réseau de neurones
- Replay Buffer = mémoire d’expérience
- Target Network = stabilité
- Epsilon-greedy = exploration
- Loss = erreur TD

---

Si tu veux, je peux aussi :
- te faire un schéma visuel du DQN
- simplifier ce code version débutant
- ou l’adapter pour un projet concret (jeu, robot, trading…)