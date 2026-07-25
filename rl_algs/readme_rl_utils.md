# 📘 Analyse du fichier `rl_utils.py` — Sections utiles pour DQN

Ce fichier ne contient **pas directement l’algorithme DQN**, mais fournit toute l’infrastructure autour :
- création de l’environnement
- gestion des actions
- évaluation
- preprocessing (normalisation, reward wrapping)

👉 C’est **ESSENTIEL** pour adapter ton projet (surtout reward + action space).

---

# 🧠 1. Classe `RL_Args` (Configuration globale)

```python
@dataclass
class RL_Args:
```

## 📌 Rôle
Classe de base pour tous les algorithmes RL (dont DQN).

## 🔑 Paramètres importants

- `seed` → reproductibilité
- `cuda` → GPU
- `wandb_project_name` → logging
- `capture_video` → debug visuel

👉 ⚠️ **Aucune logique DQN ici**, uniquement configuration.

---

# 🧩 2. Classe abstraite `Agent`

```python
class Agent(ABC):
    @abstractmethod
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
```

## 📌 Rôle
Interface commune pour tous les agents (DQN, PPO, etc.)

👉 Le DQN que tu as vu implémente cette fonction :

```python
def get_action(self, x):
    return torch.argmax(self.network(x), dim=-1)
```

## 🎯 Utilité
Permet à `eval_policy()` de fonctionner avec n’importe quel agent.

---

# 🌍 3. Création de l’environnement (`make_env`) ⭐ IMPORTANT

```python
def make_env(kwargs, idx, capture_video, run_name):
```

## 🔥 C’est ICI que ton environnement est construit

```python
env = utils.make_gym_env(kwargs, run_name=run_name)
env = utils.wrap_env_reward(env, kwargs)
env = wrappers.NormalizeObservation(env)
env = wrappers.NormalizeReward(env)
```

---

## 🧠 Décomposition

### 🔹 1. `make_gym_env`
👉 crée ton environnement agricole (PCSE)

➡️ C’est là que se trouve :
- dynamique du système
- état
- actions
- reward **initiale**

---

### 🔹 2. `wrap_env_reward` ⭐ TRÈS IMPORTANT

👉 **C’est probablement ici que ta reward est modifiée**

Tu dois chercher dans :
```python
utils.wrap_env_reward
```

👉 C’est souvent utilisé pour :
- reshaping de reward
- ajout de pénalités
- scaling

---

### 🔹 3. Normalisation

```python
NormalizeObservation
NormalizeReward
```

👉 transforme :
- observations → centrées/réduites
- rewards → stabilisées

⚠️ IMPORTANT :
- ta reward réelle est transformée ici
- pour debug → utiliser `.unnormalize()`

---

# ⚠️ 4. Contrainte DQN importante

```python
assert isinstance(envs.single_action_space, gym.spaces.Discrete)
```

👉 TON ENV DOIT ÊTRE :

✅ Discret  
❌ Pas continu  
❌ Pas MultiDiscrete (sans adaptation)

---

# 🔁 5. Évaluation (`eval_policy`) ⭐

```python
def eval_policy(policy, eval_env, ...)
```

## 📌 Rôle
Tester ton agent sans bruit (exploration)

---

## 🔥 Important pour toi

```python
env = utils.wrap_env_reward(env, kwargs)
```

👉 même pipeline que train → cohérence

---

## 📊 Reward réelle

```python
avg_reward += eval_env.envs[0].unnormalize(reward)
```

👉 Très important :
- reward affichée ≠ reward brute
- elle est **dénormalisée ici**

---

# 🔄 6. Conversion des actions (`convert_action`) ⭐⭐⭐ CRUCIAL POUR TOI

```python
def convert_action(env, act: dict) -> int:
```

## 📌 Rôle
Convertit une action dictionnaire → entier (DQN)

---

## 🧠 Format attendu

```python
act = {
    "n": int,
    "p": int,
    "k": int,
    "irrig": int
}
```

👉 UNE SEULE action active à la fois

---

## ⚠️ Contraintes

```python
if len(np.nonzero(act_vals)[0]) > 1:
    raise Exception("More than one non-zero action")
```

👉 ❗ UNE seule action possible

---

## 🔥 Mapping vers entier

Le code fait :

```python
return np.array([index])
```

👉 transforme actions complexes → index discret

---

# 🚨 PROBLÈME POUR TON CAS

Tu veux :

> ❌ irriguer + fertiliser  
> ✅ seulement fertiliser  

---

## ❗ Donc tu dois modifier `convert_action`

### AVANT :

```python
act = {
    "n": ...,
    "p": ...,
    "k": ...,
    "irrig": ...
}
```

---

### APRÈS (ton cas simplifié) :

```python
act = {
    "fertilize": int
}
```

👉 OU encore plus simple :

👉 supprimer complètement `convert_action`

---

## 💡 Solution recommandée

### Option 1 (simple)

➡️ Ton env retourne directement :

```python
action_space = Discrete(N)
```

👉 et tu ignores `convert_action`

---

### Option 2 (propre)

Modifier :

```python
convert_action()
```

pour ne gérer QUE fertilisation

---

# 📦 7. Replay Buffer (chargement)

```python
def load_data_to_buffer(...)
```

## 📌 Rôle
Charger données offline dans buffer DQN

👉 utile pour imitation learning

---

## ⚠️ Impact pour toi

```python
actions = convert_action(env, actions[k])
```

👉 si tu modifies action space → modifier ici aussi

---

# 🎯 8. Où agir pour ton projet

## ✅ 1. PRIORITÉ : environnement

Dans :
```python
utils.make_gym_env
```

👉 modifier :
- action_space
- step()
- reward

---

## ✅ 2. Reward

Dans :
```python
utils.wrap_env_reward
OU
env.step()
```

---

## ✅ 3. Actions

Modifier :

- `convert_action()` ❗
OU
- supprimer complètement ce système

---

## ✅ 4. Vérifier DQN

```python
env.single_action_space.n
```

👉 ton espace doit être :
```python
Discrete(N)
```

---

# 🧠 Résumé clair

## ❌ Ce fichier ne contient PAS :
- DQN
- reward principale

## ✅ Il contient :
- création env
- transformation reward
- conversion actions
- évaluation

---

# 🚀 Conclusion

👉 Pour ton adaptation :

### 🔥 Tu dois modifier :

1. `utils.make_gym_env` → environnement
2. `wrap_env_reward` → reward
3. `convert_action` → actions

---

# 💡 Conseil expert

👉 Simplifie ton problème :

- action = quantité fertilisant
- reward = rendement - coût - pénalité

👉 Évite :
- actions multiples
- dictionnaires complexes

---

# 🚀 Si tu veux

Je peux :
- te proposer une **fonction de reward agricole optimale**
- simplifier toute ton pipeline RL (beaucoup trop complexe ici)
- t’aider à redesign ton espace d’action proprement
