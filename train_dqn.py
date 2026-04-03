"""
train_dqn.py  v3  —  DQN for 4-action Power Grid

New reward structure requires:
  - n_actions = 4
  - Logging of power_shed, avg_util, throttle usage counts
  - Everything else (Dueling DQN, PER, Double DQN, soft target) unchanged
"""

from __future__ import annotations
import argparse, os, random, time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from power_grid_env import PowerGridEnv, N_TIME_FEATS
#---------------------------------hyperparameters---------------------------------
def objective(trial):
    # Suggest hyperparameters
    HP.hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=128)
    HP.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # HP.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    # HP.target_update_tau = trial.suggest_float("tau", 0.001, 0.01)

    try:
        reward = train(n_episodes=1)   # small for speed
        return reward
    except Exception as e:
        print("Trial failed:", e)
        return -1000.0
    

# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class HParams:
    hidden_dim:        int   = 256
    n_layers:          int   = 3
    lr:                float = 1e-4
    gamma:             float = 0.99
    batch_size:        int   = 128
    replay_capacity:   int   = 50_000
    min_replay:        int   = 1_000
    target_update_tau: float = 0.005
    train_freq:        int   = 4
    eps_start:         float = 1.0
    eps_end:           float = 0.05
    eps_decay_steps:   int   = 20_000
    temporal_dim:      int   = 32
    per_alpha:         float = 0.6
    per_beta_start:    float = 0.4
    per_beta_end:      float = 1.0
    per_beta_steps:    int   = 30_000
    per_eps:           float = 1e-6
    seed:              int   = 42
    save_dir:          str   = "checkpoints"
    log_freq:          int   = 1          # log every episode

HP = HParams()

# ──────────────────────────────────────────────────────────────────────────────
#  SumTree
# ──────────────────────────────────────────────────────────────────────────────
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree  = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data  = np.empty(capacity, dtype=object)
        self._ptr  = 0; self._full = False

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent: self._propagate(parent, delta)

    def _leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # If we've reached the leaf nodes
            if left_child_idx >= len(self.tree):
                return parent_idx

            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

    @property
    def total(self): return self.tree[0]
    @property
    def size(self): return self.capacity if self._full else self._ptr

    def add(self, priority, data):
        idx = self._ptr + self.capacity - 1
        self.data[self._ptr] = data
        self.update(idx, priority)
        self._ptr = (self._ptr + 1) % self.capacity
        if self._ptr == 0: self._full = True

    def update(self, idx, priority):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def sample(self, n):
        idxs, data_list, prios = [], [], []
        segment = self.total / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            v = random.uniform(a, b)
            idx = self._leaf(v)
            
            # The actual index in the data array
            data_idx = idx - self.capacity + 1
            
            idxs.append(idx)
            data_list.append(self.data[data_idx])
            prios.append(self.tree[idx])
            
        return idxs, data_list, np.array(prios, dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────────
#  PER buffer
# ──────────────────────────────────────────────────────────────────────────────
class Transition:
    __slots__ = ('obs','action','reward','next_obs','done')
    def __init__(self, obs, action, reward, next_obs, done):
        self.obs=obs; self.action=action; self.reward=reward
        self.next_obs=next_obs; self.done=done

class PrioritisedReplayBuffer:
    def __init__(self, capacity, alpha):
        self._tree  = SumTree(capacity)
        self._alpha = alpha
        self._max_p = 1.0

    def push(self, t: Transition):
        self._tree.add(self._max_p ** self._alpha, t)

    def sample(self, batch_size, beta):
        idxs, data, prios = self._tree.sample(batch_size)
        N     = self._tree.size
        probs = prios / self._tree.total
        w     = (N * probs) ** (-beta)
        w    /= w.max()
        return data, idxs, w.astype(np.float32)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(err) + HP.per_eps) ** self._alpha
            self._max_p = max(self._max_p, p)
            self._tree.update(idx, p)

    def __len__(self): return self._tree.size


# ──────────────────────────────────────────────────────────────────────────────
#  DuelingDQN  (n_actions=4)
# ──────────────────────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        grid_dim = obs_dim - N_TIME_FEATS

        layers = []
        d = grid_dim
        for _ in range(HP.n_layers):
            layers += [nn.Linear(d, HP.hidden_dim), nn.LayerNorm(HP.hidden_dim), nn.SiLU()]
            d = HP.hidden_dim
        self.grid_tower = nn.Sequential(*layers)

        self.temporal_tower = nn.Sequential(
            nn.Linear(N_TIME_FEATS, HP.temporal_dim), nn.SiLU(),
            nn.Linear(HP.temporal_dim, HP.temporal_dim), nn.SiLU(),
        )

        combined = HP.hidden_dim + HP.temporal_dim
        self.value_head = nn.Sequential(
            nn.Linear(combined, HP.hidden_dim // 2), nn.SiLU(),
            nn.Linear(HP.hidden_dim // 2, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(combined, HP.hidden_dim // 2), nn.SiLU(),
            nn.Linear(HP.hidden_dim // 2, n_actions),
        )

    def forward(self, x):
        g = self.grid_tower(x[:, :-N_TIME_FEATS])
        t = self.temporal_tower(x[:, -N_TIME_FEATS:])
        h = torch.cat([g, t], dim=-1)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + a - a.mean(dim=-1, keepdim=True)


# ──────────────────────────────────────────────────────────────────────────────
#  DQN Agent
# ──────────────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, obs_dim, n_actions, device):
        self.device    = device
        self.n_actions = n_actions
        self.online = DuelingDQN(obs_dim, n_actions).to(device)
        self.target = DuelingDQN(obs_dim, n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optim  = optim.Adam(self.online.parameters(), lr=HP.lr)
        self.buffer = PrioritisedReplayBuffer(HP.replay_capacity, HP.per_alpha)
        self._step  = 0

    def epsilon(self):
        p = min(self._step / HP.eps_decay_steps, 1.0)
        return HP.eps_end + (HP.eps_start - HP.eps_end) * (1.0 - p)

    def _beta(self):
        p = min(self._step / HP.per_beta_steps, 1.0)
        return HP.per_beta_start + (HP.per_beta_end - HP.per_beta_start) * p

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(t).argmax(1).item())

    def learn(self) -> Optional[float]:
        if len(self.buffer) < HP.min_replay: return None
        if self._step % HP.train_freq != 0:  return None

        transitions, idxs, weights = self.buffer.sample(HP.batch_size, self._beta())
        w_t    = torch.tensor(weights, dtype=torch.float32, device=self.device)
        obs_b  = torch.tensor(np.stack([t.obs      for t in transitions]), dtype=torch.float32, device=self.device)
        next_b = torch.tensor(np.stack([t.next_obs for t in transitions]), dtype=torch.float32, device=self.device)
        act_b  = torch.tensor([t.action  for t in transitions], dtype=torch.long,    device=self.device)
        rew_b  = torch.tensor([t.reward  for t in transitions], dtype=torch.float32, device=self.device)
        done_b = torch.tensor([t.done    for t in transitions], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_acts = self.online(next_b).argmax(1)
            next_q    = self.target(next_b).gather(1, next_acts.unsqueeze(1)).squeeze(1)
            target_q  = rew_b + HP.gamma * next_q * (1 - done_b)

        current_q = self.online(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
        td_errors  = (current_q - target_q).detach().cpu().numpy()
        loss       = (w_t * nn.functional.huber_loss(current_q, target_q, reduction='none')).mean()

        self.optim.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optim.step()
        self.buffer.update_priorities(idxs, td_errors)

        for po, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.data.mul_(1 - HP.target_update_tau).add_(HP.target_update_tau * po.data)

        return float(loss.item())

    def push(self, t: Transition):
        self.buffer.push(t); self._step += 1

    def save(self, path):
        torch.save({"online": self.online.state_dict(),
                    "target": self.target.state_dict(),
                    "optim":  self.optim.state_dict(),
                    "step":   self._step}, path)
        print(f"[Save] {path}")

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])
        self.optim.load_state_dict(ck["optim"])
        self._step = ck["step"]
        print(f"[Load] {path}  step={self._step}")


# ──────────────────────────────────────────────────────────────────────────────
#  Logger — tracks all 4 actions + new reward fields
# ──────────────────────────────────────────────────────────────────────────────
class MetricsLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, f"run_{int(time.time())}.csv")
        with open(self._path, "w") as f:
            f.write("episode,total_reward,steps,overloads_total,max_severity,"
                    "avg_util_mean,total_shed,acts_0,acts_1,acts_2,acts_3,"
                    "eps,loss_mean\n")

    def log(self, ep, reward, steps, overloads, max_sev, avg_util,
            total_shed, act_counts, eps, loss_mean):
        row = (f"{ep},{reward:.2f},{steps},{overloads},{max_sev:.4f},"
               f"{avg_util:.4f},{total_shed:.1f},"
               f"{act_counts[0]},{act_counts[1]},{act_counts[2]},{act_counts[3]},"
               f"{eps:.4f},{loss_mean:.6f}\n")
        with open(self._path, "a") as f: f.write(row)
        print(f"[Ep {ep:4d}] R={reward:8.1f}  steps={steps}  "
              f"ovrld={overloads}  maxSev={max_sev:.2f}  "
              f"avgUtil={avg_util:.2f}  shed={total_shed:.0f}  "
              f"acts=[{act_counts[0]},{act_counts[1]},{act_counts[2]},{act_counts[3]}]  "
              f"ε={eps:.3f}  loss={loss_mean:.5f}")


# ──────────────────────────────────────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def train(n_episodes=200, resume=None, eval_only=False):
    random.seed(HP.seed); np.random.seed(HP.seed); torch.manual_seed(HP.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    env = PowerGridEnv()
    print("[Train] Connecting to C++ sim — ensure simulator is running…")
    obs, info = env.reset()
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n   # 4
    print(f"[Train] obs_dim={obs_dim}  n_actions={n_actions}")

    agent  = DQNAgent(obs_dim, n_actions, device)
    logger = MetricsLogger()
    os.makedirs(HP.save_dir, exist_ok=True)
    if resume: agent.load(resume)

    best_reward = -float("inf")

    for episode in range(1, n_episodes + 1):
        obs, info    = env.reset()
        ep_reward    = 0.0
        ep_overloads = 0
        ep_max_sev   = 0.0
        ep_avg_util  = 0.0
        ep_shed      = 0.0
        ep_losses: List[float] = []
        act_counts   = [0, 0, 0, 0]
        done         = False

        while not done:
            action = agent.select_action(obs) if not eval_only else 0
            act_counts[action] += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if not eval_only:
                agent.push(Transition(obs, action, reward, next_obs, done))
                loss = agent.learn()
                if loss is not None: ep_losses.append(loss)

            ep_reward    += reward
            ep_overloads += info.get("n_overloaded", 0)
            ep_max_sev    = max(ep_max_sev,  info.get("max_severity", 0.0))
            ep_avg_util  += info.get("avg_util", 0.0)
            ep_shed      += info.get("power_shed", 0.0)

            # Monthly progress print
            csv_row = info.get("csv_row", 0)
            if csv_row > 0 and csv_row % 720 == 0:
                month = min(csv_row // 720 + 1, 12)
                print(f"  [Ep {episode}] Month {month:2d}/12  row {csv_row}/8759  "
                      f"R={ep_reward:8.1f}  ovrld={ep_overloads}  "
                      f"shed={ep_shed:.0f}  ε={agent.epsilon():.3f}  "
                      f"action={PowerGridEnv.action_name(action)}")

            obs = next_obs

        steps     = env._step_count
        loss_mean = np.mean(ep_losses) if ep_losses else 0.0
        avg_util  = ep_avg_util / max(steps, 1)

        if episode % HP.log_freq == 0:
            logger.log(episode, ep_reward, steps, ep_overloads,
                       ep_max_sev, avg_util, ep_shed,
                       act_counts, agent.epsilon(), loss_mean)

        if ep_reward > best_reward and not eval_only:
            best_reward = ep_reward
            agent.save(os.path.join(HP.save_dir, "best.pt"))

        if episode % 50 == 0 and not eval_only:
            agent.save(os.path.join(HP.save_dir, f"ep{episode:05d}.pt"))

    env.close()
    print("[Train] Done.")
    return best_reward


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--optuna", action="store_true")  # NEW FLAG
    args = p.parse_args()

    if args.optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        print("\n🔥 BEST HYPERPARAMETERS:")
        print(study.best_params)
        print("Best reward:", study.best_value)

    else:
        train(n_episodes=args.episodes)