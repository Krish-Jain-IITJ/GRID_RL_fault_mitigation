"""
power_grid_env.py  v3  —  Gymnasium wrapper for the C++ Power Grid (simulator.cpp)

Changes from v2:
  - Action space expanded to 4  (0=NoAction  1=Reroute  2=Throttle10%  3=Throttle20%)
  - New info fields forwarded: avg_util, total_demand, power_shed
  - Regional-overfit 24-h penalty retained
  - Continuous mode retained for curriculum training
"""

from __future__ import annotations
import json
import numpy as np
import zmq
import gymnasium as gym
from gymnasium import spaces

# ──────────────────────────────────────────────────────────────────────────────
N_SUBS          = 50
N_TIME_FEATS    = 8
ACTION_SPACE_N  = 4          # ← was 3
ZMQ_ENDPOINT    = "tcp://localhost:5556"
ZMQ_TIMEOUT_MS  = 10_000
OBS_UPPER_BOUND = 500
YEAR_HOURS      = 8_760
SAFETY_TIMEOUT  = 200_000


class PowerGridEnv(gym.Env):
    """
    4-action Gymnasium wrapper for the Power Grid DQN bridge.

    Actions
    -------
    0  NO_ACTION        — passive; reward depends on grid state
    1  REROUTE          — widest-path load balancing
    2  THROTTLE_10PCT   — shed 10% demand across all substations
    3  THROTTLE_20PCT   — shed 20% demand across all substations

    Observation
    -----------
    [edge_utils × E] [sub_demands × 50] [cyclic_time × 8]
    Shape discovered dynamically via ping on first reset().
    """

    metadata = {"render_modes": ["human"]}

    ACTION_NAMES = {0: "NO_ACTION", 1: "REROUTE", 2: "THROTTLE_10%", 3: "THROTTLE_20%"}

    def __init__(self,
                 endpoint: str = ZMQ_ENDPOINT,
                 timeout_ms: int = ZMQ_TIMEOUT_MS,
                 continuous: bool = False):
        super().__init__()
        self._endpoint   = endpoint
        self._timeout    = timeout_ms
        self.continuous  = continuous
        self._obs_dim    = None

        self._ctx    = zmq.Context()
        self._socket = None

        # 4-action discrete space
        self.action_space = spaces.Discrete(ACTION_SPACE_N)
        self._setup_spaces(OBS_UPPER_BOUND + N_SUBS + N_TIME_FEATS)

        self._step_count = 0
        self._overload_history: list[int] = []
        self._HISTORY_LEN = 24

    # ──────────────────────────────────────────────────────────────────────
    def _setup_spaces(self, obs_dim: int):
        self._obs_dim = obs_dim
        low  = np.zeros(obs_dim, dtype=np.float32)
        high = np.full(obs_dim, 2.0, dtype=np.float32)
        low [-N_TIME_FEATS:] = -1.0
        high[-N_TIME_FEATS:] =  1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    def _connect(self):
        if self._socket is not None:
            try: self._socket.close()
            except: pass
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._endpoint)

        try:
            reply   = self._send({"cmd": "ping", "action": 0})
            obs_dim = reply.get("obs_dim", self._obs_dim)
            n_rows  = reply.get("n_rows", YEAR_HOURS)
            if obs_dim != self._obs_dim:
                self._setup_spaces(obs_dim)
            print(f"[Env] Connected  obs_dim={obs_dim}  csv_rows={n_rows}")
            if n_rows < YEAR_HOURS:
                print(f"[Env] WARNING: only {n_rows} CSV rows (expected {YEAR_HOURS})")
        except zmq.error.Again:
            raise ConnectionError(f"[Env] C++ server not responding at {self._endpoint}")

    # ──────────────────────────────────────────────────────────────────────
    def _send(self, payload: dict) -> dict:
        self._socket.send(json.dumps(payload).encode())
        return json.loads(self._socket.recv().decode())

    # ──────────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()

        if not self.continuous:
            reply = self._send({"cmd": "reset", "action": 0})
            obs   = self._parse_obs(reply["obs"])
            info  = reply.get("info", {})
        else:
            reply = self._send({"cmd": "ping", "action": 0})
            obs   = self._parse_obs(reply.get("obs", []))
            info  = {"csv_row": "continuous"}

        self._step_count      = 0
        self._overload_history = []
        info["step"] = 0
        return obs, info

    # ──────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reply      = self._send({"cmd": "step", "action": int(action)})
        obs        = self._parse_obs(reply["obs"])
        reward     = float(reply["reward"])
        terminated = bool(reply["done"])
        info       = reply.get("info", {})

        self._step_count += 1
        truncated = self._step_count >= SAFETY_TIMEOUT

        # ── 24-h regional overload penalty ───────────────────────────────
        n_over = info.get("n_overloaded", 0)
        self._overload_history.append(n_over)
        if len(self._overload_history) > self._HISTORY_LEN:
            self._overload_history.pop(0)
        if (len(self._overload_history) == self._HISTORY_LEN and
                all(h > 0 for h in self._overload_history)):
            reward -= 5.0
            info["regional_overfit_penalty"] = True

        info["step"]        = self._step_count
        info["action_name"] = self.ACTION_NAMES.get(action, "?")
        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    def _parse_obs(self, raw: list[float]) -> np.ndarray:
        arr = np.array(raw, dtype=np.float32)
        if len(arr) < self._obs_dim:
            arr = np.pad(arr, (0, self._obs_dim - len(arr)))
        elif len(arr) > self._obs_dim:
            arr = arr[:self._obs_dim]
        return arr

    # ──────────────────────────────────────────────────────────────────────
    def close(self):
        if self._socket:
            try: self._socket.close()
            except: pass
        try: self._ctx.term()
        except: pass

    def render(self): pass   # Raylib window in C++

    @staticmethod
    def action_name(a: int) -> str:
        return PowerGridEnv.ACTION_NAMES.get(a, "?")
