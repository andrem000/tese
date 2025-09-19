"""HoldemParallelWrapper

A lightweight wrapper around PettingZoo's texas_holdem_v4 *parallel* API that
presents batched observations and action-masks similar to the SMAX wrappers.
This is **not** JAX-compiled; it is intended to smooth the transition from the
current Python AEC trainer to a fully vectorised/JIT pipeline.

Key features
------------
* `reset()` returns a dict with `observation` and `action_mask` shaped as
  (num_players, obs_dim) / (num_players, action_dim).
* `step(action_dict)` accepts the parallel-env action dict and returns the same
  batched dict along with reward / done flags suitable for GAE.
* `get_avail_actions()` mirrors the helper used in the trainers.

Once this wrapper is in place the trainer can be refactored to:
    env = HoldemParallelWrapper(num_players=3)
    obs = env.reset(seed=seed)
    action_dict = {...}
    next_obs, rewards, dones, infos = env.step(action_dict)
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
from pettingzoo.classic import texas_holdem_v4

__all__ = ["HoldemParallelWrapper"]


class HoldemParallelWrapper:
    """Batched parallel-API wrapper for Texas Hold'em.

    Parameters
    ----------
    num_players : int
        Number of poker seats (2-6 supported by PettingZoo, default 3).
    seed : int | None
        Environment RNG seed passed to ``reset``.
    **env_kwargs : Any
        Extra keyword arguments forwarded to ``texas_holdem_v4.parallel_env``.
    """

    def __init__(self, num_players: int = 3, seed: int | None = None, **env_kwargs: Any):
        self._env = texas_holdem_v4.parallel_env(num_players=num_players, **env_kwargs)
        self.seed = seed
        # Lazily reset in __init__ so that attributes like obs_dim are available immediately.
        raw_obs = self._env.reset(seed=seed)

        self.agents = self._env.possible_agents  # stable order guaranteed by PettingZoo
        self.num_players = len(self.agents)
        # Observation spec: first 72 bools are card / betting context (see PettingZoo docs)
        self.obs_dim = 72
        self.action_dim = 4  # [fold, call, raise_half, raise_pot]

        # Sanity-check dimensions once so bugs surface early
        first_vec = raw_obs[self.agents[0]]["observation"]
        assert first_vec.shape[0] == self.obs_dim, "Unexpected observation size"
        first_mask = raw_obs[self.agents[0]]["action_mask"]
        assert first_mask.shape[0] == self.action_dim, "Unexpected action-mask size"

    # ---------------------------------------------------------------------
    # Public API (mirrors JaxMARL wrappers)
    # ---------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray]:
        raw_obs = self._env.reset(seed=seed if seed is not None else self.seed)
        return self._batch_obs(raw_obs)

    def step(self, action_dict: Dict[str, int]):
        """Step the underlying env.

        Returns
        -------
        obs   : dict with batched "observation" and "action_mask" arrays
        reward: np.ndarray (num_players,)
        done  : np.ndarray (num_players,)  boolean
        info  : Dict[str, Any] keyed by agent
        """
        next_obs, rewards, terminations, truncations, infos = self._env.step(action_dict)

        # Build per-agent done mask (True if either terminated or truncated)
        done = np.asarray([terminations[a] or truncations[a] for a in self.agents], dtype=bool)
        reward_vec = np.asarray([rewards[a] for a in self.agents], dtype=np.float32)
        batched_obs = self._batch_obs(next_obs)
        return batched_obs, reward_vec, done, infos

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def get_avail_actions(obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return the legality mask from a batched observation dict."""
        return obs["action_mask"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _batch_obs(self, raw_obs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        obs_batch = np.stack([raw_obs[a]["observation"] for a in self.agents], axis=0)
        mask_batch = np.stack([raw_obs[a]["action_mask"] for a in self.agents], axis=0)
        return {"observation": obs_batch.astype(np.float32), "action_mask": mask_batch.astype(np.float32)} 