"""
Stable Baselines 3 compatible wrapper for TileUniverse environments.

Provides a VecEnv-style wrapper that works directly with SB3's PPO, A2C, etc.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

from .gridworld import ParallelGridworld


class TileUniverseSB3VecEnv(VecEnv):
    """
    Stable Baselines 3 VecEnv wrapper for TileUniverse.

    This provides the exact interface SB3 expects for vectorized environments,
    enabling direct use with PPO, A2C, and other algorithms.

    Args:
        worlds: Number of parallel environments
        size: Grid size (width, height) as powers of 2
        wall_density: Fraction of cells that are walls
        max_steps: Maximum steps per episode
        time_penalty: Per-step reward penalty
        wall_penalty: Penalty for hitting walls
        goal_reward: Reward for reaching goal
        seed: Random seed
        channel_obs: If True, return (worlds, 4, H, W) channel observations
    """

    def __init__(
        self,
        worlds: int = 64,
        size: Tuple[int, int] = (32, 32),
        wall_density: float = 0.1,
        max_steps: int = 200,
        time_penalty: float = -0.01,
        wall_penalty: float = -0.1,
        goal_reward: float = 1.0,
        seed: Optional[int] = None,
        flatten_obs: bool = True,
    ):
        self.flatten_obs = flatten_obs
        self.width, self.height = size

        # Create underlying parallel gridworld
        self.gridworld = ParallelGridworld(
            worlds=worlds,
            size=size,
            wall_density=wall_density,
            max_steps=max_steps,
            time_penalty=time_penalty,
            wall_penalty=wall_penalty,
            goal_reward=goal_reward,
            seed=seed,
        )

        # Define spaces
        if flatten_obs:
            # Flattened 4-channel observations for MlpPolicy
            # Shape: (4 * height * width,) as float32
            observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4 * self.height * self.width,),
                dtype=np.float32,
            )
        else:
            # Raw grid for custom policies
            observation_space = spaces.Box(
                low=0,
                high=3,
                shape=(self.height, self.width),
                dtype=np.uint8,
            )

        action_space = spaces.Discrete(5)

        # Initialize VecEnv
        super().__init__(worlds, observation_space, action_space)

        # Add render_mode attribute for SB3 compatibility
        self.render_mode = None

        # Track episode stats per environment
        self._rewards = np.zeros(worlds, dtype=np.float32)
        self._lengths = np.zeros(worlds, dtype=np.int32)

    def reset(self) -> np.ndarray:
        """Reset all environments and return observations."""
        self.gridworld.reset()
        self._rewards[:] = 0
        self._lengths[:] = 0
        return self._get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        """Store actions for async step."""
        self._actions = np.asarray(actions, dtype=np.int32)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Execute stored actions and return results."""
        obs_raw, rewards, dones, info = self.gridworld.step(self._actions)

        # Update episode tracking
        self._rewards += rewards
        self._lengths += 1

        # Build per-env info dicts (SB3 expects list of dicts)
        infos: List[Dict[str, Any]] = []
        goal_reached = info.get("goal_reached", np.zeros(self.num_envs, dtype=bool))

        for i in range(self.num_envs):
            env_info: Dict[str, Any] = {}
            if dones[i]:
                # Episode finished - store terminal info
                env_info["episode"] = {
                    "r": float(self._rewards[i]),
                    "l": int(self._lengths[i]),
                }
                env_info["terminal_observation"] = self._get_single_obs(obs_raw, i)
                env_info["TimeLimit.truncated"] = not goal_reached[i]
            infos.append(env_info)

        # Auto-reset done environments
        if np.any(dones):
            self.gridworld.reset(world_mask=dones)
            self._rewards[dones] = 0
            self._lengths[dones] = 0

        # Get observations (after reset for done envs)
        obs = self._get_obs()

        return obs, rewards, dones, infos

    def _get_obs(self) -> np.ndarray:
        """Get observations for all environments."""
        if self.flatten_obs:
            # Get channel observations and flatten
            channels = self.gridworld.get_observation_channels()  # (N, 4, H, W)
            return channels.reshape(self.num_envs, -1)  # (N, 4*H*W)
        else:
            return self.gridworld.get_observations()

    def _get_single_obs(self, obs_raw: np.ndarray, idx: int) -> np.ndarray:
        """Get single environment observation (for terminal_observation)."""
        if self.flatten_obs:
            # Convert single world to flattened channels
            grid = obs_raw[idx]
            channels = np.zeros((4, self.height, self.width), dtype=np.float32)
            for c in range(4):
                channels[c] = (grid == c).astype(np.float32)
            return channels.flatten()
        else:
            return obs_raw[idx]

    def close(self) -> None:
        """Clean up resources."""
        pass

    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """Get attribute from environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        if attr_name == "gridworld":
            return [self.gridworld] * len(indices)
        raise AttributeError(f"Unknown attribute: {attr_name}")

    def set_attr(
        self, attr_name: str, value: Any, indices: Optional[List[int]] = None
    ) -> None:
        """Set attribute on environments."""
        raise NotImplementedError()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Optional[List[int]] = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call method on environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        method = getattr(self.gridworld, method_name)
        return [method(*method_args, **method_kwargs)]

    def env_is_wrapped(
        self, wrapper_class: type, indices: Optional[List[int]] = None
    ) -> List[bool]:
        """Check if environments are wrapped."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [False] * len(indices)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed."""
        if seed is not None:
            self.gridworld.rng = np.random.default_rng(seed)
        return [seed] if seed is not None else []

    def render(self, mode: str = "ascii") -> Optional[str]:
        """Render first environment."""
        if mode == "ascii":
            return self.gridworld.render_world(0)
        return None

    def get_images(self) -> List[np.ndarray]:
        """Get RGB images for all environments (for video recording)."""
        # Simple visualization: convert grid to RGB
        images = []
        grids = self.gridworld.get_observations()

        # Color map: empty=white, wall=black, goal=green, agent=blue
        colors = np.array([
            [255, 255, 255],  # empty - white
            [50, 50, 50],     # wall - dark gray
            [50, 200, 50],    # goal - green
            [50, 100, 200],   # agent - blue
        ], dtype=np.uint8)

        for grid in grids:
            # Scale up for visibility
            scale = 4
            h, w = grid.shape
            img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
            for y in range(h):
                for x in range(w):
                    cell = min(grid[y, x], 3)
                    img[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = colors[cell]
            images.append(img)

        return images
