"""
TileUniverseVecEnv - Gymnasium-compatible vectorized environment wrapper.

Wraps ParallelGridworld to provide a standard RL interface compatible with
Stable Baselines 3 and other RL libraries.

Example:
    >>> from tileuniverse.rl import TileUniverseVecEnv
    >>> env = TileUniverseVecEnv(worlds=64, size=(32, 32))
    >>> obs, info = env.reset()
    >>> obs, rewards, terminated, truncated, info = env.step(actions)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = False
    except ImportError:
        raise ImportError(
            "Neither gymnasium nor gym is installed. "
            "Install with: pip install gymnasium"
        )

from .gridworld import ParallelGridworld


class TileUniverseVecEnv(gym.Env):
    """
    A Gymnasium-compatible vectorized environment backed by TileUniverse.

    This environment runs N parallel gridworlds simultaneously, making it
    ideal for high-throughput RL training. Each world has:
    - One agent navigating a grid
    - One goal to reach
    - Random wall obstacles

    Compatible with Stable Baselines 3 and similar libraries.

    Args:
        worlds: Number of parallel environments (default: 64)
        size: Grid size (width, height) as powers of 2 (default: (32, 32))
        wall_density: Fraction of cells that are walls (default: 0.1)
        max_steps: Maximum steps per episode (default: 200)
        time_penalty: Per-step reward penalty (default: -0.01)
        wall_penalty: Penalty for hitting walls (default: -0.1)
        goal_reward: Reward for reaching goal (default: 1.0)
        seed: Random seed (default: None)
        flatten_obs: If True, flatten observations to 1D (default: False)

    Observation Space:
        Box(0, 3, shape=(worlds, height, width), dtype=uint8)
        or if flatten_obs=True:
        Box(0, 3, shape=(worlds, height*width), dtype=uint8)

        Cell values: 0=empty, 1=wall, 2=goal, 3=agent

    Action Space:
        MultiDiscrete([5] * worlds) - one action per world
        Actions: 0=no-op, 1=up, 2=down, 3=left, 4=right
    """

    metadata = {"render_modes": ["ascii", "rgb_array"], "render_fps": 10}

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
        flatten_obs: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.num_envs = worlds  # SB3 convention
        self.flatten_obs = flatten_obs
        self.render_mode = render_mode

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

        width, height = size

        # Define observation space
        if flatten_obs:
            self.observation_space = spaces.Box(
                low=0,
                high=3,
                shape=(worlds, height * width),
                dtype=np.uint8,
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=3,
                shape=(worlds, height, width),
                dtype=np.uint8,
            )

        # Define action space - one action per world
        # Using MultiDiscrete for vectorized envs
        self.action_space = spaces.MultiDiscrete([5] * worlds)

        # For single-world access (some libs expect this)
        self.single_observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(height, width) if not flatten_obs else (height * width,),
            dtype=np.uint8,
        )
        self.single_action_space = spaces.Discrete(5)

        # Track episode statistics
        self._episode_rewards = np.zeros(worlds, dtype=np.float32)
        self._episode_lengths = np.zeros(worlds, dtype=np.int32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset all environments.

        Args:
            seed: Optional random seed
            options: Optional dict with 'world_mask' to reset specific worlds

        Returns:
            observations: Shape (worlds, height, width) or flattened
            info: Dict with additional info
        """
        super().reset(seed=seed)

        # Handle partial reset via options
        world_mask = None
        if options is not None:
            world_mask = options.get("world_mask", None)

        obs = self.gridworld.reset(world_mask=world_mask)

        # Reset episode stats for reset worlds
        if world_mask is None:
            self._episode_rewards[:] = 0
            self._episode_lengths[:] = 0
        else:
            self._episode_rewards[world_mask] = 0
            self._episode_lengths[world_mask] = 0

        if self.flatten_obs:
            obs = obs.reshape(self.num_envs, -1)

        info = self._get_info()
        return obs, info

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Take a step in all environments.

        Args:
            actions: Shape (worlds,) with action indices 0-4

        Returns:
            observations: Shape (worlds, height, width) or flattened
            rewards: Shape (worlds,) float32
            terminated: Shape (worlds,) bool - True if goal reached
            truncated: Shape (worlds,) bool - True if max steps exceeded
            info: Dict with additional info
        """
        actions = np.asarray(actions, dtype=np.int32)

        obs, rewards, dones, step_info = self.gridworld.step(actions)

        # Update episode stats
        self._episode_rewards += rewards
        self._episode_lengths += 1

        # Determine terminated vs truncated
        goal_reached = step_info.get("goal_reached", np.zeros(self.num_envs, dtype=bool))
        terminated = goal_reached
        truncated = dones & ~terminated  # Done but not from goal = truncated

        if self.flatten_obs:
            obs = obs.reshape(self.num_envs, -1)

        # Auto-reset completed environments (SB3 convention)
        if np.any(dones):
            # Store final episode info before reset
            done_mask = dones
            final_rewards = self._episode_rewards[done_mask].copy()
            final_lengths = self._episode_lengths[done_mask].copy()

            # Reset done worlds
            self.gridworld.reset(world_mask=done_mask)
            new_obs = self.gridworld.get_observations()
            if self.flatten_obs:
                new_obs = new_obs.reshape(self.num_envs, -1)

            # Update observations for reset worlds
            obs[done_mask] = new_obs[done_mask]

            # Reset episode stats
            self._episode_rewards[done_mask] = 0
            self._episode_lengths[done_mask] = 0
        else:
            final_rewards = None
            final_lengths = None

        info = self._get_info()
        if final_rewards is not None:
            info["final_episode_rewards"] = final_rewards
            info["final_episode_lengths"] = final_lengths

        return obs, rewards, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Get current info dict."""
        return {
            "step_counts": self.gridworld.step_counts.copy(),
            "episode_rewards": self._episode_rewards.copy(),
            "episode_lengths": self._episode_lengths.copy(),
        }

    def render(self) -> Optional[str]:
        """
        Render the environment.

        Returns:
            ASCII string if render_mode='ascii', else None
        """
        if self.render_mode == "ascii":
            return self.gridworld.render_world(0)
        return None

    def close(self):
        """Clean up resources."""
        pass

    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """Get attribute from environments (SB3 VecEnv interface)."""
        if attr_name == "gridworld":
            return [self.gridworld] * (len(indices) if indices else self.num_envs)
        raise AttributeError(f"Unknown attribute: {attr_name}")

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Optional[List[int]] = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call method on environments (SB3 VecEnv interface)."""
        method = getattr(self.gridworld, method_name)
        return [method(*method_args, **method_kwargs)]

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed (deprecated, use reset(seed=...) instead)."""
        self.gridworld.rng = np.random.default_rng(seed)
        return [seed] if seed is not None else []


# Convenience wrapper for SB3's DummyVecEnv-style interface
class TileUniverseEnv(gym.Env):
    """
    Single-world Gymnasium environment.

    For cases where you need a standard single-environment interface.
    Internally uses ParallelGridworld with worlds=1.
    """

    metadata = {"render_modes": ["ascii", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        size: Tuple[int, int] = (32, 32),
        wall_density: float = 0.1,
        max_steps: int = 200,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.render_mode = render_mode

        self.gridworld = ParallelGridworld(
            worlds=1,
            size=size,
            wall_density=wall_density,
            max_steps=max_steps,
            seed=seed,
        )

        width, height = size

        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(height, width),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(5)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        obs = self.gridworld.reset()
        return obs[0], {}

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        actions = np.array([action], dtype=np.int32)
        obs, rewards, dones, info = self.gridworld.step(actions)

        goal_reached = info.get("goal_reached", np.array([False]))[0]
        terminated = goal_reached
        truncated = dones[0] and not terminated

        return obs[0], float(rewards[0]), terminated, truncated, {}

    def render(self) -> Optional[str]:
        if self.render_mode == "ascii":
            return self.gridworld.render_world(0)
        return None

    def close(self):
        pass
