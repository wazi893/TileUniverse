"""
ParallelGridworld - A simple but non-trivial RL environment.

The agent must navigate a grid to reach a goal while avoiding walls.
Runs N parallel worlds simultaneously at GPU speed.

Entities (encoded in grid):
    0 = empty
    1 = wall/obstacle
    2 = goal
    3 = agent

Rewards:
    +1.0 for reaching goal (terminal)
    -0.01 per step (time penalty)
    -0.1 for hitting wall (optional)

Actions:
    0 = no-op
    1 = up
    2 = down
    3 = left
    4 = right
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import tileuniverse as tu


# Entity codes
EMPTY = 0
WALL = 1
GOAL = 2
AGENT = 3


class ParallelGridworld:
    """
    A parallel gridworld environment running on TileUniverse.

    Each world has:
    - One agent
    - One goal
    - Random walls/obstacles

    Args:
        worlds: Number of parallel worlds (default: 64)
        size: Grid size (width, height) - must be powers of 2 (default: (32, 32))
        wall_density: Fraction of cells that are walls (default: 0.1)
        max_steps: Maximum steps per episode (default: 200)
        time_penalty: Reward penalty per step (default: -0.01)
        wall_penalty: Reward penalty for hitting wall (default: -0.1)
        goal_reward: Reward for reaching goal (default: 1.0)
        seed: Random seed (default: None)
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
    ):
        self.worlds = worlds
        self.width, self.height = size
        self.wall_density = wall_density
        self.max_steps = max_steps
        self.time_penalty = time_penalty
        self.wall_penalty = wall_penalty
        self.goal_reward = goal_reward

        # Random state
        self.rng = np.random.default_rng(seed)

        # Create TileUniverse engine
        # Note: we use the engine for state storage but handle entities ourselves
        self.engine = tu.Engine(
            worlds=worlds,
            size=size,
            ruleset="wire",  # Simple ruleset, we override behavior
            seed=seed or 42,
            reversible=True,  # Enable for debugging
            max_history=max_steps + 10,
        )

        # Track agent and goal positions per world
        # Shape: (worlds, 2) for (x, y)
        self.agent_positions = np.zeros((worlds, 2), dtype=np.int32)
        self.goal_positions = np.zeros((worlds, 2), dtype=np.int32)

        # Track step counts and done flags
        self.step_counts = np.zeros(worlds, dtype=np.int32)
        self.dones = np.zeros(worlds, dtype=bool)

        # Grid storage: (worlds, height, width) as uint8
        self.grids = np.zeros((worlds, self.height, self.width), dtype=np.uint8)

        # Action mappings: (dx, dy)
        self.action_deltas = {
            0: (0, 0),   # no-op
            1: (0, -1),  # up
            2: (0, 1),   # down
            3: (-1, 0),  # left
            4: (1, 0),   # right
        }

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        """Returns (worlds, height, width)"""
        return (self.worlds, self.height, self.width)

    def reset(self, world_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environments.

        Args:
            world_mask: Optional bool array of shape (worlds,) specifying which
                       worlds to reset. If None, resets all worlds.

        Returns:
            observations: np.ndarray of shape (worlds, height, width) as uint8
        """
        if world_mask is None:
            world_mask = np.ones(self.worlds, dtype=bool)

        for w in range(self.worlds):
            if not world_mask[w]:
                continue

            # Reset this world
            self._reset_world(w)
            self.step_counts[w] = 0
            self.dones[w] = False

        return self.get_observations()

    def _reset_world(self, world: int):
        """Reset a single world's grid, agent, and goal."""
        grid = np.zeros((self.height, self.width), dtype=np.uint8)

        # Add random walls
        wall_mask = self.rng.random((self.height, self.width)) < self.wall_density
        grid[wall_mask] = WALL

        # Clear borders (no walls on edges for easier navigation)
        grid[0, :] = EMPTY
        grid[-1, :] = EMPTY
        grid[:, 0] = EMPTY
        grid[:, -1] = EMPTY

        # Place goal in a random empty cell (prefer far corner)
        empty_cells = np.argwhere(grid == EMPTY)
        if len(empty_cells) < 2:
            # Not enough space, clear more walls
            grid[:] = EMPTY
            empty_cells = np.argwhere(grid == EMPTY)

        # Place goal preferably in bottom-right quadrant
        br_cells = empty_cells[
            (empty_cells[:, 0] >= self.height // 2) &
            (empty_cells[:, 1] >= self.width // 2)
        ]
        if len(br_cells) > 0:
            goal_idx = self.rng.integers(len(br_cells))
            goal_y, goal_x = br_cells[goal_idx]
        else:
            goal_idx = self.rng.integers(len(empty_cells))
            goal_y, goal_x = empty_cells[goal_idx]

        grid[goal_y, goal_x] = GOAL
        self.goal_positions[world] = [goal_x, goal_y]

        # Place agent preferably in top-left quadrant
        empty_cells = np.argwhere(grid == EMPTY)
        tl_cells = empty_cells[
            (empty_cells[:, 0] < self.height // 2) &
            (empty_cells[:, 1] < self.width // 2)
        ]
        if len(tl_cells) > 0:
            agent_idx = self.rng.integers(len(tl_cells))
            agent_y, agent_x = tl_cells[agent_idx]
        else:
            agent_idx = self.rng.integers(len(empty_cells))
            agent_y, agent_x = empty_cells[agent_idx]

        grid[agent_y, agent_x] = AGENT
        self.agent_positions[world] = [agent_x, agent_y]

        # Store grid
        self.grids[world] = grid

        # Sync to engine (convert to u64 for engine storage)
        self._sync_grid_to_engine(world)

    def _sync_grid_to_engine(self, world: int):
        """Sync our uint8 grid to the engine's u64 state."""
        # Engine expects u64, we just cast
        grid_u64 = self.grids[world].astype(np.uint64)
        self.engine.set_world(world, grid_u64)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Take a step in all worlds.

        Args:
            actions: np.ndarray of shape (worlds,) with action indices 0-4

        Returns:
            observations: (worlds, height, width) uint8
            rewards: (worlds,) float32
            dones: (worlds,) bool
            infos: dict with per-world info
        """
        rewards = np.full(self.worlds, self.time_penalty, dtype=np.float32)

        for w in range(self.worlds):
            if self.dones[w]:
                # Already done, skip
                continue

            action = int(actions[w])
            reward_delta = self._step_world(w, action)
            rewards[w] += reward_delta

            self.step_counts[w] += 1

            # Check termination
            if self._check_goal_reached(w):
                rewards[w] += self.goal_reward
                self.dones[w] = True
            elif self.step_counts[w] >= self.max_steps:
                self.dones[w] = True

        # Evolve engine (for any CA dynamics we want)
        # For basic gridworld, this is optional
        # self.engine.evolve(1)

        infos = {
            "step_counts": self.step_counts.copy(),
            "goal_reached": self._check_goals_reached(),
        }

        return self.get_observations(), rewards, self.dones.copy(), infos

    def _step_world(self, world: int, action: int) -> float:
        """Move agent in a single world. Returns reward delta."""
        dx, dy = self.action_deltas.get(action, (0, 0))

        old_x, old_y = self.agent_positions[world]
        new_x = old_x + dx
        new_y = old_y + dy

        # Bounds check
        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            return self.wall_penalty  # Hit boundary

        # Wall check
        cell = self.grids[world, new_y, new_x]
        if cell == WALL:
            return self.wall_penalty  # Hit wall

        # Valid move - update position
        # Clear old position
        self.grids[world, old_y, old_x] = EMPTY

        # Set new position (don't overwrite goal)
        if cell != GOAL:
            self.grids[world, new_y, new_x] = AGENT
        else:
            # On goal - keep goal visible, agent overlaps
            self.grids[world, new_y, new_x] = AGENT  # Will trigger done

        self.agent_positions[world] = [new_x, new_y]

        # Sync to engine
        self._sync_grid_to_engine(world)

        return 0.0  # No extra reward for valid move

    def _check_goal_reached(self, world: int) -> bool:
        """Check if agent reached goal in a world."""
        ax, ay = self.agent_positions[world]
        gx, gy = self.goal_positions[world]
        return ax == gx and ay == gy

    def _check_goals_reached(self) -> np.ndarray:
        """Check goal reached for all worlds."""
        return np.all(self.agent_positions == self.goal_positions, axis=1)

    def get_observations(self) -> np.ndarray:
        """Get observations for all worlds."""
        return self.grids.copy()

    def get_observation_channels(self) -> np.ndarray:
        """
        Get observations as separate channels.

        Returns:
            (worlds, 4, height, width) with channels:
                0 = empty
                1 = wall
                2 = goal
                3 = agent
        """
        obs = np.zeros((self.worlds, 4, self.height, self.width), dtype=np.float32)
        for c in range(4):
            obs[:, c] = (self.grids == c).astype(np.float32)
        return obs

    def render_world(self, world: int = 0, chars: str = ".#GA") -> str:
        """
        Render a single world as ASCII.

        Args:
            world: World index
            chars: Characters for [empty, wall, goal, agent]

        Returns:
            ASCII string representation
        """
        grid = self.grids[world]
        lines = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                cell = grid[y, x]
                row += chars[min(cell, len(chars) - 1)]
            lines.append(row)
        return "\n".join(lines)

    def get_state(self) -> Dict[str, Any]:
        """Get full state for serialization."""
        return {
            "grids": self.grids.copy(),
            "agent_positions": self.agent_positions.copy(),
            "goal_positions": self.goal_positions.copy(),
            "step_counts": self.step_counts.copy(),
            "dones": self.dones.copy(),
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from serialization."""
        self.grids = state["grids"].copy()
        self.agent_positions = state["agent_positions"].copy()
        self.goal_positions = state["goal_positions"].copy()
        self.step_counts = state["step_counts"].copy()
        self.dones = state["dones"].copy()

        # Sync all worlds to engine
        for w in range(self.worlds):
            self._sync_grid_to_engine(w)
