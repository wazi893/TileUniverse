"""
Trajectory recording and replay for TileUniverse RL environments.

Demonstrates the reversibility feature by recording agent episodes
and allowing frame-by-frame replay and analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Frame:
    """A single frame in a trajectory."""
    step: int
    grid: np.ndarray
    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    action: int
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A complete episode trajectory."""
    world_id: int
    frames: List[Frame] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False

    def __len__(self) -> int:
        return len(self.frames)

    def add_frame(
        self,
        step: int,
        grid: np.ndarray,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        action: int,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ):
        """Add a frame to the trajectory."""
        self.frames.append(Frame(
            step=step,
            grid=grid.copy(),
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            action=action,
            reward=reward,
            done=done,
            info=info or {},
        ))
        self.total_reward += reward
        if done:
            # Check if success (reached goal)
            self.success = (agent_pos == goal_pos)


class TrajectoryRecorder:
    """
    Records trajectories from TileUniverse environments.

    Usage:
        recorder = TrajectoryRecorder()
        recorder.start(env, world_id=0)

        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            recorder.record(action, reward, done, info)

        trajectory = recorder.finish()
    """

    def __init__(self):
        self.env = None
        self.world_id: int = 0
        self.current_trajectory: Optional[Trajectory] = None
        self.trajectories: List[Trajectory] = []

    def start(self, env, world_id: int = 0):
        """Start recording a new trajectory."""
        self.env = env
        self.world_id = world_id
        self.current_trajectory = Trajectory(world_id=world_id)

        # Record initial frame (no action yet)
        grid = env.grids[world_id].copy()
        agent_pos = tuple(env.agent_positions[world_id])
        goal_pos = tuple(env.goal_positions[world_id])

        self.current_trajectory.add_frame(
            step=0,
            grid=grid,
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            action=-1,  # No action on first frame
            reward=0.0,
            done=False,
        )

    def record(
        self,
        action: int,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ):
        """Record a step."""
        if self.current_trajectory is None:
            raise RuntimeError("Must call start() before recording")

        grid = self.env.grids[self.world_id].copy()
        agent_pos = tuple(self.env.agent_positions[self.world_id])
        goal_pos = tuple(self.env.goal_positions[self.world_id])
        step = len(self.current_trajectory)

        self.current_trajectory.add_frame(
            step=step,
            grid=grid,
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            action=action,
            reward=reward,
            done=done,
            info=info,
        )

    def finish(self) -> Trajectory:
        """Finish recording and return the trajectory."""
        if self.current_trajectory is None:
            raise RuntimeError("No trajectory being recorded")

        trajectory = self.current_trajectory
        self.trajectories.append(trajectory)
        self.current_trajectory = None
        return trajectory


class TrajectoryPlayer:
    """
    Replays recorded trajectories with ASCII visualization.

    Usage:
        player = TrajectoryPlayer(trajectory)
        player.play()  # Play all frames
        player.goto(10)  # Jump to frame 10
        player.rewind(5)  # Go back 5 frames
    """

    ACTION_NAMES = ["noop", "up", "down", "left", "right"]
    ENTITY_CHARS = ".#GA"  # empty, wall, goal, agent

    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory
        self.current_frame: int = 0

    def __len__(self) -> int:
        return len(self.trajectory)

    @property
    def frame(self) -> Frame:
        """Get current frame."""
        return self.trajectory.frames[self.current_frame]

    def goto(self, frame_idx: int) -> Frame:
        """Jump to a specific frame."""
        self.current_frame = max(0, min(frame_idx, len(self) - 1))
        return self.frame

    def forward(self, steps: int = 1) -> Frame:
        """Move forward by steps."""
        return self.goto(self.current_frame + steps)

    def rewind(self, steps: int = 1) -> Frame:
        """Move backward by steps."""
        return self.goto(self.current_frame - steps)

    def reset(self) -> Frame:
        """Go to first frame."""
        return self.goto(0)

    def render_frame(self, frame: Optional[Frame] = None, chars: str = ".#GA") -> str:
        """Render a frame as ASCII."""
        if frame is None:
            frame = self.frame

        grid = frame.grid
        height, width = grid.shape
        lines = []

        # Header
        action_name = self.ACTION_NAMES[frame.action] if frame.action >= 0 else "start"
        lines.append(f"Step {frame.step} | Action: {action_name} | Reward: {frame.reward:.2f}")
        lines.append(f"Agent: {frame.agent_pos} | Goal: {frame.goal_pos}")
        lines.append("-" * (width + 2))

        # Grid
        for y in range(height):
            row = "|"
            for x in range(width):
                cell = min(grid[y, x], len(chars) - 1)
                row += chars[cell]
            row += "|"
            lines.append(row)

        lines.append("-" * (width + 2))

        if frame.done:
            if frame.agent_pos == frame.goal_pos:
                lines.append("*** GOAL REACHED! ***")
            else:
                lines.append("*** EPISODE ENDED (max steps) ***")

        return "\n".join(lines)

    def play(self, delay: float = 0.2, clear: bool = True):
        """
        Play the entire trajectory with animation.

        Args:
            delay: Seconds between frames
            clear: Whether to clear screen between frames
        """
        import time
        import sys

        for i in range(len(self)):
            self.goto(i)

            if clear:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")

            print(self.render_frame())
            print()

            if i < len(self) - 1:
                time.sleep(delay)

        # Summary
        print(f"\nTrajectory Summary:")
        print(f"  Total steps: {len(self)}")
        print(f"  Total reward: {self.trajectory.total_reward:.2f}")
        print(f"  Success: {self.trajectory.success}")

    def summary(self) -> str:
        """Get trajectory summary."""
        traj = self.trajectory
        lines = [
            f"Trajectory Summary (World {traj.world_id}):",
            f"  Length: {len(traj)} steps",
            f"  Total Reward: {traj.total_reward:.2f}",
            f"  Success: {traj.success}",
            f"  Start Position: {traj.frames[0].agent_pos}",
            f"  End Position: {traj.frames[-1].agent_pos}",
            f"  Goal Position: {traj.frames[0].goal_pos}",
        ]
        return "\n".join(lines)


def record_episode(
    env,
    policy,
    world_id: int = 0,
    max_steps: int = 200,
) -> Trajectory:
    """
    Convenience function to record a single episode.

    Args:
        env: ParallelGridworld environment
        policy: Function that takes observation and returns action
        world_id: Which world to record
        max_steps: Maximum steps to record

    Returns:
        Recorded trajectory
    """
    recorder = TrajectoryRecorder()
    recorder.start(env, world_id)

    obs = env.get_observations()

    for _ in range(max_steps):
        # Get action from policy
        action = policy(obs)
        if isinstance(action, np.ndarray):
            action = int(action[world_id])

        # Create action array (we only care about world_id)
        actions = np.zeros(env.worlds, dtype=np.int32)
        actions[world_id] = action

        # Step
        obs, rewards, dones, info = env.step(actions)

        # Record
        recorder.record(
            action=action,
            reward=float(rewards[world_id]),
            done=bool(dones[world_id]),
            info={k: v[world_id] if isinstance(v, np.ndarray) else v for k, v in info.items()},
        )

        if dones[world_id]:
            break

    return recorder.finish()


def compare_trajectories(traj1: Trajectory, traj2: Trajectory) -> str:
    """
    Compare two trajectories side-by-side.

    Useful for comparing agent behavior before/after training.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Trajectory Comparison")
    lines.append("=" * 60)

    max_len = max(len(traj1), len(traj2))

    headers = ["Metric", "Trajectory 1", "Trajectory 2"]
    lines.append(f"{headers[0]:20s} {headers[1]:15s} {headers[2]:15s}")
    lines.append("-" * 60)

    metrics = [
        ("Length", len(traj1), len(traj2)),
        ("Total Reward", f"{traj1.total_reward:.2f}", f"{traj2.total_reward:.2f}"),
        ("Success", traj1.success, traj2.success),
    ]

    for name, v1, v2 in metrics:
        lines.append(f"{name:20s} {str(v1):15s} {str(v2):15s}")

    return "\n".join(lines)
