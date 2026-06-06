"""
QuantumSNN Agent for Gymnasium environments.

Provides a drop-in agent that uses the Quantum-SNN hybrid with Triggered
interference mode for exploration/exploitation in reinforcement learning.

Key Features:
- Works with any Gymnasium environment (discrete or continuous obs)
- Automatic observation encoding to spike rates
- Triggered mode excels at sparse reward problems (+69% over epsilon-greedy)
- Built-in episode tracking and statistics

Example:
    >>> import gymnasium as gym
    >>> from tileuniverse.rl import QuantumSNNAgent
    >>>
    >>> env = gym.make("MountainCar-v0")
    >>> agent = QuantumSNNAgent.for_env(env, mode="triggered")
    >>>
    >>> # Training loop
    >>> for episode in range(1000):
    ...     obs, _ = env.reset()
    ...     total_reward = 0
    ...     done = False
    ...     while not done:
    ...         action = agent.act(obs)
    ...         obs, reward, terminated, truncated, _ = env.step(action)
    ...         agent.learn(reward)
    ...         total_reward += reward
    ...         done = terminated or truncated
    ...     agent.end_episode()
    ...     print(f"Episode {episode}: reward={total_reward}")
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from tileuniverse import InterferenceMode, QuantumSNNConfig, QuantumSNN


class ObservationEncoder:
    """
    Encodes Gymnasium observations to SNN spike rates [0-255].

    Supports:
    - Box (continuous): Scales to [0, 255] based on bounds
    - Discrete: One-hot encoding scaled to spike rates
    - MultiBinary: Direct binary to spike rate
    - MultiDiscrete: One-hot per dimension
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        encoding: str = "auto",
        resolution: int = 256,
    ):
        """
        Initialize encoder.

        Args:
            observation_space: Gymnasium observation space
            encoding: Encoding method ("auto", "linear", "log", "onehot")
            resolution: Output resolution (default 256 for [0-255])
        """
        self.observation_space = observation_space
        self.encoding = encoding
        self.resolution = resolution

        # Determine input size and encoding method
        self.n_inputs = self._compute_n_inputs()
        self._setup_encoder()

    def _compute_n_inputs(self) -> int:
        """Compute number of SNN input neurons needed."""
        space = self.observation_space

        if isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, spaces.Discrete):
            return space.n
        elif isinstance(space, spaces.MultiBinary):
            return space.n
        elif isinstance(space, spaces.MultiDiscrete):
            return int(np.sum(space.nvec))
        else:
            raise ValueError(f"Unsupported observation space: {type(space)}")

    def _setup_encoder(self):
        """Setup encoding parameters based on space type."""
        space = self.observation_space

        if isinstance(space, spaces.Box):
            # Handle infinite bounds
            self.low = np.where(np.isinf(space.low), -10.0, space.low)
            self.high = np.where(np.isinf(space.high), 10.0, space.high)
            self.range = self.high - self.low
            self.range = np.where(self.range == 0, 1.0, self.range)

    def encode(self, obs: np.ndarray) -> List[int]:
        """
        Encode observation to spike rates.

        Args:
            obs: Raw observation from environment

        Returns:
            List of spike rates [0-255] for each input neuron
        """
        space = self.observation_space

        if isinstance(space, spaces.Box):
            return self._encode_box(obs)
        elif isinstance(space, spaces.Discrete):
            return self._encode_discrete(obs)
        elif isinstance(space, spaces.MultiBinary):
            return self._encode_multibinary(obs)
        elif isinstance(space, spaces.MultiDiscrete):
            return self._encode_multidiscrete(obs)
        else:
            raise ValueError(f"Unsupported observation space: {type(space)}")

    def _encode_box(self, obs: np.ndarray) -> List[int]:
        """Encode continuous Box observation with contrast enhancement."""
        obs_flat = np.asarray(obs).flatten()

        # Clip to bounds
        obs_clipped = np.clip(obs_flat, self.low.flatten(), self.high.flatten())

        # Normalize to [-1, 1] centered at midpoint
        midpoint = (self.high.flatten() + self.low.flatten()) / 2
        half_range = self.range.flatten() / 2
        centered = (obs_clipped - midpoint) / half_range  # [-1, 1]

        # Apply contrast enhancement using tanh stretching
        # This amplifies small differences around the center
        # gain controls how much to amplify (higher = more contrast)
        gain = 3.0
        enhanced = np.tanh(centered * gain)  # Still in [-1, 1] but stretched

        # Convert to [0, 255]
        spike_rates = np.clip((enhanced + 1) * 127.5, 0, 255).astype(np.uint8)

        return spike_rates.tolist()

    def _encode_discrete(self, obs: int) -> List[int]:
        """Encode Discrete observation as one-hot spike rates."""
        n = self.observation_space.n
        spike_rates = [0] * n
        spike_rates[int(obs)] = 255
        return spike_rates

    def _encode_multibinary(self, obs: np.ndarray) -> List[int]:
        """Encode MultiBinary observation."""
        obs_flat = np.asarray(obs).flatten()
        return (obs_flat * 255).astype(np.uint8).tolist()

    def _encode_multidiscrete(self, obs: np.ndarray) -> List[int]:
        """Encode MultiDiscrete as concatenated one-hots."""
        spike_rates = []
        for i, val in enumerate(obs):
            n = self.observation_space.nvec[i]
            one_hot = [0] * n
            one_hot[int(val)] = 255
            spike_rates.extend(one_hot)
        return spike_rates


class QuantumSNNAgent:
    """
    Gymnasium-compatible agent using Quantum-SNN hybrid decision making.

    The agent automatically encodes observations to spike rates and uses
    quantum interference for exploration/exploitation trade-off.

    Attributes:
        brain: The underlying QuantumSNN network
        encoder: Observation encoder
        episodes: Number of completed episodes
        total_steps: Total environment steps taken
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        hidden_layers: Optional[List[int]] = None,
        mode: Union[str, InterferenceMode] = "triggered",
        ticks_per_decision: int = 20,
        learning_enabled: bool = True,
        reward_scale: float = 1.0,
        seed: int = 42,
        encoder: Optional[ObservationEncoder] = None,
        trigger_threshold: int = 50,
        trigger_count: int = 3,
    ):
        """
        Create a QuantumSNN agent.

        Args:
            n_inputs: Number of input features (observation size)
            n_actions: Number of discrete actions
            hidden_layers: Hidden layer sizes (default: adaptive based on inputs)
            mode: Interference mode ("triggered", "anti_grover", "grover", etc.)
            ticks_per_decision: SNN ticks per action selection
            learning_enabled: Whether to learn from rewards
            reward_scale: Scale factor for rewards (useful for sparse rewards)
            seed: Random seed
            encoder: Optional pre-configured observation encoder
            trigger_threshold: Reward threshold for triggering exploitation (0-100)
            trigger_count: Number of consecutive high rewards to trigger
        """
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.reward_scale = reward_scale
        self.encoder = encoder

        # Convert mode string to InterferenceMode if needed
        if isinstance(mode, str):
            mode = self._mode_from_string(mode)

        # Auto-size hidden layers if not specified
        if hidden_layers is None:
            hidden_layers = self._auto_hidden_layers(n_inputs, n_actions)

        # Create config
        config = QuantumSNNConfig(
            n_inputs=n_inputs,
            hidden_layers=hidden_layers,
            n_outputs=n_actions,
            mode=mode,
            ticks_per_decision=ticks_per_decision,
            learning_enabled=learning_enabled,
            trigger_threshold=trigger_threshold,
            trigger_count=trigger_count,
        )

        # Create network
        self.brain = QuantumSNN(config, seed=seed)
        self.config = config

        # Statistics
        self.episodes = 0
        self.total_steps = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_history: List[Dict[str, Any]] = []

        # Atari-specific reward shaping state
        self._reward_baseline = 0.0  # Running average of rewards
        self._reward_baseline_alpha = 0.01  # EMA smoothing factor
        self._state_counts: Dict[int, int] = {}  # State visitation counts for novelty
        self._novelty_scale = 10.0  # Scale for novelty bonus
        self._last_obs_hash = None  # Track state changes

    @classmethod
    def for_env(
        cls,
        env: gym.Env,
        mode: Union[str, InterferenceMode] = "triggered",
        hidden_layers: Optional[List[int]] = None,
        **kwargs,
    ) -> "QuantumSNNAgent":
        """
        Create an agent configured for a specific Gymnasium environment.

        Args:
            env: Gymnasium environment
            mode: Interference mode (default: "triggered" for sparse rewards)
            hidden_layers: Optional hidden layer sizes
            **kwargs: Additional arguments passed to __init__

        Returns:
            Configured QuantumSNNAgent
        """
        # Create encoder for observation space
        encoder = ObservationEncoder(env.observation_space)
        n_inputs = encoder.n_inputs

        # Get number of actions
        if isinstance(env.action_space, spaces.Discrete):
            n_actions = env.action_space.n
        else:
            raise ValueError(
                f"QuantumSNNAgent requires Discrete action space, "
                f"got {type(env.action_space)}"
            )

        return cls(
            n_inputs=n_inputs,
            n_actions=n_actions,
            hidden_layers=hidden_layers,
            mode=mode,
            encoder=encoder,
            **kwargs,
        )

    def _mode_from_string(self, mode: str) -> InterferenceMode:
        """Convert mode string to InterferenceMode."""
        modes = {
            "none": InterferenceMode.none(),
            "triggered": InterferenceMode.triggered(),
            "anti_grover": InterferenceMode.anti_grover(),
            "grover": InterferenceMode.grover(),
            "soft_mix": InterferenceMode.soft_mix(),
            "adaptive": InterferenceMode.adaptive(),
        }
        if mode not in modes:
            raise ValueError(f"Unknown mode '{mode}'. Valid: {list(modes.keys())}")
        return modes[mode]

    def _auto_hidden_layers(self, n_inputs: int, n_actions: int) -> List[int]:
        """Automatically size hidden layers based on input/output.

        NOTE: Hyperparameter sweep found single-layer networks outperform
        multi-layer by ~106%. Optimal size is around 20 neurons.
        """
        # Single layer performs best for sparse reward tasks
        # Size scales with problem complexity but stays in 16-32 range
        hidden_size = max(16, min(32, n_inputs + n_actions * 2))
        return [hidden_size]

    def act(self, observation: np.ndarray) -> int:
        """
        Select an action given an observation.

        Args:
            observation: Environment observation

        Returns:
            Selected action index
        """
        # Encode observation to spike rates
        if self.encoder is not None:
            inputs = self.encoder.encode(observation)
        else:
            # Assume observation is already spike rates
            inputs = np.asarray(observation).flatten().tolist()
            # Clip and convert to [0-255]
            inputs = [max(0, min(255, int(x))) for x in inputs]

        # Make decision
        action = self.brain.decide(inputs)

        self.total_steps += 1
        self._episode_length += 1

        return action

    def learn(self, reward: float, terminated: bool = False, observation: np.ndarray = None):
        """
        Apply learning with environment-appropriate reward shaping.

        Detects environment type by observation size and applies appropriate shaping:
        - Atari (128 bytes): Baseline + novelty + scaled extrinsic
        - Classic control (4-8 values): Angle-based shaping for CartPole
        - Generic: Simple reward scaling

        Args:
            reward: Reward from environment
            terminated: Whether episode ended due to failure
            observation: Current observation for reward shaping (optional)
        """
        obs_flat = None
        if observation is not None:
            obs_flat = np.asarray(observation).flatten()

        # Detect environment type and apply appropriate shaping
        if obs_flat is not None and len(obs_flat) == 128:
            # Atari RAM environment
            shaped_reward = self._shape_atari_reward(reward, obs_flat, terminated)
        elif obs_flat is not None and 2 <= len(obs_flat) <= 8:
            # Classic control (CartPole, MountainCar, etc.)
            shaped_reward = self._shape_classic_reward(reward, obs_flat, terminated)
        else:
            # Generic fallback
            shaped_reward = self._shape_generic_reward(reward, terminated)

        # Apply R-STDP learning
        self.brain.learn(shaped_reward)

        # Track episode reward
        self._episode_reward += reward

    def _shape_atari_reward(self, reward: float, obs: np.ndarray, terminated: bool) -> int:
        """
        Atari-specific reward shaping with baseline and novelty.

        Strategy:
        1. Survival bonus - small constant reward to maintain activity
        2. Extrinsic reward - scaled and shifted relative to baseline
        3. Novelty bonus - reward for visiting new states
        4. Action bonus - small reward for taking any action (prevents freezing)
        """
        # Update reward baseline (exponential moving average)
        self._reward_baseline = (
            self._reward_baseline_alpha * reward +
            (1 - self._reward_baseline_alpha) * self._reward_baseline
        )

        # Compute state hash for novelty
        # Use a subset of RAM bytes that tend to change (positions, scores, etc.)
        state_hash = hash(obs.tobytes()) % (2**20)  # 1M buckets

        # Novelty bonus: higher for rarely-seen states
        visit_count = self._state_counts.get(state_hash, 0)
        self._state_counts[state_hash] = visit_count + 1

        # Novelty decays with visit count: 1/sqrt(n+1)
        novelty_bonus = self._novelty_scale / np.sqrt(visit_count + 1)

        # State change bonus (reward exploring different states)
        state_change_bonus = 0
        if self._last_obs_hash is not None and state_hash != self._last_obs_hash:
            state_change_bonus = 3  # Small bonus for state transitions
        self._last_obs_hash = state_hash

        # Survival bonus (maintains network activity)
        survival_bonus = 5

        # Extrinsic reward relative to baseline
        # Shift so that average reward maps to small positive
        if reward > 0:
            # Positive reward: strong signal
            extrinsic = int(reward * 25)
        elif reward < 0:
            # Negative reward: reduce survival bonus instead of negative signal
            # This provides "disappointment" without causing activity death
            survival_bonus = max(0, survival_bonus + int(reward * 10))
            extrinsic = 0
        else:
            extrinsic = 0

        # Terminal penalty (reduce bonuses)
        if terminated:
            survival_bonus = 0
            novelty_bonus = 0

        # Combine components
        total = survival_bonus + extrinsic + int(novelty_bonus) + state_change_bonus

        return int(np.clip(total, 0, 80))

    def _shape_classic_reward(self, reward: float, obs: np.ndarray, terminated: bool) -> int:
        """
        Classic control reward shaping (CartPole, MountainCar, etc.)

        Uses observation features to provide continuous reward signal.
        """
        if len(obs) >= 4:
            # CartPole-like: use pole angle for shaping
            angle = abs(obs[2]) if len(obs) > 2 else 0
            if angle < 0.05:
                shaped_reward = 50  # Very good - strong positive
            elif angle < 0.1:
                shaped_reward = 30  # Good - moderate positive
            elif angle < 0.15:
                shaped_reward = 10  # OK - small positive
            else:
                shaped_reward = 2   # Bad - minimal reward (not zero!)
        elif len(obs) >= 2:
            # MountainCar-like: use position/velocity
            position = obs[0]
            velocity = abs(obs[1]) if len(obs) > 1 else 0
            # Reward for rightward movement and velocity
            shaped_reward = int(10 + position * 10 + velocity * 50)
            shaped_reward = max(2, min(50, shaped_reward))
        else:
            shaped_reward = int(np.clip(reward * 20 + 5, 2, 50))

        # Small penalty for termination (but not zero)
        if terminated and reward <= 0:
            shaped_reward = max(1, shaped_reward // 2)

        return shaped_reward

    def _shape_generic_reward(self, reward: float, terminated: bool) -> int:
        """Generic reward shaping with baseline activity."""
        # Ensure some baseline activity even with zero reward
        baseline = 5
        scaled = int(reward * 20)
        total = baseline + scaled
        return int(np.clip(total, 1, 60))  # Never zero

    def end_episode(self) -> Dict[str, Any]:
        """
        Signal end of episode and get statistics.

        Note: Learning now happens during learn() calls with shaped rewards.
        This method just records stats and resets counters.

        Returns:
            Dict with episode statistics
        """
        stats = {
            "episode": self.episodes,
            "reward": self._episode_reward,
            "length": self._episode_length,
            "exploitation_triggered": self.brain.is_exploitation_triggered,
            "consecutive_high_rewards": self.brain.consecutive_high_rewards,
        }

        self._episode_history.append(stats)
        self.episodes += 1

        # Reset TD state at episode boundary
        if getattr(self, "_td_enabled", False):
            self.brain.reset_td_episode()

        # Reset episode counters
        self._episode_reward = 0.0
        self._episode_length = 0

        return stats

    def reset(self):
        """Reset agent state (for new training run)."""
        self.brain.reset()
        self.episodes = 0
        self.total_steps = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_history = []

    @property
    def is_exploitation_triggered(self) -> bool:
        """Check if exploitation mode has been triggered."""
        return self.brain.is_exploitation_triggered

    @property
    def last_probabilities(self) -> List[float]:
        """Get action probabilities from last decision."""
        return self.brain.last_probabilities

    @property
    def exploration_entropy(self) -> float:
        """Get exploration entropy from last decision."""
        return self.brain.exploration_entropy()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "is_exploitation_triggered": self.brain.is_exploitation_triggered,
            "episode_history": self._episode_history[-100:],  # Last 100 episodes
        }

    # =========================================================================
    # EPIC 121: Curiosity-Driven Intrinsic Motivation
    # =========================================================================

    def enable_novelty_curiosity(
        self,
        reward_scale: float = 0.5,
        decay_rate: float = 0.0,
    ):
        """
        Enable novelty-based curiosity (count-based exploration).

        Provides intrinsic rewards for visiting novel states:
        reward = reward_scale / sqrt(visit_count)

        Args:
            reward_scale: Scale factor for intrinsic rewards (default: 0.5)
            decay_rate: How fast to forget old visits, 0-1 (default: 0.0 = never)
        """
        self.brain.enable_novelty_curiosity(reward_scale, decay_rate)
        self._curiosity_enabled = True
        self._curiosity_type = "novelty"

    def enable_prediction_curiosity(
        self,
        learning_rate: float = 0.01,
        reward_scale: float = 0.5,
    ):
        """
        Enable prediction-error curiosity.

        Provides intrinsic rewards for surprising state transitions,
        encouraging the agent to seek out poorly-understood dynamics.

        Args:
            learning_rate: Learning rate for predictor network (default: 0.01)
            reward_scale: Scale factor for intrinsic rewards (default: 0.5)
        """
        self.brain.enable_prediction_curiosity(learning_rate, reward_scale)
        self._curiosity_enabled = True
        self._curiosity_type = "prediction"

    def enable_combined_curiosity(self, reward_scale: float = 0.5):
        """
        Enable combined novelty + prediction curiosity.

        Uses both count-based and prediction-error intrinsic rewards.

        Args:
            reward_scale: Scale factor for intrinsic rewards (default: 0.5)
        """
        self.brain.enable_combined_curiosity(reward_scale)
        self._curiosity_enabled = True
        self._curiosity_type = "combined"

    def disable_curiosity(self):
        """Disable curiosity-driven exploration."""
        self.brain.disable_curiosity()
        self._curiosity_enabled = False
        self._curiosity_type = None

    @property
    def has_curiosity(self) -> bool:
        """Check if curiosity is enabled."""
        return getattr(self, "_curiosity_enabled", False)

    def learn_with_curiosity(
        self,
        extrinsic_reward: float,
        observation: np.ndarray,
    ) -> float:
        """
        Learn with curiosity-augmented rewards.

        Combines extrinsic reward with intrinsic curiosity reward.

        Args:
            extrinsic_reward: Reward from environment
            observation: Current observation for curiosity computation

        Returns:
            The intrinsic curiosity reward (for monitoring)
        """
        if self.encoder is not None:
            state = [float(x) / 255.0 for x in self.encoder.encode(observation)]
        else:
            state = [float(x) for x in np.asarray(observation).flatten()]

        scaled_reward = int(np.clip(extrinsic_reward * self.reward_scale * 100, -127, 127))
        intrinsic = self.brain.learn_with_curiosity(scaled_reward, state)

        self._episode_reward += extrinsic_reward + intrinsic

        return intrinsic

    # =========================================================================
    # EPIC 122: TD-Style Credit Assignment
    # =========================================================================

    def enable_td_credit(
        self,
        alpha: float = 0.05,
        gamma: float = 0.95,
        td_scale: float = 50.0,
    ):
        """
        Enable TD-style credit assignment for delayed rewards.

        Uses temporal difference learning to bootstrap value estimates,
        providing better learning signals for delayed reward problems.

        Args:
            alpha: Value function learning rate (default: 0.05)
            gamma: Discount factor (default: 0.95)
            td_scale: Scale factor for TD error signal (default: 50.0)
        """
        self.brain.enable_td_credit(alpha, gamma, td_scale)
        self._td_enabled = True

    def disable_td_credit(self):
        """Disable TD credit assignment."""
        self.brain.disable_td_credit()
        self._td_enabled = False

    @property
    def has_td_credit(self) -> bool:
        """Check if TD credit assignment is enabled."""
        return getattr(self, "_td_enabled", False)

    def learn_with_td(
        self,
        reward: float,
        observation: np.ndarray,
        done: bool = False,
    ) -> float:
        """
        Learn using TD-modulated reward signal.

        Uses TD error instead of raw reward for better delayed credit.

        Args:
            reward: Reward from environment
            observation: Current observation
            done: Whether episode ended

        Returns:
            The TD error (for monitoring)
        """
        if self.encoder is not None:
            state = [float(x) / 255.0 for x in self.encoder.encode(observation)]
        else:
            state = [float(x) for x in np.asarray(observation).flatten()]

        td_error = self.brain.learn_with_td(reward, state, done)

        self._episode_reward += reward

        return td_error

    def state_value(self, observation: np.ndarray) -> Optional[float]:
        """
        Get estimated value of a state.

        Only available when TD credit is enabled.

        Args:
            observation: State to evaluate

        Returns:
            Estimated value, or None if TD not enabled
        """
        if not getattr(self, "_td_enabled", False):
            return None

        if self.encoder is not None:
            state = [float(x) / 255.0 for x in self.encoder.encode(observation)]
        else:
            state = [float(x) for x in np.asarray(observation).flatten()]

        return self.brain.state_value(state)

    def reset_td_episode(self):
        """Reset TD state at episode boundary."""
        if getattr(self, "_td_enabled", False):
            self.brain.reset_td_episode()


def train_agent(
    env: gym.Env,
    agent: QuantumSNNAgent,
    n_episodes: int = 1000,
    max_steps: Optional[int] = None,
    render: bool = False,
    verbose: int = 1,
    callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Train a QuantumSNN agent on a Gymnasium environment.

    Args:
        env: Gymnasium environment
        agent: QuantumSNNAgent to train
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode (None = use env default)
        render: Whether to render during training
        verbose: Verbosity level (0=silent, 1=episode summaries, 2=step details)
        callback: Optional callback(episode, stats) called after each episode

    Returns:
        List of episode statistics
    """
    all_stats = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:
            if render:
                env.render()

            # Agent selects action
            action = agent.act(obs)

            # Environment step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Agent learns from reward
            agent.learn(reward)

            step += 1
            if max_steps is not None and step >= max_steps:
                done = True

            if verbose >= 2:
                print(f"  Step {step}: action={action}, reward={reward:.3f}")

        # End episode
        stats = agent.end_episode()
        all_stats.append(stats)

        if verbose >= 1:
            triggered_str = " [EXPLOITATION]" if stats["exploitation_triggered"] else ""
            print(
                f"Episode {episode + 1}/{n_episodes}: "
                f"reward={stats['reward']:.2f}, "
                f"length={stats['length']}{triggered_str}"
            )

        if callback is not None:
            callback(episode, stats)

    return all_stats


def evaluate_agent(
    env: gym.Env,
    agent: QuantumSNNAgent,
    n_episodes: int = 100,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained QuantumSNN agent.

    Args:
        env: Gymnasium environment
        agent: Trained QuantumSNNAgent
        n_episodes: Number of evaluation episodes
        deterministic: If True, use classical (greedy) action selection

    Returns:
        Dict with evaluation statistics
    """
    rewards = []
    lengths = []

    # Optionally force exploitation mode for deterministic evaluation
    original_triggered = agent.brain.is_exploitation_triggered
    if deterministic:
        agent.brain.trigger_exploitation()

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            if deterministic:
                # Use classical winner-take-all
                action = agent.brain.classical_action()
            else:
                action = agent.act(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)

    # Restore original state
    if deterministic and not original_triggered:
        agent.brain.reset_trigger()

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "n_episodes": n_episodes,
    }


def compare_modes(
    env_name: str,
    modes: List[str] = None,
    n_episodes: int = 500,
    n_seeds: int = 5,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compare different interference modes on an environment.

    Args:
        env_name: Gymnasium environment name (e.g., "CartPole-v1")
        modes: List of modes to compare (default: all)
        n_episodes: Training episodes per run
        n_seeds: Number of random seeds to average over
        verbose: Print progress

    Returns:
        Dict mapping mode -> {mean_reward, std_reward, ...}
    """
    if modes is None:
        modes = ["triggered", "anti_grover", "grover", "adaptive", "none"]

    results = {}

    for mode in modes:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Testing mode: {mode}")
            print(f"{'='*50}")

        mode_rewards = []

        for seed in range(n_seeds):
            env = gym.make(env_name)
            agent = QuantumSNNAgent.for_env(env, mode=mode, seed=seed * 1000)

            # Train
            stats = train_agent(
                env, agent, n_episodes=n_episodes, verbose=0
            )

            # Get final performance (average of last 100 episodes)
            final_rewards = [s["reward"] for s in stats[-100:]]
            mode_rewards.append(np.mean(final_rewards))

            if verbose:
                print(f"  Seed {seed}: final_reward={np.mean(final_rewards):.2f}")

            env.close()

        results[mode] = {
            "mean_reward": np.mean(mode_rewards),
            "std_reward": np.std(mode_rewards),
            "all_rewards": mode_rewards,
        }

        if verbose:
            print(f"  {mode}: {results[mode]['mean_reward']:.2f} +/- {results[mode]['std_reward']:.2f}")

    return results
