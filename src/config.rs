//! EPIC 76.4: Configuration file support for universe scenarios
//!
//! Supports loading simulation configurations from JSON and YAML files.
//! This enables researchers to define reproducible experiments and share scenarios.
//!
//! # Example YAML config
//! ```yaml
//! name: "multi-world-logic-test"
//! worlds: 5
//! width: 512
//! height: 512
//! ruleset: logic
//! seed: 42
//! steps: 1000
//! depth: 50
//! initial_patterns:
//!   - world: 0
//!     pattern: "random"
//!     density: 0.3
//!   - world: 1
//!     pattern: "glider"
//!     position: [10, 10]
//! ```

#[cfg(feature = "config")]
use std::path::Path;

#[cfg(feature = "config")]
use serde::{Deserialize, Serialize};

/// Error type for configuration operations
#[derive(Debug)]
pub enum ConfigError {
    IoError(std::io::Error),
    JsonError(String),
    YamlError(String),
    UnknownFormat(String),
    ValidationError(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::IoError(e) => write!(f, "IO error: {}", e),
            ConfigError::JsonError(e) => write!(f, "JSON parse error: {}", e),
            ConfigError::YamlError(e) => write!(f, "YAML parse error: {}", e),
            ConfigError::UnknownFormat(ext) => write!(f, "Unknown config format: {}", ext),
            ConfigError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        ConfigError::IoError(e)
    }
}

pub type ConfigResult<T> = Result<T, ConfigError>;

/// Supported rulesets for configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum ConfigRuleset {
    GameOfLife,
    Rule110,
    Wire,
    #[default]
    Logic,
}

impl ConfigRuleset {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConfigRuleset::GameOfLife => "game_of_life",
            ConfigRuleset::Rule110 => "rule110",
            ConfigRuleset::Wire => "wire",
            ConfigRuleset::Logic => "logic",
        }
    }
}

impl std::str::FromStr for ConfigRuleset {
    type Err = ConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "game_of_life" | "gameoflife" | "gol" | "life" => Ok(ConfigRuleset::GameOfLife),
            "rule110" | "rule_110" | "r110" => Ok(ConfigRuleset::Rule110),
            "wire" | "wireworld" => Ok(ConfigRuleset::Wire),
            "logic" => Ok(ConfigRuleset::Logic),
            _ => Err(ConfigError::ValidationError(format!(
                "Unknown ruleset: {}",
                s
            ))),
        }
    }
}

/// Initial pattern configuration for a world
#[derive(Debug, Clone)]
#[cfg_attr(feature = "config", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "config", serde(tag = "pattern", rename_all = "snake_case"))]
pub enum InitialPattern {
    /// Fill with random cells at given density (0.0-1.0)
    Random {
        world: usize,
        #[cfg_attr(feature = "config", serde(default = "default_density"))]
        density: f64,
    },
    /// Clear all cells
    Clear { world: usize },
    /// Place a glider (for Game of Life)
    Glider {
        world: usize,
        #[cfg_attr(feature = "config", serde(default))]
        position: [usize; 2],
    },
    /// Place a custom bit pattern
    Custom {
        world: usize,
        position: [usize; 2],
        /// Row-major bit pattern (each u64 is one row of 64 bits)
        bits: Vec<u64>,
    },
}

#[cfg(feature = "config")]
fn default_density() -> f64 {
    0.3
}

/// Main configuration structure for a universe simulation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "config", derive(Serialize, Deserialize))]
pub struct UniverseConfig {
    /// Optional name for the scenario
    #[cfg_attr(feature = "config", serde(default))]
    pub name: Option<String>,

    /// Number of parallel worlds (default: 5)
    #[cfg_attr(feature = "config", serde(default = "default_worlds"))]
    pub worlds: usize,

    /// Grid width in cells (default: 512)
    #[cfg_attr(feature = "config", serde(default = "default_size"))]
    pub width: usize,

    /// Grid height in cells (default: 512)
    #[cfg_attr(feature = "config", serde(default = "default_size"))]
    pub height: usize,

    /// Cellular automaton ruleset
    #[cfg_attr(feature = "config", serde(default))]
    pub ruleset: ConfigRuleset,

    /// Random seed for reproducibility
    #[cfg_attr(feature = "config", serde(default = "default_seed"))]
    pub seed: u64,

    /// Number of evolution steps to run
    #[cfg_attr(feature = "config", serde(default = "default_steps"))]
    pub steps: u32,

    /// GPU depth batching (steps per kernel launch)
    #[cfg_attr(feature = "config", serde(default = "default_depth"))]
    pub depth: u32,

    /// Initial patterns to apply
    #[cfg_attr(feature = "config", serde(default))]
    pub initial_patterns: Vec<InitialPattern>,

    /// Enable reversible mode (history tracking)
    #[cfg_attr(feature = "config", serde(default))]
    pub reversible: bool,

    /// Warmup iterations before benchmarking
    #[cfg_attr(feature = "config", serde(default = "default_warmup"))]
    pub warmup: u32,
}

fn default_worlds() -> usize {
    5
}
fn default_size() -> usize {
    512
}
fn default_seed() -> u64 {
    42
}
fn default_steps() -> u32 {
    1000
}
fn default_depth() -> u32 {
    50
}
fn default_warmup() -> u32 {
    3
}

impl Default for UniverseConfig {
    fn default() -> Self {
        Self {
            name: None,
            worlds: default_worlds(),
            width: default_size(),
            height: default_size(),
            ruleset: ConfigRuleset::default(),
            seed: default_seed(),
            steps: default_steps(),
            depth: default_depth(),
            initial_patterns: Vec::new(),
            reversible: false,
            warmup: default_warmup(),
        }
    }
}

impl UniverseConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from a file (JSON or YAML based on extension)
    #[cfg(feature = "config")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> ConfigResult<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        match ext.as_deref() {
            Some("json") => Self::from_json(&content),
            Some("yaml") | Some("yml") => Self::from_yaml(&content),
            Some(ext) => Err(ConfigError::UnknownFormat(ext.to_string())),
            None => {
                // Try JSON first, then YAML
                Self::from_json(&content).or_else(|_| Self::from_yaml(&content))
            }
        }
    }

    /// Parse configuration from JSON string
    #[cfg(feature = "config")]
    pub fn from_json(json: &str) -> ConfigResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| ConfigError::JsonError(e.to_string()))
            .and_then(|c: Self| c.validate())
    }

    /// Parse configuration from YAML string
    #[cfg(feature = "config")]
    pub fn from_yaml(yaml: &str) -> ConfigResult<Self> {
        serde_yaml::from_str(yaml)
            .map_err(|e| ConfigError::YamlError(e.to_string()))
            .and_then(|c: Self| c.validate())
    }

    /// Serialize configuration to JSON
    #[cfg(feature = "config")]
    pub fn to_json(&self) -> ConfigResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| ConfigError::JsonError(e.to_string()))
    }

    /// Serialize configuration to YAML
    #[cfg(feature = "config")]
    pub fn to_yaml(&self) -> ConfigResult<String> {
        serde_yaml::to_string(self).map_err(|e| ConfigError::YamlError(e.to_string()))
    }

    /// Save configuration to a file
    #[cfg(feature = "config")]
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> ConfigResult<()> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        let content = match ext.as_deref() {
            Some("json") => self.to_json()?,
            Some("yaml") | Some("yml") => self.to_yaml()?,
            _ => self.to_yaml()?, // Default to YAML
        };

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration values
    pub fn validate(self) -> ConfigResult<Self> {
        if self.worlds == 0 {
            return Err(ConfigError::ValidationError("worlds must be > 0".into()));
        }
        if self.width == 0 || self.height == 0 {
            return Err(ConfigError::ValidationError(
                "width and height must be > 0".into(),
            ));
        }
        if self.width % 64 != 0 {
            return Err(ConfigError::ValidationError(format!(
                "width must be multiple of 64, got {}",
                self.width
            )));
        }
        if self.depth == 0 {
            return Err(ConfigError::ValidationError("depth must be > 0".into()));
        }

        // Validate initial patterns
        for pattern in &self.initial_patterns {
            let world_idx = match pattern {
                InitialPattern::Random { world, density } => {
                    if *density < 0.0 || *density > 1.0 {
                        return Err(ConfigError::ValidationError(format!(
                            "density must be 0.0-1.0, got {}",
                            density
                        )));
                    }
                    *world
                }
                InitialPattern::Clear { world } => *world,
                InitialPattern::Glider { world, .. } => *world,
                InitialPattern::Custom { world, .. } => *world,
            };

            if world_idx >= self.worlds {
                return Err(ConfigError::ValidationError(format!(
                    "pattern references world {} but only {} worlds exist",
                    world_idx, self.worlds
                )));
            }
        }

        Ok(self)
    }

    // Builder pattern methods

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_worlds(mut self, worlds: usize) -> Self {
        self.worlds = worlds;
        self
    }

    pub fn with_size(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_ruleset(mut self, ruleset: ConfigRuleset) -> Self {
        self.ruleset = ruleset;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    pub fn with_depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_reversible(mut self, reversible: bool) -> Self {
        self.reversible = reversible;
        self
    }

    pub fn with_pattern(mut self, pattern: InitialPattern) -> Self {
        self.initial_patterns.push(pattern);
        self
    }
}

/// Builder for creating scenarios programmatically
#[derive(Debug, Default)]
pub struct ScenarioBuilder {
    config: UniverseConfig,
}

impl ScenarioBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Quick preset: Game of Life with random soup
    pub fn game_of_life_soup(worlds: usize, size: usize, density: f64) -> UniverseConfig {
        let mut config = UniverseConfig::new()
            .with_worlds(worlds)
            .with_size(size, size)
            .with_ruleset(ConfigRuleset::GameOfLife);

        for w in 0..worlds {
            config
                .initial_patterns
                .push(InitialPattern::Random { world: w, density });
        }
        config
    }

    /// Quick preset: Logic gates benchmark
    pub fn logic_benchmark(worlds: usize, size: usize) -> UniverseConfig {
        let mut config = UniverseConfig::new()
            .with_worlds(worlds)
            .with_size(size, size)
            .with_ruleset(ConfigRuleset::Logic)
            .with_steps(10000)
            .with_depth(50);

        for w in 0..worlds {
            config.initial_patterns.push(InitialPattern::Random {
                world: w,
                density: 0.3,
            });
        }
        config
    }

    /// Quick preset: Rule110 Turing machine
    pub fn rule110_compute(width: usize, steps: u32) -> UniverseConfig {
        UniverseConfig::new()
            .with_worlds(1)
            .with_size(width, 1)
            .with_ruleset(ConfigRuleset::Rule110)
            .with_steps(steps)
            .with_pattern(InitialPattern::Random {
                world: 0,
                density: 0.5,
            })
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = Some(name.into());
        self
    }

    pub fn worlds(mut self, n: usize) -> Self {
        self.config.worlds = n;
        self
    }

    pub fn size(mut self, width: usize, height: usize) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    pub fn ruleset(mut self, ruleset: ConfigRuleset) -> Self {
        self.config.ruleset = ruleset;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    pub fn steps(mut self, steps: u32) -> Self {
        self.config.steps = steps;
        self
    }

    pub fn depth(mut self, depth: u32) -> Self {
        self.config.depth = depth;
        self
    }

    pub fn reversible(mut self, enabled: bool) -> Self {
        self.config.reversible = enabled;
        self
    }

    pub fn random_init(mut self, world: usize, density: f64) -> Self {
        self.config
            .initial_patterns
            .push(InitialPattern::Random { world, density });
        self
    }

    pub fn clear_init(mut self, world: usize) -> Self {
        self.config
            .initial_patterns
            .push(InitialPattern::Clear { world });
        self
    }

    pub fn build(self) -> ConfigResult<UniverseConfig> {
        self.config.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = UniverseConfig::default();
        assert_eq!(config.worlds, 5);
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
        assert_eq!(config.ruleset, ConfigRuleset::Logic);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_builder_pattern() {
        let config = UniverseConfig::new()
            .with_name("test-scenario")
            .with_worlds(10)
            .with_size(1024, 1024)
            .with_ruleset(ConfigRuleset::GameOfLife)
            .with_seed(12345)
            .validate()
            .unwrap();

        assert_eq!(config.name, Some("test-scenario".to_string()));
        assert_eq!(config.worlds, 10);
        assert_eq!(config.width, 1024);
        assert_eq!(config.ruleset, ConfigRuleset::GameOfLife);
    }

    #[test]
    fn test_validation_width() {
        let result = UniverseConfig::new()
            .with_size(100, 512) // Not multiple of 64
            .validate();

        assert!(result.is_err());
    }

    #[test]
    fn test_scenario_presets() {
        let gol = ScenarioBuilder::game_of_life_soup(5, 512, 0.3);
        assert_eq!(gol.ruleset, ConfigRuleset::GameOfLife);
        assert_eq!(gol.initial_patterns.len(), 5);

        let logic = ScenarioBuilder::logic_benchmark(5, 512);
        assert_eq!(logic.ruleset, ConfigRuleset::Logic);
        assert_eq!(logic.steps, 10000);
    }

    #[test]
    fn test_ruleset_parsing() {
        assert_eq!(
            "gol".parse::<ConfigRuleset>().unwrap(),
            ConfigRuleset::GameOfLife
        );
        assert_eq!(
            "life".parse::<ConfigRuleset>().unwrap(),
            ConfigRuleset::GameOfLife
        );
        assert_eq!(
            "rule110".parse::<ConfigRuleset>().unwrap(),
            ConfigRuleset::Rule110
        );
        assert_eq!(
            "logic".parse::<ConfigRuleset>().unwrap(),
            ConfigRuleset::Logic
        );
    }

    #[cfg(feature = "config")]
    #[test]
    fn test_json_roundtrip() {
        let config = UniverseConfig::new().with_name("json-test").with_worlds(3);

        let json = config.to_json().unwrap();
        let parsed = UniverseConfig::from_json(&json).unwrap();

        assert_eq!(parsed.name, Some("json-test".to_string()));
        assert_eq!(parsed.worlds, 3);
    }

    #[cfg(feature = "config")]
    #[test]
    fn test_yaml_roundtrip() {
        let config = UniverseConfig::new()
            .with_name("yaml-test")
            .with_ruleset(ConfigRuleset::GameOfLife);

        let yaml = config.to_yaml().unwrap();
        let parsed = UniverseConfig::from_yaml(&yaml).unwrap();

        assert_eq!(parsed.name, Some("yaml-test".to_string()));
        assert_eq!(parsed.ruleset, ConfigRuleset::GameOfLife);
    }
}
