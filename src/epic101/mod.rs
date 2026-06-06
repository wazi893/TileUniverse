//! EPIC 101: LSTM vs Simple NN Evolution Experiment
//!
//! This module implements the experimental framework for testing whether
//! LSTM memory provides evolutionary advantage over simple feedforward NNs.
//!
//! # Key Question
//! Does LSTM memory provide enough evolutionary advantage to overcome
//! its 6.2× speed disadvantage vs SimpleBrain?
//!
//! # Tasks
//! - Task 1 (Stationary): Control - no memory needed
//! - Task 5 (Delayed Reward): Key test - memory required
//!
//! # Usage
//! ```ignore
//! use engine::epic101::{Task, StationaryTask, DelayedRewardTask, Evolution};
//! ```

mod delayed;
mod evolution;
mod organism;
mod stationary;
mod task;
mod world;

pub use delayed::{DelayedRewardConfig, DelayedRewardTask};
pub use evolution::{Evolution, EvolutionConfig, GenerationStats};
pub use organism::{Organism, OrganismState};
pub use stationary::{StationaryConfig, StationaryTask};
pub use task::{Task, TaskConfig};
pub use world::{Beacon, FoodPatch, World};
