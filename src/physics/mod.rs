// Physics simulation module - field-based physics (heat, charge, diffusion, etc.)

pub mod charge;
pub mod coupling;
pub mod diffuse;
pub mod experiments;
pub mod feedback;
pub mod fieldstep;
pub mod heat;
pub mod interact;
pub mod logic_coupling;
pub mod reaction;
pub mod scenarios;

// Re-export commonly used types
pub use charge::ChargeParams;
pub use coupling::CoupledParams;
pub use diffuse::DiffuseParams;
pub use experiments::{
    ExperimentKind, ExperimentParam, ExperimentResult, Range1D, RunResult, run_experiment,
};
pub use feedback::FeedbackParams;
pub use fieldstep::FieldStepParams;
pub use heat::HeatParams;
pub use interact::InteractionParams;
pub use reaction::ReactionParams;
pub use scenarios::{PhysicsScenario, Region, ScenarioKind, ScenarioSummary, run_scenario};

// Physics-to-logic coupling
pub use logic_coupling::{
    ChargeBiasAffectedTiles, ChargeBiasCoupling, HeatDegradationMode, HeatErrorCoupling,
    PhysicsCouplingConfig, PhysicsCouplingContext, PowerEnableCoupling, UnpoweredBehavior,
    apply_charge_bias_to_comparison, apply_charge_bias_to_mux, apply_charge_bias_to_zero,
    apply_heat_degradation, apply_physics_coupling, calculate_charge_bias, is_charge_bias_affected,
};
