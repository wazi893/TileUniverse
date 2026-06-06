use engine::synth::sequential::{SeqError, SequentialCircuit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// PySequentialCircuit
// ---------------------------------------------------------------------------

#[pyclass(unsendable, name = "SequentialCircuit")]
pub struct PySequentialCircuit {
    inner: SequentialCircuit,
}

#[pymethods]
impl PySequentialCircuit {
    /// Build an N-bit binary counter.
    #[staticmethod]
    #[pyo3(signature = (bits))]
    fn counter(bits: u32) -> PyResult<Self> {
        let inner = SequentialCircuit::counter(bits).map_err(seq_err)?;
        Ok(PySequentialCircuit { inner })
    }

    /// Build an N-bit shift register.
    #[staticmethod]
    #[pyo3(signature = (bits))]
    fn shift_register(bits: u32) -> PyResult<Self> {
        let inner = SequentialCircuit::shift_register(bits).map_err(seq_err)?;
        Ok(PySequentialCircuit { inner })
    }

    /// Build from an AIG spec: truth tables for each output of the combinational
    /// next-state logic. Convention: first `num_latches` inputs are state feedback,
    /// first `num_latches` outputs are next-state.
    #[staticmethod]
    #[pyo3(signature = (num_latches, truth_tables))]
    fn from_aig_spec(num_latches: usize, truth_tables: Vec<(u64, u32)>) -> PyResult<Self> {
        use engine::synth::Aig;

        if truth_tables.is_empty() {
            return Err(PyValueError::new_err("truth_tables must not be empty"));
        }
        let num_inputs = truth_tables[0].1;
        for (i, &(_, n)) in truth_tables.iter().enumerate() {
            if n != num_inputs {
                return Err(PyValueError::new_err(format!(
                    "all truth tables must have same num_inputs (output {} has {} != {})",
                    i, n, num_inputs
                )));
            }
        }

        // Use the correct LSB-first multi-output AIG builder from aig.rs.
        let aig = Aig::from_truth_tables(&truth_tables);
        let inner = SequentialCircuit::from_aig(&aig, num_latches).map_err(seq_err)?;
        Ok(PySequentialCircuit { inner })
    }

    /// Advance one clock cycle.
    #[pyo3(signature = (inputs = None))]
    fn tick(&mut self, inputs: Option<Vec<bool>>) -> PyResult<Vec<bool>> {
        let input_slice = inputs.as_deref().unwrap_or(&[]);
        if input_slice.len() != self.inner.num_inputs {
            return Err(PyValueError::new_err(format!(
                "expected {} inputs, got {}",
                self.inner.num_inputs,
                input_slice.len()
            )));
        }
        Ok(self.inner.tick(input_slice))
    }

    /// Read current state as list of bools (LSB first).
    fn state(&self) -> Vec<bool> {
        self.inner.read_state()
    }

    /// Read current state as integer.
    fn state_value(&self) -> u64 {
        self.inner.state_value()
    }

    /// Set state from list of bools (LSB first).
    fn set_state(&mut self, state: Vec<bool>) -> PyResult<()> {
        if state.len() != self.inner.num_state_bits {
            return Err(PyValueError::new_err(format!(
                "expected {} bits, got {}",
                self.inner.num_state_bits,
                state.len()
            )));
        }
        self.inner.set_state(&state);
        Ok(())
    }

    /// Set state from integer value.
    fn set_state_value(&mut self, val: u64) {
        self.inner.set_state_value(val);
    }

    /// Read external outputs (empty for counter/shift_register).
    fn outputs(&self) -> Vec<bool> {
        // External outputs are returned by tick(), not stored.
        // For circuits with no external outputs, this returns empty.
        vec![]
    }

    #[getter]
    fn num_state_bits(&self) -> usize {
        self.inner.num_state_bits
    }

    #[getter]
    fn num_inputs(&self) -> usize {
        self.inner.num_inputs
    }

    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs
    }

    #[getter]
    fn num_cycles(&self) -> u64 {
        self.inner.cycle_count()
    }

    /// Grid info: (width, height, layers) or None for shift register.
    fn grid_info(&self) -> Option<(usize, usize, usize)> {
        self.inner.grid_info()
    }

    fn __repr__(&self) -> String {
        format!(
            "SequentialCircuit(state_bits={}, inputs={}, outputs={}, cycles={})",
            self.inner.num_state_bits,
            self.inner.num_inputs,
            self.inner.num_outputs,
            self.inner.cycle_count(),
        )
    }
}

fn seq_err(e: SeqError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySequentialCircuit>()?;
    Ok(())
}
