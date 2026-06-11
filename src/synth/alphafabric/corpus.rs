//! Parametric circuit corpus and a deterministic train/test split.
//!
//! AF1 goal D: a generator that produces many circuits across a range of sizes,
//! with a held-out test split, so later phases can pretrain on `Train` and
//! evaluate generalization on `Test`. The builders mirror the proven idioms in
//! `synth::benchmark` so they map and route the same way.

use super::circuit::Circuit;
use crate::synth::aig::{Aig, AigLit};

/// Which split a corpus circuit belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Test,
}

/// One corpus circuit with its split tag.
pub struct CorpusEntry {
    pub circuit: Circuit,
    pub split: Split,
}

/// Width-parametric ripple-carry adder (`width`-bit a + b -> `width+1`-bit sum).
pub fn build_ripple_adder(width: u32) -> Aig {
    let mut aig = Aig::new();
    let a = aig.add_input_bus("a", width);
    let b = aig.add_input_bus("b", width);
    let mut carry = AigLit::FALSE;
    let mut sum_bits = Vec::new();
    for i in 0..width as usize {
        let axb = aig.xor(a[i], b[i]);
        let sum = aig.xor(axb, carry);
        let ab = aig.and(a[i], b[i]);
        let caxb = aig.and(carry, axb);
        carry = aig.or(ab, caxb);
        sum_bits.push(sum);
    }
    sum_bits.push(carry);
    aig.add_output_bus("sum", &sum_bits);
    aig
}

/// Width-parametric equality comparator (left-skewed AND chain of XNORs).
pub fn build_eq_comparator(width: u32) -> Aig {
    let mut aig = Aig::new();
    let a = aig.add_input_bus("a", width);
    let b = aig.add_input_bus("b", width);
    let mut eq = AigLit::TRUE;
    for i in 0..width as usize {
        let diff = aig.xor(a[i], b[i]);
        let same = aig.not(diff);
        eq = aig.and(eq, same);
    }
    aig.add_output("eq", eq);
    aig
}

/// Width-parametric parity (XOR reduction over `width` inputs).
pub fn build_parity(width: u32) -> Aig {
    let mut aig = Aig::new();
    let x = aig.add_input_bus("x", width);
    let mut acc = AigLit::FALSE;
    for &xi in &x {
        acc = aig.xor(acc, xi);
    }
    aig.add_output("parity", acc);
    aig
}

/// Width-parametric AND reduction over `width` inputs.
pub fn build_and_reduce(width: u32) -> Aig {
    let mut aig = Aig::new();
    let x = aig.add_input_bus("x", width);
    let mut acc = AigLit::TRUE;
    for &xi in &x {
        acc = aig.and(acc, xi);
    }
    aig.add_output("all", acc);
    aig
}

/// A named, width-parametric circuit family.
type Family = (&'static str, fn(u32) -> Aig);

/// The four parametric families in the corpus.
const FAMILIES: [Family; 4] = [
    ("adder", build_ripple_adder),
    ("eq", build_eq_comparator),
    ("parity", build_parity),
    ("andreduce", build_and_reduce),
];

/// Width range swept per family (inclusive).
const WIDTH_RANGE: std::ops::RangeInclusive<u32> = 2..=20;

/// Build the full corpus: each family across [`WIDTH_RANGE`], with a deterministic
/// split (test = widths divisible by 4, train = the rest).
pub fn corpus() -> Vec<CorpusEntry> {
    let mut out = Vec::new();
    for (fam, build) in FAMILIES {
        for width in WIDTH_RANGE {
            let split = if width % 4 == 0 {
                Split::Test
            } else {
                Split::Train
            };
            let name = format!("{fam}{width}");
            out.push(CorpusEntry {
                circuit: Circuit::from_aig(name, build(width)),
                split,
            });
        }
    }
    out
}

/// Circuits belonging to a given split.
pub fn split(which: Split) -> Vec<Circuit> {
    corpus()
        .into_iter()
        .filter(|e| e.split == which)
        .map(|e| e.circuit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_split_has_enough_train_and_test() {
        let all = corpus();
        let train = all.iter().filter(|e| e.split == Split::Train).count();
        let test = all.iter().filter(|e| e.split == Split::Test).count();
        // Roadmap goal D: >= 50 train + >= 10 held-out test.
        assert!(train >= 50, "expected >= 50 train circuits, got {train}");
        assert!(test >= 10, "expected >= 10 test circuits, got {test}");
    }

    #[test]
    fn corpus_is_deterministic() {
        let a: Vec<String> = corpus().iter().map(|e| e.circuit.name.clone()).collect();
        let b: Vec<String> = corpus().iter().map(|e| e.circuit.name.clone()).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn small_corpus_circuits_map_correctly() {
        // Keep the check cheap: only verify circuits whose exhaustive equivalence
        // check is small (<= 12 inputs).
        for entry in corpus() {
            if entry.circuit.num_inputs() <= 12 {
                assert!(
                    entry.circuit.mapping_is_correct(),
                    "{} mapped incorrectly",
                    entry.circuit.name
                );
            }
        }
    }
}
