extern crate prusti_contracts;
use prusti_contracts::*;

struct PrustiVec {
    v: Vec<u8>
}

struct A;

impl PrustiVec {
    /// # Invocation
    /// Swaps {a:IDENT}, {b:IDENT}
    /// {a:IDENT}, {b:IDENT} are swapped.
    /// Switches {a:IDENT}, {b:ident}
    ///
    /// # Specification
    /// a and b must be equal to 0 or `self.len()`.
    /// `self.len()` will not change.
    /// `self.lookup(a)` will be equal to `old(self.lookup(b))`.
    /// `self.lookup(b)` is equal to `old(self.lookup(a))`.
    /// For all indices between `0` and `self.len()`, not equal to `a`, not equal to `b`,
    /// `self.lookup(i)` will not change.
    fn swap(&mut self, a: usize, b: usize) {
        let va = self.lookup(a);
        let vb = self.lookup(b);
        self.replace(a, vb);
        self.replace(b, va);
    }

    /// # Specification
    /// Swaps a, b
    fn swap2(&mut self, a: usize, b: usize) {

    }
}

/// Mention ambiguous definitions, mention stored definitions
/// Support for traits - eg. swap() for index
/// eg. #[derive(Swap)]
/// Support core operations (eg. Trait / generic fns). (make sure to support reference types)
/// Check if "." occurs commonly.
/// VSCode extension

/// Returns true if the index is equal to 1
fn check_index_equals(index: usize) -> bool {
    index == 1
}
// QUANT: FOR ALL `i`
// RANGE: `between `0` and `self.len()`
// QUANT RANGEMOD CC MREL