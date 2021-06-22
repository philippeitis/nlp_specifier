/// HELLO WORLD
extern crate prusti_contracts;
/// HELLO WORLD
use prusti_contracts::*;

struct PrustiVec {
    /// A vector of u8s
    v: Vec<u8>
}

struct A;

/// Doc comment
impl PrustiVec {
    /// Returns the length of self.
    #[trusted]
    #[pure]
    #[ensures(0 <= result)]
    fn len(&'a self) -> usize {
        self.v.len()
    }
    /// Returns the item at `index`. `index` must be less than
    /// the length of `self`.
    #[trusted]
    #[pure] // automatically detect if used in specs
    #[requires(0 <= index && index < self.len())]
    pub fn lookup(&self, index: usize) -> u8 {
        self.v[index]
    }

    // /// Returns the last item from the list. `self.len()` must be
    // /// greater than 1.
    // #[trusted]
    // #[pure]
    // #[requires(self.len() >= 1)]
    // fn last(&self) -> u8 {
    //     self.v.last().unwrap()
    // }

    /// length increases by one: self.len() += 1
    /// the last value is now `val`: self.last() == `val`
    /// indexes up to the original length: 0 up to (not including) old(self.len())
    /// remain the same: == old(self.lookup())
    /// The list's length increases by one, and the last value
    /// is now val. Items at indexes up to the original length
    /// remain the same.
    #[trusted]
    #[ensures(self.len() == old(self.len()) + 1)]
    #[ensures(self.len() >= 1)]
    #[ensures(self.lookup(self.len() - 1) == val)]
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < old(self.len())) ==> self.lookup(i) == old(self.lookup(i))
        )
    )]
    fn push(&mut self, val: u8) {
        self.v.push(val)
    }

    #[trusted]
    #[requires(0 <= index && index < self.len())]
    #[ensures(self.len() == old(self.len()) - 1)]
    #[ensures(self.len() >= 0)]
    #[ensures(result == old(self.lookup(index)))]
    /// check everything beforehand is unchanged
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < index && i < old(self.len()) && i < self.len()) ==> self.lookup(i) == old(self.lookup(i))
        )
    )]
    /// check that everything after index is moved over by one.
    #[ensures(
        forall(|i: usize|
            (1 <= i && index <= i && i < old(self.len()) && i < self.len()) ==> self.lookup(i - 1) == old(self.lookup(i))
        )
    )]
    fn remove(&mut self, index: usize) -> u8 {
        self.v.remove(index)
    }

    #[requires(self.len() >= 1)]
    #[ensures(self.len() == old(self.len()) - 1)]
    #[ensures(self.len() >= 0)]
    #[ensures(result == old({
        let l = self.len() - 1;
        self.lookup(l)
    }))]
    /// check everything beforehand is unchanged
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < old(self.len()) - 1) ==> self.lookup(i) == old(self.lookup(i))
        )
    )]
    fn pop(&mut self) -> u8 {
        let l = self.len() - 1;
        self.remove(l)
    }

    #[ensures(self.len() == old(self.len()) + other.len())]
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < other.len()) ==>
                self.lookup(i + old(self.len())) == other.lookup(i)
        )
    )]
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < old(self.len())) ==> self.lookup(i) == old(self.lookup(i))
        )
    )]
    // For each index from self.len() to self.len() + other.len()
    fn append(&mut self, other: &PrustiVec) {
        let mut i = 0;
        let old_len = self.len();
        while i < other.len() {
            body_invariant!(0 <= i && i < other.len());
            // The length of `self` is equal the old length of self plus i
            body_invariant!(self.len() == old(self.len()) + i);

            body_invariant!(
                forall(|j: usize|
                    (0 <= j && j < i)
                     && (0 <= j + old(self.len()) && j + old(self.len()) < self.len())
                     && j < other.len() ==>
                        self.lookup(j + old(self.len())) == other.lookup(j)
                )
            );
            body_invariant!(        forall(|j: usize|
                (0 <= j && j < old(self.len())) ==> self.lookup(j) == old(self.lookup(j))
            )
    );
            self.push(other.lookup(i));
            i += 1;
        }
    }

    #[pure]
    #[trusted]
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < self.len() && self.lookup(i) == val) ==> result
        )
    )]
    #[ensures(
        forall(|i: usize|
            !result ==> (0 <= i && i < self.len() && self.lookup(i) != val)
        )
    )]
    // #[ensures(result != forall(|i: usize| (0 <= i && i < self.len() && self.lookup(i) == val)))]
    fn contains(&self, val: u8) -> bool {
        self.v.contains(&val)
    }

    #[trusted]
    #[requires(0 <= index && index <= self.len())]
    #[ensures(self.len() == old(self.len()) + 1)]
    #[ensures(self.len() >= 1)]
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < old(self.len()) && i < index) ==> self.lookup(i) == old(self.lookup(i))
        )
    )]
    #[ensures(self.lookup(index) == val)]
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < old(self.len()) && i > index && i < self.len()) ==> self.lookup(i + 1) == old(self.lookup(i))
        )
    )]
    fn insert(&mut self, index: usize, val: u8) {
        self.v.insert(index, val)
    }

    #[trusted]
    #[requires(0 <= index && index <= self.len())]
    #[ensures(self.len() == old(self.len()))]
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < self.len() && i < index) ==> self.lookup(i) == old(self.lookup(i))
        )
    )]
    #[ensures(self.lookup(index) == val)]
    #[ensures(
        forall(|i: usize|
            (0 <= i && index < i && i < self.len()) ==> self.lookup(i) == old(self.lookup(i))
        )
    )]
    fn replace(&mut self, index: usize, val: u8) {
        self.v[index] = val;
    }

    #[ensures(self.contains(val))]
    fn insert_somewhere(&mut self, val: u8) {
        self.insert(0, val);
    }

    #[requires(0 <= a && a < self.len())]
    #[requires(0 <= b && b < self.len())]
    #[ensures(self.len() == old(self.len()))]
    #[ensures(
        forall(|i: usize|
            (0 <= i && i != a && i != b && i < self.len()) ==> self.lookup(i) == old(self.lookup(i))
        )
    )]
    #[ensures(self.lookup(a) == old(self.lookup(b)))]
    #[ensures(self.lookup(b) == old(self.lookup(a)))]
    fn swap(&mut self, a: usize, b: usize) {
        let va = self.lookup(a);
        let vb = self.lookup(b);
        self.replace(a, vb);
        self.replace(b, va);
    }

    #[requires(0 <= index && index < self.len())]
    #[ensures(self.len() == old(self.len()) - 1)]
    #[ensures(self.len() >= 0)]
    #[ensures(result == old(self.lookup(index)))]
    /// check everything beforehand is unchanged
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < index && i < old(self.len()) && i < self.len()) ==> self.lookup(i) == old(self.lookup(i))
        )
    )]
    /// check that everything after index and before the last index are moved over by one.
    #[ensures(
        forall(|i: usize|
            (index < i && i < old(self.len()) - 1 && i < self.len()) ==> self.lookup(i) == old(self.lookup(i))
        )
    )]
    #[ensures(index != old(self.len()) - 1 ==> self.lookup(index) == old({
        let l = self.len() - 1;
        self.lookup(l)
    }))]
    fn swap_remove(&mut self, index: usize) -> u8 {
        let l = self.len() - 1;
        self.swap(index, l);
        self.pop()
    }

    /// Creates a copy of `self`. The resulting `PrustiVec` will
    /// have the same length as `self`, and for each index `i` from `0` to `self.len()`,
    /// `self.lookup(i)` will be the same as `result.lookup(i)`
    #[trusted]
    #[ensures(self.len() == result.len())]
    #[ensures(forall(|i: usize|
        (0 <= i && i < self.len()) ==> self.lookup(i) == result.lookup(i)
    ))]
    fn clone(&self) -> Self {
        Self {
            v: self.v.clone()
        }
    }

    #[requires(self.len() > 0)]
    #[ensures(self.len() == result.len())]
    #[ensures(
        forall(|i: usize|
            (0 <= i && i < self.len() && i < result.len()) ==> self.lookup(i) == result.lookup(self.len() - i - 1)
        )
    )]
    fn reverse(&self) -> Self {
        let mut out = self.clone();
        let mut i = 0;

        while i < self.len() && i < out.len() {
            // Prusti encounters an error when no body condition is specified
            // Also, if j < i and i < self.len(), then j < self.len()
            body_invariant!(self.len() == out.len());
            body_invariant!(0 <= i && i < self.len() && i < out.len());
            body_invariant!(
                forall(|j: usize|
                    (0 <= j && j < i && j < self.len() && j < out.len()) ==> out.lookup(j) == {
                        let s = self.len() - j - 1;
                        self.lookup(s)
                    }
                )
            );
            let s = self.len() - i - 1;
            out.replace(i, self.lookup(s));
            i += 1;
        }

        out
    }

    // #[requires(self.len() > 0)]
    // #[ensures(
    //     forall(|i: usize|
    //         (0 <= i && i < self.len() ==> self.lookup(i) == old(self.lookup(self.len() - i - 1)))
    //     )
    // )]
    // fn reverse(&mut self) {
    //     let mut i = 0;
    //     while i < self.len() && self.len() - i - 1 > i {
    //         // Prusti encounters an error when no body condition is specified
    //         // Also, if j < i and i < self.len(), then j < self.len()
    //         body_invariant!(0 <= i && i < self.len());
    //         let s = self.len() - i - 1;
    //         body_invariant!(0 <= s && s < self.len());
    //         body_invariant!(s > i);

    //         body_invariant!(
    //             forall(|j: usize|
    //                 (j < i) && (0 <= j && j < self.len())
    //                 ==> self.lookup(j) == old({
    //                     let l = self.len() - j - 1;
    //                     // Doesn't specify which precondition fails.
    //                     self.lookup(l)
    //                 })
    //             )
    //         );
    //         // body_invariant!(
    //         //     forall(|j: usize|
    //         //         (j < i) &&
    //         //         (0 <= j && j < self.len()) &&
    //         //         (0 <= self.len() - j - 1 && self.len() - j - 1 < self.len())
    //         //         ==> self.lookup(j) == old({
    //         //             let s = self.len() - j - 1;
    //         //
    //         //             self.lookup(s)
    //         //         }))
    //         // );
    //         self.swap(i, s);
    //         i += 1;
    //     }
    // }

    // #[predicate]
    // fn not_contains(&self, val: u8) -> bool {
    //     forall(|i: usize|
    //         (0 <= i && i < self.len()) ==> self.lookup(i) != val
    //     )
    // }

    // #[predicate]
    // fn contains(&self, val: u8) -> bool {
    //     !self.not_contains(val)
    // }
}

// #[predicate]
// fn not_contains(s: &PrustiVec, val: u8) -> bool {
//     forall(|i: usize|
//         (0 <= i && i < s.len()) ==> s.lookup(i) != val
//     )
// }

/// #[predicate]
/// fn contains(s: &PrustiVec, val: u8) -> bool {
///     !s.not_contains(val)
/// }
enum PrustiOpt {
    Some(u8),
    None
}

struct PrustiIter {
    v: PrustiVec,
    i: usize,
}

impl PrustiIter {
    fn new(v: PrustiVec) -> Self {
        Self {
            v, i: 0,
        }
    }

    #[pure]
    #[requires(0 <= self.i && self.i < self.v.len())]
    #[ensures(result == {
        let i = self.i;
        self.v.lookup(i)
    })]
    fn lookup_cur(&self) -> u8 {
        let i = self.i;
        self.v.lookup(i)
    }

    #[requires(self.i > 0)]
    #[ensures(self.i == old(self.i) + 1)]
    #[ensures(old(self.i) >= self.v.len() ==> matches!(result, PrustiOpt::None))]
    #[ensures(old(self.i) < self.v.len() ==> old({
        let v = self.lookup_cur();
        matches!(result, PrustiOpt::Some(v))
    }))]
    fn next(&mut self) -> PrustiOpt {
        let i = self.i;
        let result = if i < self.v.len() {
            PrustiOpt::Some(self.v.lookup(i))
        } else {
            PrustiOpt::None
        };
        self.i += 1;
        result
    }
}

/// Module
mod Hello {
    fn wassap() {
    }
}