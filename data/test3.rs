/// Prints obj.
fn print(obj: u32) {}

/// Really prints obj.
fn print2(obj: u32) {}

struct PVec {
    v: Vec<u32>
}
struct PInt {}

impl PVec {
    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    fn remove(&mut self, index: usize) -> u32 {
        self.v.remove(index).unwrap()
    }

    /// Removes the last element from a vector and returns it, or `None` if it
    /// is empty.
    fn pop(&mut self) -> u32 {
        self.remove(self.len() - 1)
    }

    /// Returns `true` if `self` contains 0u32
    fn contains(&self, val: u32) -> bool {
        self.v.contains(val)
    }
}

impl PInt {
    #[invoke = "the reciprocal of {OBJ:self}"]
    /// Returns the reciprocal of `self`.
    fn reciprocal(&self) -> Self {}

    #[specify]
    /// Returns the reciprocal of `self` plus 1
    fn reciprocal_p(&self) -> Self {}

    #[specify]
    /// Returns `self` plus `arg1` plus `arg3` plus `arg2`
    fn fn_with_three_args(&self, arg1: usize, arg2: usize, arg3: usize) -> Self {}

    #[specify]
    /// Returns `self` plus `agh` plus `aghh` plus `aghhh`
    fn three_arg_fn(&self, agh: usize, aghh: usize, aghhh: usize) -> Self {}

    #[specify]
    /// Returns the frobinical of `self`
    fn frobinical(&self) -> Self {}

}