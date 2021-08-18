/// Prints obj.
fn print(obj: u32) {}

/// Really prints obj.
fn print2(obj: u32) {}

struct PVec {}
struct PInt {}

impl PVec {
    /// Returns `true` if `self` contains 0u32
    fn contains(&self) -> bool { true }
}

impl PInt {
    #[invoke = "the reciprocal of {OBJ:self}"]
    /// Returns the reciprocal of `self`. Hello!
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
}