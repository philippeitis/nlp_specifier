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
    #[invoke("the reciprocal of {OBJ:self}")]
    /// Returns the reciprocal of `self`
    fn reciprocal(&self) -> Self {}

    #[specify]
    /// Returns the reciprocal of `self` plus 1
    fn reciprocal_p(&self) -> Self {}
}