struct usize;

struct PInt {}

impl PInt {
    #[specify]
    #[invoke = "the reciprocal of {OBJ:self}"]
    /// Returns the reciprocal of `self`.
    fn reciprocal(&self) -> Self {
        unimplemented!()
    }

    #[specify]
    /// Returns `true` if `self` is equal to `0` or `1`
    fn boolish(&self) -> Self {
        unimplemented!()
    }

    #[specify]
    /// Returns the reciprocal of `self` plus 1
    fn reciprocal_p(&self) -> Self {
        unimplemented!()
    }

    #[specify]
    /// Returns `self` plus `arg1` plus `arg3` plus `arg2`
    fn fn_with_three_args(&self, arg1: usize, arg2: usize, arg3: usize) -> Self {
        unimplemented!()
    }

    #[specify]
    /// Returns `self` plus `agh` plus `aghh` plus `aghhh`
    fn three_arg_fn(&self, agh: usize, aghh: usize, aghhh: usize) -> Self {
        unimplemented!()
    }
}
