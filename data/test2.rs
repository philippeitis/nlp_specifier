const AAAA: crate2::Banana = 1;
const BBBB: &'static str = "hello world";

const CCCC: AAAAA<u32> = AAAAA { a: 5, b: 2 };

/// Does stuff.
/** Does other stuff. */
/// # Exceptions:
/// ```
/// rust code here
/// ```
#[wahoo]
#[yahoo(1)]
fn lemon_magic(y: u32, a: crate2::Lemon) -> u32 {
    let x = 1 + y;
    x
}

/** Does other stuff. */
/// # Exceptions:
/// ```
struct AAAAA<'a, T> {
    a: &'a T,
    b: crate2::Orange
}

/// Doc comments!
impl<'a, T> AAAAA<'a, T> {
    /// Hello world
    fn new() -> Self {}

    fn aaaaa_fn(&mut self) -> bool {}

}

struct BBBBB(u32, crate2::Avocado);

fn lime_operations(y: crate2::Lime) -> (u32, crate2::Lime) {
    let x = 1 + y;
    (x, y)
}

fn smoothify<X: Banana + Apricot, Y>(x: X, y: Y) {
}

fn blend<X, Y>(x: X, y: Y) where X: Banana + Apricot {
}

use std::banana::vec;

mod test {
    fn do_this() {}
}