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

    #[specify]
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
