mod eir;
pub mod tree;

pub use eir::{Symbol, SymbolTree};

#[derive(Debug, Clone)]
pub struct Terminal {
    pub word: String,
    pub lemma: String,
}
