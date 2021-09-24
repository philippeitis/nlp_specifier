mod eir;
pub mod tree;

pub use eir::{SymbolTree, Symbol};

#[derive(Debug, Clone)]
pub struct Terminal {
    pub word: String,
    pub lemma: String,
}