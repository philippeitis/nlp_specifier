use ndarray::Array1;
use ndarray_linalg::Norm;

use crate::parse_tree::tree::TerminalSymbol;
use crate::parse_tree::{Symbol, SymbolTree, Terminal};
use chartparse::ChartParser;

#[derive(Clone)]
pub struct Token {
    pub tag: TerminalSymbol,
    pub word: String,
    pub lemma: String,
}

impl From<(String, String, String)> for Token {
    fn from(s: (String, String, String)) -> Self {
        Self {
            tag: TerminalSymbol::from_terminal(s.0).unwrap(),
            word: s.1,
            lemma: s.2,
        }
    }
}

pub struct Sentence {
    pub text: String,
    pub tokens: Vec<Token>,
    pub vector: Array1<f32>,
}

impl Sentence {
    pub fn new(text: String, tokens: Vec<Token>, vector: Vec<f32>) -> Self {
        let vector = Array1::from_vec(vector);
        let norm = vector.norm_l2();
        Sentence {
            text,
            tokens,
            vector: vector / norm,
        }
    }

    pub fn similarity(&self, other: &Sentence) -> f32 {
        self.vector.dot(&other.vector)
    }

    pub fn parse_trees(&self, parser: &ChartParser<Symbol>) -> Vec<SymbolTree> {
        let tokens: Vec<_> = self.tokens.iter().map(|x| Symbol::from(x.tag)).collect();

        let terminals: Vec<_> = self
            .tokens
            .iter()
            .cloned()
            .map(|token| Terminal {
                word: token.word.clone(),
                lemma: token.lemma.to_lowercase(),
            })
            .collect();
        match parser.parse(&tokens) {
            Ok(trees) => trees
                .into_iter()
                .map(|t| SymbolTree::from_iter(t, &mut terminals.clone().into_iter()))
                .collect(),
            Err(_) => return Vec::new(),
        }
    }
}
