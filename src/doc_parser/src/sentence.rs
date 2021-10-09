use ndarray::Array1;
use ndarray_linalg::Norm;

use chartparse::ChartParser;
use chartparse::grammar::ParseTerminal;

use crate::parse_tree::tree::TerminalSymbol;
use crate::parse_tree::{Symbol, SymbolTree, Terminal};

#[derive(Clone)]
pub struct Token {
    pub tag: TerminalSymbol,
    pub word: String,
    pub lemma: String,
}

impl From<(String, String, String)> for Token {
    fn from(s: (String, String, String)) -> Self {
        Self {
            tag: TerminalSymbol::parse_terminal(&s.0).unwrap(),
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

    /// Produces the parse trees for this sentence's tokens, using the provided parser.
    /// Whitespace tokens, and trailing DOT tokens are filtered.
    pub fn parse_trees(&self, parser: &ChartParser<TerminalSymbol, Symbol>) -> Vec<SymbolTree> {
        let tokens = {
            let mut tokens: Vec<_> = self
                .tokens
                .iter()
                .filter(|x| x.tag != TerminalSymbol::SPACE)
                .map(|x| x.tag)
                .collect();
            while tokens.last() == Some(&TerminalSymbol::DOT) {
                tokens.pop();
            }
            tokens
        };

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
