use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use itertools::Itertools;
use smallvec::SmallVec;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Symbol<T, N> {
    Terminal(T),
    NonTerminal(N),
}

impl<T: Display, N: Display> Debug for Symbol<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::NonTerminal(nt) => write!(f, "NonTerminal::{}", nt),
            Symbol::Terminal(t) => write!(f, "Terminal::{{ \"{}\" }}", t),
        }
    }
}

impl<T: Display, N: Display> Display for Symbol<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::NonTerminal(nt) => write!(f, "{}", nt),
            Symbol::Terminal(t) => write!(f, "\"{}\"", t),
        }
    }
}

// Correct
impl<T, N> Symbol<T, N> {
    pub fn is_nonterminal(&self) -> bool {
        matches!(self, Symbol::NonTerminal(_))
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Symbol::Terminal(_))
    }
}

impl<T, N> Symbol<T, N> {
    pub(crate) fn terminal(self) -> Option<T> {
        match self {
            Symbol::Terminal(t) => Some(t),
            Symbol::NonTerminal(_) => None,
        }
    }

    pub(crate) fn nonterminal(self) -> Option<N> {
        match self {
            Symbol::Terminal(_) => None,
            Symbol::NonTerminal(nt) => Some(nt),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Production<T, N> {
    pub lhs: N,
    pub rhs: SmallVec<[Symbol<T, N>; 6]>,
}

impl<T, N> Production<T, N> {
    pub fn new(lhs: N, rhs: Vec<Symbol<T, N>>) -> Self {
        Self {
            lhs,
            rhs: rhs.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.rhs.len()
    }

    pub fn is_nonlexical(&self) -> bool {
        self.rhs.iter().all(Symbol::is_nonterminal)
    }

    pub fn is_lexical(&self) -> bool {
        !self.is_nonlexical()
    }
}

impl<T: Display, N: Display> Display for Production<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {}",
            self.lhs,
            self.rhs.iter().map(|x| x.to_string()).join(" ")
        )
    }
}
