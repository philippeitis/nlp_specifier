use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use itertools::Itertools;
use smallvec::SmallVec;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Symbol<N, T> {
    NonTerminal(N),
    Terminal(T),
}

impl<N: Display, T: Display> Debug for Symbol<N, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::NonTerminal(nt) => write!(f, "NonTerminal::{}", nt),
            Symbol::Terminal(t) => write!(f, "Terminal::{{ \"{}\" }}", t),
        }
    }
}

impl<N: Display, T: Display> Display for Symbol<N, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::NonTerminal(nt) => write!(f, "{}", nt),
            Symbol::Terminal(t) => write!(f, "\"{}\"", t),
        }
    }
}

impl<N, T> Symbol<N, T> {
    pub fn is_nonterminal(&self) -> bool {
        matches!(self, Symbol::NonTerminal(_))
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Symbol::Terminal(_))
    }
}

impl<N, T> Symbol<N, T> {
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
pub struct Production<N, T> {
    pub lhs: N,
    pub rhs: SmallVec<[Symbol<N, T>; 6]>,
}

impl<N, T> Production<N, T> {
    pub fn new(lhs: N, rhs: Vec<Symbol<N, T>>) -> Self {
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

impl<N: Display, T: Display> Display for Production<N, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {}",
            self.lhs,
            self.rhs.iter().map(|x| x.to_string()).join(" ")
        )
    }
}

impl<N: Display, T: Display> Debug for Production<N, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}
