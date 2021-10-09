use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

use itertools::Itertools;
use smallvec::SmallVec;

pub enum Symbol<T, N> {
    Terminal(T),
    NonTerminal(N),
}

impl<T: Clone, N: Clone> Clone for Symbol<T, N> {
    fn clone(&self) -> Self {
        match &self {
            Symbol::Terminal(t) => Symbol::Terminal(t.clone()),
            Symbol::NonTerminal(nt) => Symbol::NonTerminal(nt.clone()),
        }
    }
}

impl<T: PartialEq, N: PartialEq> PartialEq for Symbol<T, N> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Symbol::Terminal(st), Symbol::Terminal(ot)) => st == ot,
            (Symbol::NonTerminal(snt), Symbol::NonTerminal(ont)) => snt == ont,
            _ => false,
        }
    }
}

impl<T: PartialEq + Eq, N: PartialEq + Eq> Eq for Symbol<T, N> {}

impl<T: Hash, N: Hash> Hash for Symbol<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Symbol::Terminal(t) => {
                0u8.hash(state);
                t.hash(state);
            }
            Symbol::NonTerminal(nt) => {
                1u8.hash(state);
                nt.hash(state);
            }
        }
    }
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

pub struct Production<T, N> {
    pub lhs: N,
    pub rhs: SmallVec<[Symbol<T, N>; 6]>,
}

impl<T: Clone, N: Clone> Clone for Production<T, N> {
    fn clone(&self) -> Self {
        Production {
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
        }
    }
}

impl<T: PartialEq, N: PartialEq> PartialEq for Production<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.lhs == other.lhs && self.rhs == other.rhs
    }
}

impl<T: PartialEq + Eq, N: PartialEq + Eq> Eq for Production<T, N> {}

impl<T: Hash, N: Hash> Hash for Production<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.lhs.hash(state);
        self.rhs.hash(state);
    }
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
