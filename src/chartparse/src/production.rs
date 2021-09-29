use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use itertools::Itertools;
use smallvec::SmallVec;

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct NonTerminal<S: Hash + Clone + PartialEq + Eq> {
    pub symbol: S,
}

impl<S: Hash + Clone + PartialEq + Eq> NonTerminal<S> {
    pub(crate) fn to_symbol(self) -> SymbolWrapper<S> {
        SymbolWrapper {
            inner: Symbol::_NT(self),
        }
    }
}

impl<S: Hash + Clone + PartialEq + Eq> NonTerminal<S> {
    fn new(symbol: S) -> Self {
        NonTerminal { symbol }
    }
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Debug for NonTerminal<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "NonTerminal::{}", self.symbol)
    }
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Display for NonTerminal<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Terminal<S: Hash + Clone + PartialEq + Eq> {
    pub symbol: S,
}

impl<S: Hash + Clone + PartialEq + Eq> Terminal<S> {
    fn new(symbol: S) -> Self {
        Terminal { symbol }
    }
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Debug for Terminal<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Terminal::{{ \"{}\" }}", self.symbol)
    }
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Display for Terminal<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "\"{}\"", self.symbol)
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub enum Symbol<S: Hash + Clone + PartialEq + Eq> {
    _NT(NonTerminal<S>),
    _T(Terminal<S>),
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct SymbolWrapper<S: Hash + Clone + PartialEq + Eq> {
    pub inner: Symbol<S>,
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Debug for Symbol<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::_NT(nt) => write!(f, "{:?}", nt),
            Symbol::_T(t) => write!(f, "{:?}", t),
        }
    }
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Debug for SymbolWrapper<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}

// Correct
impl<S: Hash + Clone + PartialEq + Eq> Symbol<S> {
    fn is_nonterminal(&self) -> bool {
        matches!(self, Symbol::_NT(_))
    }

    fn is_terminal(&self) -> bool {
        matches!(self, Symbol::_T(_))
    }
}

impl<S: Hash + Clone + PartialEq + Eq> SymbolWrapper<S> {
    pub(crate) fn nonterminal(symbol: S) -> Self {
        Self {
            inner: Symbol::_NT(NonTerminal { symbol }),
        }
    }

    pub(crate) fn terminal(symbol: S) -> Self {
        Self {
            inner: Symbol::_T(Terminal { symbol }),
        }
    }
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Display for SymbolWrapper<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            Symbol::_NT(nt) => write!(f, "{}", nt.symbol),
            Symbol::_T(t) => write!(f, "\"{}\"", t.symbol),
        }
    }
}

impl<S: Hash + Clone + PartialEq + Eq> SymbolWrapper<S> {
    pub(crate) fn is_nonterminal(&self) -> bool {
        self.inner.is_nonterminal()
    }

    pub(crate) fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    pub(crate) fn as_terminal(&self) -> Result<Terminal<S>, &'static str> {
        match &self.inner {
            Symbol::_NT(_) => Err("SymbolWrapper::NonTerminal can not be converted to a terminal"),
            Symbol::_T(t) => Ok(t.clone()),
        }
    }

    pub(crate) fn as_nonterminal(&self) -> Result<NonTerminal<S>, &'static str> {
        match &self.inner {
            Symbol::_T(_) => Err("Called as_nonterminal on Terminal"),
            Symbol::_NT(nt) => Ok(nt.clone()),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Production<S: Hash + Clone + PartialEq + Eq> {
    pub lhs: NonTerminal<S>,
    pub rhs: SmallVec<[SymbolWrapper<S>; 6]>,
}

// CORRECT
impl<S: Hash + Clone + PartialEq + Eq> Production<S> {
    pub fn lhs(&self) -> &NonTerminal<S> {
        &self.lhs
    }
    pub fn rhs(&self) -> &[SymbolWrapper<S>] {
        &self.rhs
    }

    pub fn new(lhs: NonTerminal<S>, rhs: Vec<SymbolWrapper<S>>) -> Self {
        Self {
            lhs,
            rhs: rhs.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.rhs.len()
    }

    pub fn is_nonlexical(&self) -> bool {
        self.rhs.iter().all(SymbolWrapper::is_nonterminal)
    }

    pub fn is_lexical(&self) -> bool {
        !self.is_nonlexical()
    }
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Display for Production<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {}",
            self.lhs.symbol,
            self.rhs.iter().map(|x| x.to_string()).join(" ")
        )
    }
}
