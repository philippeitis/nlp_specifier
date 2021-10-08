use std::fmt::{Display, Formatter};
use std::ops::Index;

use itertools::Itertools;

use crate::production::Symbol;

#[derive(Clone)]
pub enum Tree<T, N> {
    Terminal(T),
    Branch(N, Vec<Tree<T, N>>),
}

impl<T: Display, N: Display> Display for Tree<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Tree::Terminal(t) => f.write_str(&t.to_string()),
            Tree::Branch(nt, children) => {
                let next_trees = children.iter().map(|x| x.to_string()).join("\n  ");
                write!(f, "({} {})", nt.to_string(), next_trees)
            }
        }
    }
}

impl<T, N> Tree<T, N> {
    pub(crate) fn from_nonterminal(lhs: N, v: Vec<Tree<T, N>>) -> Self {
        Self::Branch(lhs, v)
    }

    pub(crate) fn from_terminal(lhs: T) -> Self {
        Tree::Terminal(lhs)
    }

    pub(crate) fn extend(&mut self, v: Vec<Tree<T, N>>) {
        if let Tree::Branch(_, rhs) = self {
            rhs.extend(v);
        } else {
            panic!("Can not extend Terminal node");
        }
    }

    pub fn unwrap_terminal(self) -> T {
        match self {
            Tree::Terminal(t) => t,
            Tree::Branch(_, _) => panic!("Called unwrap_terminal with non-terminal Tree"),
        }
    }

    pub fn unwrap_branch(self) -> (N, Vec<Tree<T, N>>) {
        match self {
            Tree::Terminal(_) => panic!("Called unwrap_branch with terminal Tree"),
            Tree::Branch(nt, trees) => (nt, trees),
        }
    }
}

impl<T, N> From<Symbol<T, N>> for Tree<T, N> {
    fn from(symbol: Symbol<T, N>) -> Self {
        match symbol {
            Symbol::Terminal(t) => Tree::Terminal(t),
            Symbol::NonTerminal(nt) => Tree::Branch(nt, vec![]),
        }
    }
}

impl<T, N> Index<&[usize]> for Tree<T, N> {
    type Output = Tree<T, N>;

    fn index(&self, index: &[usize]) -> &Self::Output {
        if index.len() == 0 {
            return self;
        }

        match self {
            Tree::Terminal(_) => panic!("Invalid index {:?} for Terminal", index),
            Tree::Branch(_, branch) => branch[index[0]].index(&index[1..]),
        }
    }
}
