use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::ops::Index;

use itertools::Itertools;

use crate::production::{NonTerminal, SymbolWrapper, Terminal};

#[derive(Clone)]
pub enum TreeNode<S: Hash + Clone + PartialEq + Eq> {
    Terminal(Terminal<S>),
    Branch(NonTerminal<S>, Vec<TreeWrapper<S>>),
}

#[derive(Clone)]
pub struct TreeWrapper<S: Hash + Clone + PartialEq + Eq> {
    pub inner: TreeNode<S>,
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Display for TreeWrapper<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            TreeNode::Terminal(t) => f.write_str(&t.to_string()),
            TreeNode::Branch(nt, children) => {
                let next_trees = children.iter().map(|x| x.to_string()).join("\n  ");
                write!(f, "({} {})", nt.to_string(), next_trees)
            }
        }
    }
}

impl<S: Hash + Clone + PartialEq + Eq> TreeNode<S> {
    fn from_list(lhs: SymbolWrapper<S>, v: Vec<TreeWrapper<S>>) -> Self {
        assert!(!v.is_empty());
        Self::Branch(lhs.as_nonterminal().unwrap(), v)
    }

    fn from_terminal(lhs: SymbolWrapper<S>) -> Self {
        Self::Terminal(lhs.as_terminal().unwrap())
    }
}

impl<S: Hash + Clone + PartialEq + Eq> TreeWrapper<S> {
    pub(crate) fn from_list(lhs: SymbolWrapper<S>, v: Vec<TreeWrapper<S>>) -> Self {
        TreeWrapper {
            inner: TreeNode::from_list(lhs, v),
        }
    }

    pub(crate) fn from_terminal(lhs: SymbolWrapper<S>) -> Self {
        TreeWrapper {
            inner: TreeNode::from_terminal(lhs),
        }
    }

    pub(crate) fn extend(&mut self, v: Vec<TreeWrapper<S>>) {
        if let TreeNode::Branch(_, rhs) = &mut self.inner {
            rhs.extend(v);
        } else {
            panic!("Can not extend Terminal node");
        }
    }

    pub fn unwrap_terminal(self) -> Terminal<S> {
        match self.inner {
            TreeNode::Terminal(t) => t,
            TreeNode::Branch(_, _) => panic!("Called unwrap_terminal with non-terminal Tree"),
        }
    }

    pub fn unwrap_branch(self) -> (NonTerminal<S>, Vec<TreeWrapper<S>>) {
        match self.inner {
            TreeNode::Terminal(_) => panic!("Called unwrap_branch with terminal Tree"),
            TreeNode::Branch(nt, trees) => (nt, trees),
        }
    }

    pub fn label(&self) -> &S {
        match &self.inner {
            TreeNode::Terminal(t) => &t.symbol,
            TreeNode::Branch(nt, _) => &nt.symbol,
        }
    }
}

impl<S: Hash + Clone + PartialEq + Eq> Index<&[usize]> for TreeWrapper<S> {
    type Output = TreeWrapper<S>;

    fn index(&self, index: &[usize]) -> &Self::Output {
        if index.len() == 0 {
            return self;
        }

        match &self.inner {
            TreeNode::Terminal(_) => panic!("Invalid index {:?} for Terminal", index),
            TreeNode::Branch(_, branch) => branch[index[0]].index(&index[1..]),
        }
    }
}
