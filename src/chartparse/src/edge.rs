use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use smallvec::SmallVec;

use crate::production::{Production, Symbol};
use std::collections::hash_map::DefaultHasher;
use std::marker::PhantomData;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Span {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

pub(crate) trait EdgeI<T: Clone, N: Clone>: Clone {
    fn span(&self) -> Span;

    fn start(&self) -> usize {
        self.span().start
    }

    fn end(&self) -> usize {
        self.span().end
    }

    fn length(&self) -> usize {
        self.span().end - self.span().start
    }

    fn lhs(&self) -> Symbol<T, N>;

    fn rhs(&self) -> &[Symbol<T, N>];

    fn dot(&self) -> usize;

    fn next_sym(&self) -> Option<&Symbol<T, N>>;

    fn is_complete(&self) -> bool;
}

pub struct TreeEdge<T, N> {
    span: Span,
    dot: usize,
    lhs: N,
    rhs: SmallVec<[Symbol<T, N>; 6]>,
}

impl<T: Clone, N: Clone> Clone for TreeEdge<T, N> {
    fn clone(&self) -> Self {
        Self {
            span: self.span.clone(),
            dot: self.dot,
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
        }
    }
}

impl<T: PartialEq, N: PartialEq> PartialEq for TreeEdge<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.span == other.span
            && self.dot == other.dot
            && self.lhs == other.lhs
            && self.rhs == other.rhs
    }
}

impl<T: PartialEq + Eq, N: PartialEq + Eq> Eq for TreeEdge<T, N> {}

impl<T: Hash, N: Hash> Hash for TreeEdge<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.span.hash(state);
        self.dot.hash(state);
        self.lhs.hash(state);
        self.rhs.hash(state);
    }
}

impl<T: Clone, N: Clone> EdgeI<T, N> for TreeEdge<T, N> {
    fn span(&self) -> Span {
        self.span
    }

    fn lhs(&self) -> Symbol<T, N> {
        Symbol::NonTerminal(self.lhs.clone())
    }

    fn rhs(&self) -> &[Symbol<T, N>] {
        &self.rhs
    }

    fn dot(&self) -> usize {
        self.dot
    }

    fn next_sym(&self) -> Option<&Symbol<T, N>> {
        self.rhs.get(self.dot)
    }

    fn is_complete(&self) -> bool {
        self.dot == self.rhs.len()
    }
}

impl<T: Clone, N: Clone> TreeEdge<T, N> {
    pub(crate) fn new(span: Span, lhs: N, rhs: SmallVec<[Symbol<T, N>; 6]>) -> Self {
        Self::with_dot(span, lhs, rhs, 0)
    }

    pub(crate) fn with_dot(
        span: Span,
        lhs: N,
        rhs: SmallVec<[Symbol<T, N>; 6]>,
        dot: usize,
    ) -> Self {
        TreeEdge {
            span,
            dot,
            lhs,
            rhs,
        }
    }

    pub(crate) fn from_production(production: Production<T, N>, index: usize) -> Self {
        TreeEdge::new(
            Span {
                start: index,
                end: index,
            },
            production.lhs.clone(),
            production.rhs.clone(),
        )
    }
}

impl<T: Clone, N: Clone> TreeEdge<T, N> {
    pub(crate) fn move_dot_forward(&self, new_end: usize) -> Self {
        TreeEdge::with_dot(
            Span {
                start: self.start(),
                end: new_end,
            },
            self.lhs.clone(),
            self.rhs.clone(),
            self.dot + 1,
        )
    }
}

impl<T: Display + Clone, N: Display + Clone> Display for TreeEdge<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut arrow_string = self.lhs.to_string();
        arrow_string.push_str(" ->");
        for (i, val) in self.rhs.iter().enumerate() {
            if i == self.dot {
                arrow_string.push_str(&format!(" * {}", val.to_string()));
            } else {
                arrow_string.push_str(&format!(" {}", val.to_string()));
            }
        }

        if self.dot == self.length() {
            arrow_string.push(' ');
            arrow_string.push('*');
        }

        write!(f, "[{}:{}] {}", self.start(), self.end(), arrow_string)
    }
}

impl<T: Display + Clone, N: Display + Clone> Debug for TreeEdge<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Edge: {}]", self.to_string())
    }
}

pub struct LeafEdge<T, N> {
    leaf: T,
    index: usize,
    nonterm: PhantomData<N>,
}

impl<T, N> LeafEdge<T, N> {
    pub fn new(leaf: T, index: usize) -> Self {
        Self {
            leaf,
            index,
            nonterm: PhantomData,
        }
    }
}

impl<T: Clone, N: Clone> Clone for LeafEdge<T, N> {
    fn clone(&self) -> Self {
        Self {
            leaf: self.leaf.clone(),
            index: self.index.clone(),
            nonterm: PhantomData,
        }
    }
}

impl<T: PartialEq, N: PartialEq> PartialEq for LeafEdge<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.leaf == other.leaf && self.index == other.index
    }
}

impl<T: PartialEq + Eq, N: PartialEq + Eq> Eq for LeafEdge<T, N> {}

impl<T: Hash, N: Hash> Hash for LeafEdge<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.leaf.hash(state);
        self.index.hash(state);
    }
}

impl<T: Clone, N: Clone> EdgeI<T, N> for LeafEdge<T, N> {
    fn span(&self) -> Span {
        Span {
            start: self.index,
            end: self.index + 1,
        }
    }

    fn lhs(&self) -> Symbol<T, N> {
        Symbol::Terminal(self.leaf.clone())
    }

    fn rhs(&self) -> &[Symbol<T, N>] {
        &[]
    }

    fn dot(&self) -> usize {
        0
    }

    fn next_sym(&self) -> Option<&Symbol<T, N>> {
        None
    }

    fn is_complete(&self) -> bool {
        true
    }
}

impl<T: Display + Clone, N: Display + Clone> Display for LeafEdge<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}:{}] {}",
            self.start(),
            self.end(),
            self.leaf.to_string()
        )
    }
}

impl<T: Display + Clone, N: Display + Clone> Debug for LeafEdge<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Edge: {}]", self.to_string())
    }
}

pub enum Edge<T, N> {
    _L(LeafEdge<T, N>),
    _T(TreeEdge<T, N>),
}

impl<T: Clone, N: Clone> Clone for Edge<T, N> {
    fn clone(&self) -> Self {
        match self {
            Edge::_L(e) => Edge::_L(e.clone()),
            Edge::_T(e) => Edge::_T(e.clone()),
        }
    }
}

impl<T: PartialEq, N: PartialEq> PartialEq for Edge<T, N> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Edge::_L(sl), Edge::_L(ol)) => sl == ol,
            (Edge::_T(st), Edge::_T(ot)) => st == ot,
            _ => false,
        }
    }
}

impl<T: PartialEq + Eq, N: PartialEq + Eq> Eq for Edge<T, N> {}

impl<T: Hash, N: Hash> Hash for Edge<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Edge::_L(l) => {
                0u8.hash(state);
                l.hash(state);
            }
            Edge::_T(t) => {
                1u8.hash(state);
                t.hash(state);
            }
        }
    }
}

impl<T: Clone, N: Clone> EdgeI<T, N> for Edge<T, N> {
    fn span(&self) -> Span {
        match self {
            Edge::_L(l) => l.span(),
            Edge::_T(t) => t.span(),
        }
    }

    fn lhs(&self) -> Symbol<T, N> {
        match self {
            Edge::_L(l) => l.lhs(),
            Edge::_T(t) => t.lhs(),
        }
    }

    fn rhs(&self) -> &[Symbol<T, N>] {
        match self {
            Edge::_L(l) => l.rhs(),
            Edge::_T(t) => t.rhs(),
        }
    }

    fn dot(&self) -> usize {
        match self {
            Edge::_L(l) => l.dot(),
            Edge::_T(t) => t.dot(),
        }
    }

    fn next_sym(&self) -> Option<&Symbol<T, N>> {
        match self {
            Edge::_L(l) => l.next_sym(),
            Edge::_T(t) => t.next_sym(),
        }
    }

    fn is_complete(&self) -> bool {
        match self {
            Edge::_L(l) => l.is_complete(),
            Edge::_T(t) => t.is_complete(),
        }
    }
}

pub struct EdgeWrapper<T, N> {
    pub(crate) inner: Rc<Edge<T, N>>,
    inner_hash: u64,
}

impl<T: Clone, N: Clone> Clone for EdgeWrapper<T, N> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            inner_hash: self.inner_hash,
        }
    }
}

impl<T: PartialEq, N: PartialEq> PartialEq for EdgeWrapper<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.inner_hash == other.inner_hash && self.inner == other.inner
    }
}

impl<T: PartialEq + Eq, N: PartialEq + Eq> Eq for EdgeWrapper<T, N> {}

impl<T: Hash, N: Hash> Hash for EdgeWrapper<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner_hash.hash(state);
    }
}

impl<T: Clone + Hash, N: Clone + Hash> EdgeWrapper<T, N> {
    pub(crate) fn is_leafedge(&self) -> bool {
        matches!(self.inner.as_ref(), Edge::_L(_))
    }

    fn span(&self) -> (usize, usize) {
        (self.start(), self.end())
    }

    pub(crate) fn start(&self) -> usize {
        self.inner.start()
    }

    pub(crate) fn end(&self) -> usize {
        self.inner.end()
    }

    pub(crate) fn lhs(&self) -> Symbol<T, N> {
        self.inner.lhs()
    }

    pub(crate) fn rhs(&self) -> &[Symbol<T, N>] {
        self.inner.rhs()
    }

    pub(crate) fn dot(&self) -> usize {
        self.inner.dot()
    }

    pub(crate) fn next_sym(&self) -> Option<&Symbol<T, N>> {
        self.inner.next_sym()
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    pub(crate) fn move_dot_forward(&self, new_end: usize) -> Result<Self, &'static str> {
        match self.inner.as_ref() {
            Edge::_L(_) => Err("Can not move dot forward on LeafEdge"),
            Edge::_T(t) => Ok(t.move_dot_forward(new_end).into()),
        }
    }
}

impl<T: Hash, N: Hash> From<TreeEdge<T, N>> for EdgeWrapper<T, N> {
    fn from(t: TreeEdge<T, N>) -> Self {
        let inner = Rc::new(Edge::_T(t));
        let mut hasher = DefaultHasher::new();
        inner.hash(&mut hasher);
        let inner_hash = hasher.finish();
        Self { inner, inner_hash }
    }
}

impl<T: Hash, N: Hash> From<LeafEdge<T, N>> for EdgeWrapper<T, N> {
    fn from(l: LeafEdge<T, N>) -> Self {
        let inner = Rc::new(Edge::_L(l));
        let mut hasher = DefaultHasher::new();
        inner.hash(&mut hasher);
        let inner_hash = hasher.finish();
        Self { inner, inner_hash }
    }
}

impl<T: Display + Clone, N: Display + Clone> Display for Edge<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Edge::_L(l) => f.write_str(&l.to_string()),
            Edge::_T(t) => f.write_str(&t.to_string()),
        }
    }
}

impl<T: Display + Clone, N: Display + Clone> Display for EdgeWrapper<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}
