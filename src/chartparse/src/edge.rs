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

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TreeEdge<N, T> {
    span: Span,
    dot: usize,
    lhs: N,
    rhs: SmallVec<[Symbol<N, T>; 6]>,
}

impl<N: Clone, T> TreeEdge<N, T> {
    fn lhs(&self) -> Symbol<N, T> {
        Symbol::NonTerminal(self.lhs.clone())
    }
}

impl<N, T> TreeEdge<N, T> {
    fn length(&self) -> usize {
        self.span.end - self.span.start
    }

    fn span(&self) -> Span {
        self.span
    }

    fn rhs(&self) -> &[Symbol<N, T>] {
        &self.rhs
    }

    fn dot(&self) -> usize {
        self.dot
    }

    fn next_sym(&self) -> Option<&Symbol<N, T>> {
        self.rhs.get(self.dot)
    }

    fn is_complete(&self) -> bool {
        self.dot == self.rhs.len()
    }
}

impl<N: Clone, T: Clone> TreeEdge<N, T> {
    pub(crate) fn new(span: Span, lhs: N, rhs: SmallVec<[Symbol<N, T>; 6]>) -> Self {
        Self::with_dot(span, lhs, rhs, 0)
    }

    pub(crate) fn with_dot(
        span: Span,
        lhs: N,
        rhs: SmallVec<[Symbol<N, T>; 6]>,
        dot: usize,
    ) -> Self {
        TreeEdge {
            span,
            dot,
            lhs,
            rhs,
        }
    }

    pub(crate) fn from_production(production: Production<N, T>, index: usize) -> Self {
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

impl<N: Clone, T: Clone> TreeEdge<N, T> {
    pub(crate) fn move_dot_forward(&self, new_end: usize) -> Self {
        TreeEdge::with_dot(
            Span {
                start: self.span.start,
                end: new_end,
            },
            self.lhs.clone(),
            self.rhs.clone(),
            self.dot + 1,
        )
    }
}

impl<N: Display, T: Display> Display for TreeEdge<N, T> {
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

        write!(
            f,
            "[{}:{}] {}",
            self.span.start, self.span.end, arrow_string
        )
    }
}

impl<N: Display, T: Display> Debug for TreeEdge<N, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Edge: {}]", self.to_string())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct LeafEdge<N, T> {
    leaf: T,
    index: usize,
    nonterm: PhantomData<N>,
}

impl<N, T: Clone> LeafEdge<N, T> {
    fn lhs(&self) -> Symbol<N, T> {
        Symbol::Terminal(self.leaf.clone())
    }
}

impl<N, T> LeafEdge<N, T> {
    pub fn new(leaf: T, index: usize) -> Self {
        Self {
            leaf,
            index,
            nonterm: PhantomData,
        }
    }

    fn span(&self) -> Span {
        Span {
            start: self.index,
            end: self.index + 1,
        }
    }

    fn rhs(&self) -> &[Symbol<N, T>] {
        &[]
    }

    fn dot(&self) -> usize {
        0
    }

    fn next_sym(&self) -> Option<&Symbol<N, T>> {
        None
    }

    fn is_complete(&self) -> bool {
        true
    }
}

impl<N: Display, T: Display> Display for LeafEdge<N, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}:{}] {}",
            self.span().start,
            self.span().end,
            self.leaf.to_string()
        )
    }
}

impl<N: Display, T: Display> Debug for LeafEdge<N, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Edge: {}]", self.to_string())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Edge<N, T> {
    _L(LeafEdge<N, T>),
    _T(TreeEdge<N, T>),
}

impl<N: Clone, T: Clone> Edge<N, T> {
    pub fn lhs(&self) -> Symbol<N, T> {
        match self {
            Edge::_L(l) => l.lhs(),
            Edge::_T(t) => t.lhs(),
        }
    }
}

impl<N, T> Edge<N, T> {
    pub fn start(&self) -> usize {
        self.span().start
    }

    pub fn end(&self) -> usize {
        self.span().end
    }

    pub fn length(&self) -> usize {
        self.end() - self.start()
    }

    pub fn span(&self) -> Span {
        match self {
            Edge::_L(l) => l.span(),
            Edge::_T(t) => t.span(),
        }
    }

    pub fn rhs(&self) -> &[Symbol<N, T>] {
        match self {
            Edge::_L(l) => l.rhs(),
            Edge::_T(t) => t.rhs(),
        }
    }

    pub fn dot(&self) -> usize {
        match self {
            Edge::_L(l) => l.dot(),
            Edge::_T(t) => t.dot(),
        }
    }

    pub fn next_sym(&self) -> Option<&Symbol<N, T>> {
        match self {
            Edge::_L(l) => l.next_sym(),
            Edge::_T(t) => t.next_sym(),
        }
    }

    pub fn is_complete(&self) -> bool {
        match self {
            Edge::_L(l) => l.is_complete(),
            Edge::_T(t) => t.is_complete(),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct EdgeWrapper<N, T> {
    pub(crate) inner: Rc<Edge<N, T>>,
    inner_hash: u64,
}

impl<N: Hash, T: Hash> Hash for EdgeWrapper<N, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner_hash.hash(state);
    }
}

impl<N: Clone + Hash, T: Clone + Hash> EdgeWrapper<N, T> {
    pub(crate) fn is_leafedge(&self) -> bool {
        matches!(self.inner.as_ref(), Edge::_L(_))
    }

    pub(crate) fn start(&self) -> usize {
        self.inner.start()
    }

    pub(crate) fn end(&self) -> usize {
        self.inner.end()
    }

    pub(crate) fn lhs(&self) -> Symbol<N, T> {
        self.inner.lhs()
    }

    pub(crate) fn rhs(&self) -> &[Symbol<N, T>] {
        self.inner.rhs()
    }

    pub(crate) fn dot(&self) -> usize {
        self.inner.dot()
    }

    pub(crate) fn next_sym(&self) -> Option<&Symbol<N, T>> {
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

impl<N: Hash, T: Hash> From<TreeEdge<N, T>> for EdgeWrapper<N, T> {
    fn from(t: TreeEdge<N, T>) -> Self {
        let inner = Rc::new(Edge::_T(t));
        let mut hasher = DefaultHasher::new();
        inner.hash(&mut hasher);
        let inner_hash = hasher.finish();
        Self { inner, inner_hash }
    }
}

impl<N: Hash, T: Hash> From<LeafEdge<N, T>> for EdgeWrapper<N, T> {
    fn from(l: LeafEdge<N, T>) -> Self {
        let inner = Rc::new(Edge::_L(l));
        let mut hasher = DefaultHasher::new();
        inner.hash(&mut hasher);
        let inner_hash = hasher.finish();
        Self { inner, inner_hash }
    }
}

impl<N: Display, T: Display> Display for Edge<N, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Edge::_L(l) => f.write_str(&l.to_string()),
            Edge::_T(t) => f.write_str(&t.to_string()),
        }
    }
}

impl<N: Display, T: Display> Display for EdgeWrapper<N, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}
