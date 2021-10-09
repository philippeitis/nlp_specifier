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
pub struct TreeEdge<T, N> {
    span: Span,
    dot: usize,
    lhs: N,
    rhs: SmallVec<[Symbol<T, N>; 6]>,
}

impl<T, N: Clone> TreeEdge<T, N> {
    fn lhs(&self) -> Symbol<T, N> {
        Symbol::NonTerminal(self.lhs.clone())
    }
}

impl<T, N> TreeEdge<T, N> {
    fn length(&self) -> usize {
        self.span.end - self.span.start
    }

    fn span(&self) -> Span {
        self.span
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
                start: self.span.start,
                end: new_end,
            },
            self.lhs.clone(),
            self.rhs.clone(),
            self.dot + 1,
        )
    }
}

impl<T: Display, N: Display> Display for TreeEdge<T, N> {
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

impl<T: Display, N: Display> Debug for TreeEdge<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Edge: {}]", self.to_string())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct LeafEdge<T, N> {
    leaf: T,
    index: usize,
    nonterm: PhantomData<N>,
}

impl<T: Clone, N> LeafEdge<T, N> {
    fn lhs(&self) -> Symbol<T, N> {
        Symbol::Terminal(self.leaf.clone())
    }
}

impl<T, N> LeafEdge<T, N> {
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

impl<T: Display, N: Display> Display for LeafEdge<T, N> {
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

impl<T: Display, N: Display> Debug for LeafEdge<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Edge: {}]", self.to_string())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Edge<T, N> {
    _L(LeafEdge<T, N>),
    _T(TreeEdge<T, N>),
}

impl<T: Clone, N: Clone> Edge<T, N> {
    pub fn lhs(&self) -> Symbol<T, N> {
        match self {
            Edge::_L(l) => l.lhs(),
            Edge::_T(t) => t.lhs(),
        }
    }
}

impl<T, N> Edge<T, N> {
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

    pub fn rhs(&self) -> &[Symbol<T, N>] {
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

    pub fn next_sym(&self) -> Option<&Symbol<T, N>> {
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
pub struct EdgeWrapper<T, N> {
    pub(crate) inner: Rc<Edge<T, N>>,
    inner_hash: u64,
}

impl<T: Hash, N: Hash> Hash for EdgeWrapper<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner_hash.hash(state);
    }
}

impl<T: Clone + Hash, N: Clone + Hash> EdgeWrapper<T, N> {
    pub(crate) fn is_leafedge(&self) -> bool {
        matches!(self.inner.as_ref(), Edge::_L(_))
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

impl<T: Display, N: Display> Display for Edge<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Edge::_L(l) => f.write_str(&l.to_string()),
            Edge::_T(t) => f.write_str(&t.to_string()),
        }
    }
}

impl<T: Display, N: Display> Display for EdgeWrapper<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}
