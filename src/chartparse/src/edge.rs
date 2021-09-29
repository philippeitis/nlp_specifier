use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use smallvec::SmallVec;

use crate::production::{Production, SymbolWrapper};
use std::collections::hash_map::DefaultHasher;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Span {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

pub(crate) trait EdgeI<S: Hash + Clone + PartialEq + Eq>: Clone {
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

    fn lhs(&self) -> &SymbolWrapper<S>;

    fn rhs(&self) -> &[SymbolWrapper<S>];

    fn dot(&self) -> usize;

    fn next_sym(&self) -> Option<&SymbolWrapper<S>>;

    fn is_complete(&self) -> bool;
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct TreeEdge<S: Hash + Clone + PartialEq + Eq> {
    span: Span,
    dot: usize,
    lhs: SymbolWrapper<S>,
    rhs: SmallVec<[SymbolWrapper<S>; 6]>,
}

impl<S: Hash + Clone + PartialEq + Eq> EdgeI<S> for TreeEdge<S> {
    fn span(&self) -> Span {
        self.span
    }

    fn lhs(&self) -> &SymbolWrapper<S> {
        &self.lhs
    }

    fn rhs(&self) -> &[SymbolWrapper<S>] {
        &self.rhs
    }

    fn dot(&self) -> usize {
        self.dot
    }

    fn next_sym(&self) -> Option<&SymbolWrapper<S>> {
        self.rhs.get(self.dot)
    }

    fn is_complete(&self) -> bool {
        self.dot == self.rhs.len()
    }
}

impl<S: Hash + Clone + PartialEq + Eq> TreeEdge<S> {
    pub(crate) fn new(
        span: Span,
        lhs: SymbolWrapper<S>,
        rhs: SmallVec<[SymbolWrapper<S>; 6]>,
    ) -> Self {
        Self::with_dot(span, lhs, rhs, 0)
    }

    pub(crate) fn with_dot(
        span: Span,
        lhs: SymbolWrapper<S>,
        rhs: SmallVec<[SymbolWrapper<S>; 6]>,
        dot: usize,
    ) -> Self {
        TreeEdge {
            span,
            dot,
            lhs,
            rhs,
        }
    }

    pub(crate) fn from_production(production: Production<S>, index: usize) -> Self {
        TreeEdge::new(
            Span {
                start: index,
                end: index,
            },
            production.lhs.clone().to_symbol(),
            production.rhs.clone(),
        )
    }
}

impl<S: Hash + Clone + PartialEq + Eq> TreeEdge<S> {
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

impl<S: Hash + Clone + PartialEq + Eq + Display> Display for TreeEdge<S> {
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

impl<S: Hash + Clone + PartialEq + Eq + Display> Debug for TreeEdge<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Edge: {}]", self.to_string())
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct LeafEdge<S: Hash + Clone + PartialEq + Eq> {
    leaf: SymbolWrapper<S>,
    index: usize,
}

impl<S: Hash + Clone + PartialEq + Eq> LeafEdge<S> {
    pub fn new(leaf: SymbolWrapper<S>, index: usize) -> Self {
        Self { leaf, index }
    }
}

impl<S: Hash + Clone + PartialEq + Eq> EdgeI<S> for LeafEdge<S> {
    fn span(&self) -> Span {
        Span {
            start: self.index,
            end: self.index + 1,
        }
    }

    fn lhs(&self) -> &SymbolWrapper<S> {
        &self.leaf
    }

    fn rhs(&self) -> &[SymbolWrapper<S>] {
        &[]
    }

    fn dot(&self) -> usize {
        0
    }

    fn next_sym(&self) -> Option<&SymbolWrapper<S>> {
        None
    }

    fn is_complete(&self) -> bool {
        true
    }
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Display for LeafEdge<S> {
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

impl<S: Hash + Clone + PartialEq + Eq + Display> Debug for LeafEdge<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[Edge: {}]", self.to_string())
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub enum Edge<S: Hash + Clone + PartialEq + Eq> {
    _L(LeafEdge<S>),
    _T(TreeEdge<S>),
}

impl<S: Hash + Clone + PartialEq + Eq> EdgeI<S> for Edge<S> {
    fn span(&self) -> Span {
        match self {
            Edge::_L(l) => l.span(),
            Edge::_T(t) => t.span(),
        }
    }

    fn lhs(&self) -> &SymbolWrapper<S> {
        match self {
            Edge::_L(l) => l.lhs(),
            Edge::_T(t) => t.lhs(),
        }
    }

    fn rhs(&self) -> &[SymbolWrapper<S>] {
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

    fn next_sym(&self) -> Option<&SymbolWrapper<S>> {
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

#[derive(Clone, PartialEq, Eq)]
pub struct EdgeWrapper<S: Hash + Clone + PartialEq + Eq> {
    pub(crate) inner: Rc<Edge<S>>,
    inner_hash: u64,
}

impl<S: Hash + Clone + PartialEq + Eq> Hash for EdgeWrapper<S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner_hash.hash(state);
    }
}

impl<S: Hash + Clone + PartialEq + Eq> EdgeWrapper<S> {
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

    pub(crate) fn lhs(&self) -> &SymbolWrapper<S> {
        self.inner.lhs()
    }

    pub(crate) fn rhs(&self) -> &[SymbolWrapper<S>] {
        self.inner.rhs()
    }

    pub(crate) fn dot(&self) -> usize {
        self.inner.dot()
    }

    pub(crate) fn next_sym(&self) -> Option<&SymbolWrapper<S>> {
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

impl<S: Hash + Clone + PartialEq + Eq> From<TreeEdge<S>> for EdgeWrapper<S> {
    fn from(t: TreeEdge<S>) -> Self {
        let inner = Rc::new(Edge::_T(t));
        let mut hasher = DefaultHasher::new();
        inner.hash(&mut hasher);
        let inner_hash = hasher.finish();
        Self {
            inner,
            inner_hash
        }
    }
}

impl<S: Hash + Clone + PartialEq + Eq> From<LeafEdge<S>> for EdgeWrapper<S> {
    fn from(l: LeafEdge<S>) -> Self {
        let inner = Rc::new(Edge::_L(l));
        let mut hasher = DefaultHasher::new();
        inner.hash(&mut hasher);
        let inner_hash = hasher.finish();
        Self {
            inner,
            inner_hash
        }
    }
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Display for Edge<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Edge::_L(l) => f.write_str(&l.to_string()),
            Edge::_T(t) => f.write_str(&t.to_string()),
        }
    }
}

impl<S: Hash + Clone + PartialEq + Eq + Display> Display for EdgeWrapper<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}
