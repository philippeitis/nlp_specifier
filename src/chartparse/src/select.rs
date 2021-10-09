use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use crate::edge::{Edge, EdgeI};
use crate::production::Symbol;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Restrictions<T, N> {
    pub(crate) start: Option<usize>,
    pub(crate) end: Option<usize>,
    pub(crate) length: Option<usize>,
    pub(crate) lhs: Option<Symbol<T, N>>,
    pub(crate) rhs: Option<Vec<Symbol<T, N>>>,
    pub(crate) next_sym: Option<Symbol<T, N>>,
    pub(crate) dot: Option<usize>,
    pub(crate) is_complete: Option<bool>,
}

impl<T, N> Default for Restrictions<T, N> {
    fn default() -> Self {
        Self {
            start: None,
            end: None,
            length: None,
            lhs: None,
            rhs: None,
            next_sym: None,
            dot: None,
            is_complete: None,
        }
    }
}

#[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
pub struct RestrictionKeys(u32);

#[repr(u32)]
enum RestrictionKey {
    Start = 1 << 0,
    End = 1 << 1,
    Length = 1 << 2,
    Lhs = 1 << 3,
    Rhs = 1 << 4,
    NextSym = 1 << 5,
    Dot = 1 << 6,
    Completion = 1 << 7,
}

impl RestrictionKeys {
    fn overlaps(&self, rk: RestrictionKey) -> bool {
        self.0 & rk as u32 != 0
    }
}

impl RestrictionKeys {
    pub(crate) fn read_edge<T: Clone, N: Clone>(&self, edge: &Edge<T, N>) -> Restrictions<T, N> {
        use RestrictionKey::*;

        Restrictions {
            start: if self.overlaps(Start) {
                Some(edge.start())
            } else {
                None
            },
            end: if self.overlaps(End) {
                Some(edge.end())
            } else {
                None
            },
            length: if self.overlaps(Length) {
                Some(edge.length())
            } else {
                None
            },
            lhs: if self.overlaps(Lhs) {
                Some(edge.lhs().clone())
            } else {
                None
            },
            rhs: if self.overlaps(Rhs) {
                Some(edge.rhs().to_vec())
            } else {
                None
            },
            next_sym: if self.overlaps(NextSym) {
                edge.next_sym().cloned()
            } else {
                None
            },
            dot: if self.overlaps(Dot) {
                Some(edge.dot())
            } else {
                None
            },
            is_complete: if self.overlaps(Completion) {
                Some(edge.is_complete())
            } else {
                None
            },
        }
    }
}

impl<T, N> Restrictions<T, N> {
    pub(crate) fn is_empty(&self) -> bool {
        self.start.is_none()
            && self.end.is_none()
            && self.length.is_none()
            && self.lhs.is_none()
            && self.rhs.is_none()
            && self.next_sym.is_none()
            && self.dot.is_none()
            && self.is_complete.is_none()
    }

    pub(crate) fn keys(&self) -> RestrictionKeys {
        use RestrictionKey::*;
        let mut key = 0;

        if self.start.is_some() {
            key |= Start as u32;
        }
        if self.end.is_some() {
            key |= End as u32;
        }
        if self.length.is_some() {
            key |= Length as u32;
        }
        if self.lhs.is_some() {
            key |= Lhs as u32;
        }
        if self.rhs.is_some() {
            key |= Rhs as u32;
        }
        if self.next_sym.is_some() {
            key |= NextSym as u32;
        }
        if self.dot.is_some() {
            key |= Dot as u32;
        }
        if self.is_complete.is_some() {
            key |= Completion as u32;
        }

        RestrictionKeys(key)
    }

    pub fn start(self, start: usize) -> Self {
        Restrictions {
            start: Some(start),
            ..self
        }
    }

    pub fn end(self, end: usize) -> Self {
        Restrictions {
            end: Some(end),
            ..self
        }
    }

    pub fn lhs(self, lhs: Symbol<T, N>) -> Self {
        Restrictions {
            lhs: Some(lhs),
            ..self
        }
    }
}

impl<T: Clone, N: Clone> Restrictions<T, N> {
    pub fn read_edge(&self, edge: &Edge<T, N>) -> Self {
        Restrictions {
            start: self.start.map(|_| edge.start()),
            end: self.end.map(|_| edge.end()),
            length: self.length.map(|_| edge.length()),
            lhs: self.lhs.as_ref().map(|_| edge.lhs().clone()),
            rhs: self.rhs.as_ref().map(|_| edge.rhs().to_vec()),
            next_sym: self
                .next_sym
                .as_ref()
                .map(|_| edge.next_sym().cloned())
                .flatten(),
            dot: self.dot.map(|_| edge.dot()),
            is_complete: self.is_complete.map(|_| edge.is_complete()),
        }
    }
}
