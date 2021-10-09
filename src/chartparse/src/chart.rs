use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::rc::Rc;

use fnv::FnvHashMap;
use indexmap::IndexSet;

use crate::edge::EdgeWrapper;
use crate::production::Symbol;
use crate::select::{RestrictionKeys, Restrictions};
use crate::tree::Tree;

pub struct Chart<'a, T: Hash + PartialEq + Eq, N: Hash + PartialEq + Eq> {
    tokens: &'a [T],
    edges: Rc<Vec<EdgeWrapper<T, N>>>,
    edge_to_cpl: FnvHashMap<EdgeWrapper<T, N>, IndexSet<Rc<Vec<EdgeWrapper<T, N>>>>>,
    indexes: RefCell<
        FnvHashMap<RestrictionKeys, FnvHashMap<Restrictions<T, N>, Rc<Vec<EdgeWrapper<T, N>>>>>,
    >,
}

impl<'a, T: Clone + Hash + PartialEq + Eq, N: Clone + Hash + PartialEq + Eq> Chart<'a, T, N> {
    pub(crate) fn new(tokens: &'a [T]) -> Self {
        Chart {
            tokens,
            edges: Default::default(),
            edge_to_cpl: Default::default(),
            indexes: Default::default(),
        }
    }

    pub(crate) fn num_leaves(&self) -> usize {
        self.tokens.len()
    }

    fn leaf(&self, index: usize) -> &T {
        &self.tokens[index]
    }

    pub(crate) fn leaves(&self) -> &[T] {
        &self.tokens
    }

    pub(crate) fn edges(&self) -> Rc<Vec<EdgeWrapper<T, N>>> {
        self.edges.clone()
    }

    fn num_edges(&self) -> usize {
        self.edge_to_cpl.len()
    }

    pub(crate) fn select(
        &self,
        restriction: Restrictions<T, N>,
    ) -> Option<Rc<Vec<EdgeWrapper<T, N>>>> {
        if restriction.is_empty() {
            return Some(self.edges.clone());
        }

        let keys = restriction.keys();
        if !self.indexes.borrow().contains_key(&keys) {
            self.add_index(keys);
        }

        self.indexes
            .borrow()
            .get(&keys)
            .unwrap()
            .get(&restriction)
            .cloned()
    }

    fn add_index(&self, keys: RestrictionKeys) {
        let mut indexes = self.indexes.borrow_mut();
        let index = indexes.entry(keys).or_default();

        for edge in self.edges.iter() {
            let index_key = keys.read_edge(&edge.inner);
            if !index.contains_key(&index_key) {
                index.insert(index_key, Rc::new(vec![edge.clone()]));
            } else {
                let edges = index.get_mut(&index_key).unwrap();
                Rc::make_mut(edges).push(edge.clone());
            }
        }
    }

    fn register_with_indexes(&self, edge: EdgeWrapper<T, N>) {
        let mut indexes = self.indexes.borrow_mut();
        for (keys, values) in indexes.iter_mut() {
            let index_key = keys.read_edge(&edge.inner);
            if !values.contains_key(&index_key) {
                values.insert(index_key, Rc::new(vec![edge.clone()]));
            } else {
                let edges = values.get_mut(&index_key).unwrap();
                Rc::make_mut(edges).push(edge.clone());
            }
        }
    }

    fn append_edge(&mut self, edge: EdgeWrapper<T, N>) {
        Rc::make_mut(&mut self.edges).push(edge)
    }

    fn child_pointer_lists(&self, edge: &EdgeWrapper<T, N>) -> Vec<Rc<Vec<EdgeWrapper<T, N>>>> {
        self.edge_to_cpl
            .get(edge)
            .map(|x| x.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub(crate) fn insert_with_backpointer(
        &mut self,
        new_edge: EdgeWrapper<T, N>,
        previous_edge: &EdgeWrapper<T, N>,
        child_edge: &EdgeWrapper<T, N>,
    ) -> bool {
        let mut cpls = self.child_pointer_lists(previous_edge);
        cpls.iter_mut()
            .for_each(|v| Rc::make_mut(v).push(child_edge.clone()));
        self.insert_rc(new_edge, cpls)
    }

    pub(crate) fn insert(
        &mut self,
        edge: EdgeWrapper<T, N>,
        new_cpls: Vec<Vec<EdgeWrapper<T, N>>>,
    ) -> bool {
        if !self.edge_to_cpl.contains_key(&edge) {
            self.append_edge(edge.clone());
            self.register_with_indexes(edge.clone());
        }

        let cpls = self.edge_to_cpl.entry(edge).or_default();
        let mut chart_was_modified = false;
        for cpl in new_cpls {
            if !cpls.contains(&cpl) {
                cpls.insert(Rc::new(cpl));
                chart_was_modified = true;
            }
        }

        chart_was_modified
    }

    pub(crate) fn insert_rc(
        &mut self,
        edge: EdgeWrapper<T, N>,
        new_cpls: Vec<Rc<Vec<EdgeWrapper<T, N>>>>,
    ) -> bool {
        if !self.edge_to_cpl.contains_key(&edge) {
            self.append_edge(edge.clone());
            self.register_with_indexes(edge.clone());
        }

        let cpls = self.edge_to_cpl.entry(edge).or_default();
        let mut chart_was_modified = false;
        for cpl in new_cpls {
            if !cpls.contains(&cpl) {
                cpls.insert(cpl);
                chart_was_modified = true;
            }
        }

        chart_was_modified
    }

    pub(crate) fn parses(&self, root: Symbol<T, N>) -> Vec<Tree<T, N>> {
        let edges = match self.select(
            Restrictions::default()
                .start(0)
                .end(self.num_leaves())
                .lhs(root),
        ) {
            None => return vec![],
            Some(e) => e,
        };
        edges
            .iter()
            .map(|edge| self.trees(edge, true))
            .flatten()
            .collect()
    }

    /// Lazy version of parse.
    pub(crate) fn parse_iter<'b>(
        &'b self,
        root: Symbol<T, N>,
    ) -> Box<dyn Iterator<Item = Tree<T, N>> + 'b> {
        let edges = match self.select(
            Restrictions::default()
                .start(0)
                .end(self.num_leaves())
                .lhs(root),
        ) {
            None => return Box::new(std::iter::empty()),
            Some(e) => e,
        };
        let mut trees = match edges.first() {
            Some(edge) => self.trees(edge, true),
            None => return Box::new(std::iter::empty()),
        }
        .into_iter();
        let mut index = 1;
        Box::new(std::iter::from_fn(move || loop {
            match trees.next() {
                Some(tree) => return Some(tree),
                None => {
                    trees = self.trees(edges.get(index)?, true).into_iter();
                    index += 1;
                }
            }
        }))
    }

    pub(crate) fn trees(&self, edge: &EdgeWrapper<T, N>, complete: bool) -> Vec<Tree<T, N>> {
        self.tree_helper(&edge, complete, &mut HashMap::new())
    }
}

impl<
        'a,
        T: Clone + Hash + PartialEq + Eq + Display + Debug,
        N: Clone + Hash + PartialEq + Eq + Display + Debug,
    > Chart<'a, T, N>
{
    pub(crate) fn pretty_format_edge(
        &self,
        edge: EdgeWrapper<T, N>,
        width: Option<usize>,
    ) -> String {
        let width = match width {
            None => 50 / (self.num_leaves() + 1),
            Some(x) => x,
        };

        let (start, end) = (edge.start(), edge.end());

        let mut output = format!("|{}", format!(".{}", " ".repeat(width - 1)).repeat(start));
        if start == end {
            if edge.is_complete() {
                output.push('#');
            } else {
                output.push('>');
            }
        } else if edge.is_complete() && start == 0 && end == self.num_leaves() {
            output.push_str(&format!(
                "[{}]",
                "=".repeat(width * (end - start - 1) + (width - 1))
            ));
        } else if edge.is_complete() {
            output.push_str(&format!(
                "[{}]",
                "-".repeat(width * (end - start - 1) + (width - 1))
            ));
        } else {
            output.push_str(&format!(
                "[{}>",
                "-".repeat(width * (end - start - 1) + (width - 1))
            ));
        }
        output += &format!("{}.", " ".repeat(width - 1)).repeat(self.num_leaves() - end);
        output.push_str("| ");
        output.push_str(&edge.to_string());

        output
    }
}

pub fn partial_cartesian<T: Clone>(a: Vec<Vec<T>>, b: &[T]) -> Vec<Vec<T>> {
    a.into_iter()
        .flat_map(|xs| {
            b.iter()
                .cloned()
                .map(|y| {
                    let mut vec = xs.clone();
                    vec.push(y);
                    vec
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn cartesian_product<T: Clone>(lists: &[&[T]]) -> Vec<Vec<T>> {
    match lists.split_first() {
        Some((first, rest)) => {
            let init: Vec<Vec<T>> = first.iter().cloned().map(|n| vec![n]).collect();

            rest.iter()
                .cloned()
                .fold(init, |vec, list| partial_cartesian(vec, list))
        }
        None => {
            vec![]
        }
    }
}

impl<'a, T: Hash + Clone + PartialEq + Eq, N: Hash + Clone + PartialEq + Eq> Chart<'a, T, N> {
    pub(crate) fn tree_helper(
        &self,
        edge: &EdgeWrapper<T, N>,
        complete: bool,
        memo: &mut HashMap<EdgeWrapper<T, N>, Vec<Tree<T, N>>>,
    ) -> Vec<Tree<T, N>> {
        if let Some(trees) = memo.get(edge) {
            return trees.clone();
        }

        if complete && !edge.is_complete() {
            return Vec::new();
        }

        if edge.is_leafedge() {
            let leaf = vec![Tree::from_terminal(self.leaf(edge.start()).clone())];

            memo.insert(edge.clone(), leaf.clone());

            return leaf;
        }

        memo.insert(edge.clone(), vec![]);
        let mut trees = Vec::new();
        let lhs = edge.lhs().nonterminal().expect("If edge is LeafEdge, early return occurs. TreeEdge will never produce a terminal symbol as the LHS");
        for cpl in self.child_pointer_lists(edge) {
            let child_choices: Vec<_> = cpl
                .iter()
                .map(|cp| self.tree_helper(cp, complete, memo))
                .collect();
            let child_refs: Vec<_> = child_choices.iter().map(|x| x.as_slice()).collect();
            for children in cartesian_product(&child_refs) {
                // should be lazy
                trees.push(Tree::from_nonterminal(lhs.clone(), children));
            }
        }

        if !edge.is_complete() {
            let unexpanded: Vec<_> = edge
                .rhs()
                .iter()
                .skip(edge.dot())
                .cloned()
                .map(Tree::from)
                .collect();
            for tree in &mut trees {
                tree.extend(unexpanded.clone());
            }
        }

        memo.insert(edge.clone(), trees.clone());
        trees
    }
}
