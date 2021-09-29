use std::hash::Hash;

use crate::chart::Chart;
use crate::edge::{EdgeI, EdgeWrapper, LeafEdge, TreeEdge};
use crate::grammar::ContextFreeGrammar;
use crate::select::Restrictions;

pub trait ChartRule<S: Hash + Clone + PartialEq + Eq> {
    fn num_edges(&self) -> usize;

    fn apply(
        &self,
        chart: &mut Chart<S>,
        grammar: &ContextFreeGrammar<S>,
        edges: Vec<EdgeWrapper<S>>,
    ) -> Vec<EdgeWrapper<S>>;

    fn apply_everywhere(
        &self,
        chart: &mut Chart<S>,
        grammar: &ContextFreeGrammar<S>,
    ) -> Vec<EdgeWrapper<S>> {
        match self.num_edges() {
            0 => self.apply(chart, grammar, vec![]),
            1 => {
                let edges = (*chart.edges()).clone();
                let mut res = vec![];
                for e1 in &edges {
                    res.extend(self.apply(chart, grammar, vec![e1.clone()]));
                }
                res
            }
            2 => {
                let mut res = vec![];
                let edges = (*chart.edges()).clone();
                for e1 in &edges {
                    for e2 in &edges {
                        res.extend(self.apply(chart, grammar, vec![e1.clone(), e2.clone()]));
                    }
                }
                res
            }
            3 => {
                let mut res = vec![];
                let edges = (*chart.edges()).clone();
                for e1 in &edges {
                    for e2 in &edges {
                        for e3 in &edges {
                            res.extend(self.apply(
                                chart,
                                grammar,
                                vec![e1.clone(), e2.clone(), e3.clone()],
                            ));
                        }
                    }
                }
                res
            }
            _ => unimplemented!(""),
        }
    }

    fn box_clone(&self) -> Box<dyn ChartRule<S>>;
}

pub struct LeafInitRule {}

impl<S: Hash + Clone + PartialEq + Eq> ChartRule<S> for LeafInitRule {
    fn num_edges(&self) -> usize {
        0
    }

    fn apply(
        &self,
        chart: &mut Chart<S>,
        _grammar: &ContextFreeGrammar<S>,
        _edges: Vec<EdgeWrapper<S>>,
    ) -> Vec<EdgeWrapper<S>> {
        let mut edges = vec![];
        for (i, leaf) in chart.leaves().into_iter().enumerate() {
            let new_edge: EdgeWrapper<_> = LeafEdge::new(leaf, i).into();

            if chart.insert(new_edge.clone(), vec![]) {
                edges.push(new_edge);
            }
        }
        edges
    }

    fn box_clone(&self) -> Box<dyn ChartRule<S>> {
        Box::new(Self {})
    }
}

#[derive(Copy, Clone)]
pub struct FundamentalRule {}

impl<S: Hash + Clone + PartialEq + Eq> ChartRule<S> for FundamentalRule {
    fn num_edges(&self) -> usize {
        2
    }

    fn apply(
        &self,
        chart: &mut Chart<S>,
        _grammar: &ContextFreeGrammar<S>,
        mut edges: Vec<EdgeWrapper<S>>,
    ) -> Vec<EdgeWrapper<S>> {
        let lhs = edges.remove(0);
        let rhs = edges.remove(0);
        if !(!lhs.is_complete()
            && rhs.is_complete()
            && lhs.end() == rhs.start()
            && lhs.next_sym() == Some(rhs.lhs()))
        {
            return vec![];
        }

        let new_edge = lhs.move_dot_forward(rhs.end()).unwrap();

        if chart.insert_with_backpointer(new_edge.clone(), &lhs, &rhs) {
            vec![new_edge]
        } else {
            vec![]
        }
    }

    fn box_clone(&self) -> Box<dyn ChartRule<S>> {
        Box::new(Self {})
    }
}

pub struct SingleEdgeFundamentalRule {}

impl<S: Hash + Clone + PartialEq + Eq> ChartRule<S> for SingleEdgeFundamentalRule {
    fn num_edges(&self) -> usize {
        1
    }

    fn apply(
        &self,
        chart: &mut Chart<S>,
        _grammar: &ContextFreeGrammar<S>,
        mut edges: Vec<EdgeWrapper<S>>,
    ) -> Vec<EdgeWrapper<S>> {
        let edge = edges.remove(0);
        let mut edges = vec![];

        if !edge.is_complete() {
            let restrictions = Restrictions {
                start: Some(edge.end()),
                is_complete: Some(true),
                lhs: edge.next_sym().cloned(),
                ..Default::default()
            };
            let right_edges = match chart.select(restrictions) {
                None => return edges,
                Some(edges) => edges,
            };
            for right_edge in right_edges.as_ref() {
                let new_edge = edge.move_dot_forward(right_edge.end()).unwrap();
                if chart.insert_with_backpointer(new_edge.clone(), &edge, right_edge) {
                    edges.push(new_edge);
                }
            }
        } else {
            let mut restrictions = Restrictions::default();
            restrictions.end = Some(edge.start());
            restrictions.is_complete = Some(false);
            restrictions.next_sym = Some(edge.lhs().clone());
            let left_edges = match chart.select(restrictions) {
                None => return edges,
                Some(edges) => edges,
            };
            for left_edge in left_edges.as_ref() {
                let new_edge = left_edge.move_dot_forward(edge.end()).unwrap();
                if chart.insert_with_backpointer(new_edge.clone(), left_edge, &edge) {
                    edges.push(new_edge);
                }
            }
        }

        edges
    }

    fn box_clone(&self) -> Box<dyn ChartRule<S>> {
        Box::new(Self {})
    }
}

pub struct EmptyPredictRule {}

impl<S: Hash + Clone + PartialEq + Eq> ChartRule<S> for EmptyPredictRule {
    fn num_edges(&self) -> usize {
        0
    }

    fn apply(
        &self,
        chart: &mut Chart<S>,
        grammar: &ContextFreeGrammar<S>,
        _edges: Vec<EdgeWrapper<S>>,
    ) -> Vec<EdgeWrapper<S>> {
        let mut edges = vec![];
        for production in grammar.productions(None, None, true).unwrap() {
            for i in 0..chart.num_leaves() + 1 {
                let new_edge: EdgeWrapper<_> =
                    TreeEdge::from_production(production.as_ref().clone(), i).into();

                if chart.insert(new_edge.clone(), vec![]) {
                    edges.push(new_edge);
                }
            }
        }
        edges
    }

    fn box_clone(&self) -> Box<dyn ChartRule<S>> {
        Box::new(Self {})
    }
}

pub struct BottomUpPredictCombineRule {}

impl<S: Hash + Clone + PartialEq + Eq> ChartRule<S> for BottomUpPredictCombineRule {
    fn num_edges(&self) -> usize {
        1
    }

    fn apply(
        &self,
        chart: &mut Chart<S>,
        grammar: &ContextFreeGrammar<S>,
        mut edges: Vec<EdgeWrapper<S>>,
    ) -> Vec<EdgeWrapper<S>> {
        let edge = edges.remove(0);
        if !edge.is_complete() {
            return vec![];
        }

        let mut edges = vec![];
        for production in grammar
            .productions(None, Some(edge.lhs().clone()), false)
            .unwrap()
        {
            let new_edge: EdgeWrapper<_> = TreeEdge::with_dot(
                edge.inner.span(),
                production.lhs().clone().to_symbol(),
                production.rhs().into(),
                1,
            )
            .into();
            if chart.insert(new_edge.clone(), vec![vec![edge.clone()]]) {
                edges.push(new_edge)
            }
        }
        edges
    }

    fn box_clone(&self) -> Box<dyn ChartRule<S>> {
        Box::new(Self {})
    }
}
