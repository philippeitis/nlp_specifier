use std::hash::Hash;

pub mod chart;
pub mod chartrule;
pub mod edge;
pub mod grammar;
pub mod production;
pub mod select;
pub mod tree;

use crate::chart::Chart;
use crate::chartrule::{
    BottomUpPredictCombineRule, ChartRule, EmptyPredictRule, LeafInitRule,
    SingleEdgeFundamentalRule,
};
pub use crate::grammar::ContextFreeGrammar;
pub use crate::production::SymbolWrapper;
pub use crate::tree::TreeWrapper;

pub struct ChartParser<'a, S: Hash + Clone + PartialEq + Eq> {
    grammar: &'a ContextFreeGrammar<S>,
    use_agenda: bool,
    strategy: Vec<Box<dyn ChartRule<S>>>,
    axioms: Vec<Box<dyn ChartRule<S>>>,
    inference_rules: Vec<Box<dyn ChartRule<S>>>,
}

impl<'a, S: Hash + Clone + PartialEq + Eq> ChartParser<'a, S> {
    pub fn from_grammar(grammar: &'a ContextFreeGrammar<S>) -> Self {
        ChartParser::from_grammar_with_strategy(
            grammar,
            vec![
                Box::new(LeafInitRule {}),
                Box::new(EmptyPredictRule {}),
                Box::new(BottomUpPredictCombineRule {}),
                Box::new(SingleEdgeFundamentalRule {}),
            ],
            true,
        )
    }

    pub fn from_grammar_with_strategy(
        grammar: &'a ContextFreeGrammar<S>,
        strategy: Vec<Box<dyn ChartRule<S>>>,
        use_agenda: bool,
    ) -> Self {
        Self {
            grammar,
            axioms: strategy
                .iter()
                .cloned()
                .filter(|cr| cr.num_edges() == 0)
                .collect(),
            inference_rules: strategy
                .iter()
                .cloned()
                .filter(|cr| cr.num_edges() == 1)
                .collect(),
            use_agenda: strategy.iter().all(|cr| cr.num_edges() < 2) && use_agenda,
            strategy,
        }
    }

    pub fn chart_parse(&self, tokens: &[S]) -> Result<Chart<S>, &'static str> {
        if !self.grammar.check_coverage(tokens) {
            return Err("Bad coverage");
        }
        let mut chart = Chart::new(
            tokens
                .iter()
                .cloned()
                .map(SymbolWrapper::terminal)
                .collect(),
        )
        .unwrap();

        if !self.use_agenda {
            unimplemented!();
        } else {
            loop {
                let mut edges_added = false;
                for rule in &self.strategy {
                    let new_edges = rule.apply_everywhere(&mut chart, self.grammar);
                    edges_added |= !new_edges.is_empty();
                }

                if !edges_added {
                    break;
                }
            }
        }
        Ok(chart)
    }

    pub fn parse(&self, tokens: &[S]) -> Result<Vec<TreeWrapper<S>>, &'static str> {
        let chart = self.chart_parse(tokens)?;
        Ok(chart.parses(self.grammar.start()))
    }
}

impl<S: Hash + Clone + PartialEq + Eq> Clone for Box<dyn ChartRule<S>> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

pub fn main(tokens: Vec<Vec<String>>, grammar: String) -> Vec<Vec<TreeWrapper<String>>> {
    use std::time::Instant;
    let start = Instant::now();
    let grammar = ContextFreeGrammar::fromstring(grammar).unwrap();
    println!(
        "{} CFG::fromstring Rust",
        (Instant::now() - start).as_micros()
    );
    let chart_parser = ChartParser::from_grammar(&grammar);
    let mut results = Vec::new();

    for string in tokens {
        let start = Instant::now();
        results.push(chart_parser.parse(&string).unwrap_or_default());
        let elapsed = (Instant::now() - start).as_micros();
        println!("{} ChartParser::parse({:?}) Rust", elapsed.max(1), string);
    }

    results
}
