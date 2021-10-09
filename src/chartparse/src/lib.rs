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
use crate::production::Symbol;
pub use crate::tree::Tree;

pub struct ChartParser<'a, N: Hash + Clone + PartialEq + Eq, T: Hash + Clone + PartialEq + Eq> {
    grammar: &'a ContextFreeGrammar<N, T>,
    use_agenda: bool,
    strategy: Vec<Box<dyn ChartRule<N, T>>>,
    axioms: Vec<Box<dyn ChartRule<N, T>>>,
    inference_rules: Vec<Box<dyn ChartRule<N, T>>>,
}

impl<'a, N: Hash + Clone + PartialEq + Eq, T: Hash + Clone + PartialEq + Eq> ChartParser<'a, N, T> {
    pub fn from_grammar(grammar: &'a ContextFreeGrammar<N, T>) -> Self {
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
        grammar: &'a ContextFreeGrammar<N, T>,
        strategy: Vec<Box<dyn ChartRule<N, T>>>,
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

    pub fn chart_parse<'b>(&self, tokens: &'b [T]) -> Result<Chart<'b, N, T>, &'static str> {
        if !self.grammar.check_coverage(tokens) {
            return Err("Bad coverage");
        }

        let mut chart = Chart::new(tokens);

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

    pub fn parse(&self, tokens: &[T]) -> Result<Vec<Tree<N, T>>, &'static str> {
        let chart = self.chart_parse(tokens)?;
        Ok(chart.parses(Symbol::NonTerminal(self.grammar.start().clone())))
    }
}

impl<N: Hash + Clone + PartialEq + Eq, T: Hash + Clone + PartialEq + Eq> Clone
    for Box<dyn ChartRule<N, T>>
{
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

pub fn main<S: AsRef<String>>(
    tokens: Vec<Vec<String>>,
    grammar: S,
) -> Vec<Vec<Tree<String, String>>> {
    use std::str::FromStr;
    use std::time::Instant;

    let start = Instant::now();
    let grammar = ContextFreeGrammar::from_str(grammar.as_ref()).unwrap();
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
