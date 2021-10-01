use std::collections::HashMap;
use std::ops::AddAssign;

use crate::parse_tree::tree::TerminalSymbol;
use crate::parse_tree::Symbol;

use itertools::Itertools;

fn count_subsequences(
    mut sentence: String,
    terminals: &[Symbol],
    counts: &mut HashMap<Vec<Symbol>, usize>,
    samples: &mut HashMap<Vec<Symbol>, String>,
) {
    for window_size in 1..terminals.len() {
        for window in terminals.windows(window_size) {
            if counts.contains_key(window) {
                counts.get_mut(window).unwrap().add_assign(1);
                if samples.get(window).unwrap().is_empty() {
                    std::mem::swap(samples.get_mut(window).unwrap(), &mut sentence);
                }
            } else {
                samples.insert(window.to_vec(), std::mem::take(&mut sentence));
                counts.insert(window.to_vec(), 1);
            }
        }
    }
}

pub fn count_subsequences_from_tokens(tokens: &[&[(String, String, String)]]) {
    let mut counts = HashMap::new();
    let mut samples = HashMap::new();
    for token_vec in tokens {
        let terminals: Vec<_> = token_vec
            .iter()
            .map(|(tag, _, _)| tag)
            .map(TerminalSymbol::from_terminal)
            .map(Result::unwrap)
            .map(Symbol::from)
            .collect();
        let sentence = token_vec.iter().map(|(_, word, _)| word).join(" ");

        count_subsequences(sentence, &terminals, &mut counts, &mut samples);
    }

    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort_by(|(_, count1), (_, count2)| count1.cmp(count2));
    counts.reverse();

    for (syms, count) in counts
        .iter()
        .filter(|(syms, _count)| syms.len() > 1)
        .take(50)
    {
        println!(
            "{}: {}",
            count,
            syms.iter().map(|x| x.to_string()).join(" ")
        );
        println!("{}", samples.get(syms).unwrap());
    }
}
