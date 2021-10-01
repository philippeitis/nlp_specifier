use std::collections::HashMap;
use std::ops::AddAssign;

use itertools::Itertools;

use crate::parse_tree::Symbol;
use crate::sentence::Sentence;

fn count_subsequences(
    sentence: String,
    terminals: &[Symbol],
    counts: &mut HashMap<Vec<Symbol>, usize>,
    samples: &mut HashMap<Vec<Symbol>, Vec<String>>,
) {
    for window_size in 1..terminals.len() {
        for window in terminals.windows(window_size) {
            if counts.contains_key(window) {
                counts.get_mut(window).unwrap().add_assign(1);
                samples.get_mut(window).unwrap().push(sentence.clone());
            } else {
                samples.insert(window.to_vec(), vec![sentence.clone()]);
                counts.insert(window.to_vec(), 1);
            }
        }
    }
}

pub fn count_subsequences_from_tokens(tokens: &[&Sentence]) {
    let mut counts = HashMap::new();
    let mut samples = HashMap::new();
    for sent in tokens {
        let terminals: Vec<_> = sent
            .tokens
            .iter()
            .map(|token| token.tag.clone())
            .map(Symbol::from)
            .collect();
        count_subsequences(sent.text.clone(), &terminals, &mut counts, &mut samples);
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
        println!("{}", samples.get(syms).unwrap().iter().take(5).join("\n"));
    }
}
