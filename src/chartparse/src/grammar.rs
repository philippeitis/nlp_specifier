use std::collections::{HashMap, HashSet};

use std::fmt::Debug;
use std::hash::Hash;
use std::rc::Rc;
use std::str::FromStr;

use fnv::FnvHashMap;
use unicode_width::UnicodeWidthChar;

use crate::production::{Production, Symbol};

#[derive(Clone)]
struct LeftCorners<N: Hash + PartialEq + Eq, T: Hash + PartialEq + Eq> {
    immediate_categories: HashMap<N, HashSet<N>>,
    immediate_words: HashMap<N, HashSet<T>>,
    left_corners: HashMap<N, HashSet<N>>,
    parents: HashMap<N, HashSet<N>>,
    left_corner_words: Option<bool>,
}

#[derive(Clone)]
pub struct ContextFreeGrammar<N: Hash + PartialEq + Eq, T: Hash + PartialEq + Eq> {
    pub start: N,
    pub productions: Vec<Rc<Production<N, T>>>,
    categories: HashSet<N>,
    // Grammar forms
    is_lexical: bool,
    is_nonlexical: bool,
    min_len: usize,
    max_len: usize,
    all_unary_are_lexical: bool,
    pub lhs_index: FnvHashMap<N, Vec<Rc<Production<N, T>>>>,
    rhs_index: FnvHashMap<Symbol<N, T>, Vec<Rc<Production<N, T>>>>,
    empty_index: FnvHashMap<N, Rc<Production<N, T>>>,
    lexical_index: FnvHashMap<T, HashSet<Rc<Production<N, T>>>>,
    leftcorner_relationships: Option<LeftCorners<N, T>>,
}

impl<N: Hash + PartialEq + Eq + Clone, T: Hash + PartialEq + Eq + Clone> ContextFreeGrammar<N, T> {
    fn new(start: N, productions: Vec<Production<N, T>>) -> Self {
        let productions = productions.into_iter().map(Rc::from).collect();
        Self::new_from_rc(start, productions)
    }

    fn new_from_rc(start: N, productions: Vec<Rc<Production<N, T>>>) -> Self {
        let mut lexical_index: FnvHashMap<T, HashSet<Rc<Production<N, T>>>> = FnvHashMap::default();
        let mut rhs_index = FnvHashMap::default();
        let mut lhs_index = FnvHashMap::default();
        let mut empty_index = FnvHashMap::default();

        for prod in productions.iter() {
            if !lhs_index.contains_key(&prod.lhs) {
                lhs_index.insert(prod.lhs.clone(), Vec::new());
            }
            lhs_index.get_mut(&prod.lhs).unwrap().push(prod.clone());

            if let Some(rhs) = prod.rhs.first() {
                if !rhs_index.contains_key(rhs) {
                    rhs_index.insert(rhs.clone(), vec![]);
                }
                rhs_index.get_mut(&rhs).unwrap().push(prod.clone());
            } else {
                empty_index.insert(prod.lhs.clone(), prod.clone());
            }
            for token in &prod.rhs {
                match &token {
                    Symbol::NonTerminal(_) => continue,
                    Symbol::Terminal(t) => {
                        if lexical_index.contains_key(t) {
                            lexical_index.get_mut(t).unwrap().insert(prod.clone());
                        } else {
                            let mut set = HashSet::new();
                            set.insert(prod.clone());
                            lexical_index.insert(t.clone(), set);
                        }
                    }
                }
            }
        }

        ContextFreeGrammar {
            start,
            lexical_index,
            lhs_index,
            rhs_index,
            empty_index,
            is_lexical: productions.iter().all(|p| p.is_lexical()),
            is_nonlexical: productions
                .iter()
                .filter(|p| p.len() != 1)
                .all(|p| p.is_nonlexical()),
            min_len: productions.iter().map(|p| p.len()).min().unwrap(),
            max_len: productions.iter().map(|p| p.len()).max().unwrap(),
            all_unary_are_lexical: productions
                .iter()
                .filter(|p| p.len() == 1)
                .all(|p| p.is_lexical()),
            categories: productions.iter().map(|p| p.lhs.clone()).collect(),
            productions,
            leftcorner_relationships: None,
        }
    }
}

impl<N: Hash + Clone + PartialEq + Eq, T: Hash + Clone + PartialEq + Eq> ContextFreeGrammar<N, T> {
    pub(crate) fn check_coverage(&self, tokens: &[T]) -> bool {
        tokens.iter().all(|t| self.lexical_index.contains_key(t))
    }

    fn is_lexical(&self) -> bool {
        self.is_lexical
    }

    fn is_nonlexical(&self) -> bool {
        self.is_nonlexical
    }

    fn min_len(&self) -> usize {
        self.min_len
    }

    fn max_len(&self) -> usize {
        self.max_len
    }

    fn len(&self) -> usize {
        self.productions.len()
    }

    pub(crate) fn start(&self) -> &N {
        &self.start
    }

    fn is_nonempty(&self) -> bool {
        self.min_len != 0
    }

    fn is_binarised(&self) -> bool {
        self.max_len <= 2
    }

    fn is_flexible_chomsky_normal_form(&self) -> bool {
        self.is_nonempty() && self.is_nonlexical && self.is_binarised()
    }

    fn is_chomsky_normal_form(&self) -> bool {
        self.is_flexible_chomsky_normal_form() && self.all_unary_are_lexical
    }

    pub(crate) fn productions(
        &self,
        lhs: Option<N>,
        rhs: Option<Symbol<N, T>>,
        empty: bool,
    ) -> Result<Vec<Rc<Production<N, T>>>, &'static str> {
        Ok(match (lhs, rhs, empty) {
            (None, None, false) => self.productions.clone(),
            (None, None, true) => self.empty_index.values().cloned().collect(),
            (None, Some(rhs), false) => self.rhs_index.get(&rhs).cloned().unwrap_or(vec![]),
            (None, Some(_), true) => {
                return Err("You cannot select empty and non-empty productions at the same time.");
            }
            (Some(lhs), None, false) => self.lhs_index.get(&lhs).cloned().unwrap_or(vec![]),
            (Some(lhs), None, true) => self
                .empty_index
                .get(&lhs)
                .cloned()
                .map(|x| vec![x])
                .unwrap_or(vec![]),
            (Some(lhs), Some(rhs), false) => {
                match (self.lhs_index.get(&lhs), self.rhs_index.get(&rhs)) {
                    (Some(lhs_ind), Some(rhs_ind)) => {
                        let mut productions = vec![];
                        for prod in lhs_ind {
                            if rhs_ind.contains(prod) {
                                productions.push(prod.clone());
                            }
                        }
                        productions
                    }
                    _ => vec![],
                }
            }
            (Some(_), Some(_), true) => {
                return Err("You cannot select empty and non-empty productions at the same time.");
            }
        })
    }
}

impl<
        N: Hash + PartialEq + Eq + Clone + ParseNonTerminal,
        T: Hash + PartialEq + Eq + Clone + ParseTerminal,
    > FromStr for ContextFreeGrammar<N, T>
{
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        read_grammar(&s, &standard_nonterm_parser)
    }
}

impl ContextFreeGrammar<String, String> {
    fn eliminate_start(&self) -> ContextFreeGrammar<String, String> {
        let start = Symbol::NonTerminal(self.start.clone());
        let new_start = "S0_SIGMA";
        if self.productions.iter().any(|p| p.rhs.contains(&start)) {
            let mut productions = self.productions.clone();
            productions.push(Rc::new(Production::new(new_start.to_string(), vec![start])));
            ContextFreeGrammar::new_from_rc(new_start.to_string(), productions)
        } else {
            self.clone()
        }
    }
}

pub fn standard_nonterm_parser<N: ParseNonTerminal>(
    s: &str,
    pos: usize,
) -> Result<(N, usize), &'static str> {
    let mut index = pos;
    // Capture
    let mut chars = s[pos..].chars().peekable();
    match chars.next() {
        Some(c) => match c {
            '/' => {
                index += c.width().unwrap();
            }
            x if x.is_alphanumeric() => {
                index += c.width().unwrap();
            }
            _ => {
                return Err("unexpected next char");
            }
        },
        _ => return Err("no next char"),
    }

    while let Some(c) = chars.peek() {
        match c {
            '_' | '^' | '<' | '>' | '-' => {
                index += c.width().unwrap();
            }
            x if x.is_alphanumeric() => {
                index += c.width().unwrap();
            }
            _ => break,
        }
        chars.next();
    }

    let nt = N::parse_nonterminal(&s[pos..index]).map_err(|_| "could not parse nonterminal")?;
    for c in chars {
        match c {
            x if x.is_whitespace() => {
                index += x.width().unwrap();
            }
            _ => break,
        }
    }

    Ok((nt, index))
}

fn read_arrow(line: &str, mut pos: usize) -> Option<usize> {
    let mut chars = line[pos..].chars().peekable();

    match (chars.next(), chars.next(), chars.next()) {
        (Some('-'), Some('>'), Some(' ')) => {
            pos += 3;
        }
        _ => return None,
    }

    while let Some(' ') = chars.next() {
        pos += 1;
    }

    Some(pos)
}

/// Returns the end of the string, plus the start of the next string
fn terminate_str(line: &str, mut pos: usize) -> Option<(usize, usize)> {
    let mut chars = line[pos..].chars().peekable();

    let q = chars.next().unwrap();

    if q != '"' && q != '\'' {
        return None;
    }

    pos += 1;

    let mut broke = false;
    while let Some(c) = chars.next() {
        if q != c {
            pos += c.width().unwrap();
        } else {
            pos += 1;
            broke = true;
            break;
        }
    }

    let end = pos;

    while let Some(c) = chars.next() {
        if c.is_whitespace() {
            pos += c.width().unwrap();
        } else {
            break;
        }
    }

    if !broke {
        None
    } else {
        Some((end, pos))
    }
}

fn eat_disjunction(line: &str, mut pos: usize) -> Option<usize> {
    let mut chars = line[pos..].chars();

    match (chars.next(), chars.next()) {
        (Some('|'), Some(' ')) => {
            pos += 2;
        }
        _ => return None,
    }

    while let Some(c) = chars.next() {
        if c.is_whitespace() {
            pos += c.width().unwrap();
        } else {
            break;
        }
    }

    Some(pos)
}

pub fn read_production<
    N: Clone + ParseNonTerminal,
    T: Clone + ParseTerminal,
    F: Fn(&str, usize) -> Result<(N, usize), &'static str>,
>(
    line: &str,
    nt_parser: &F,
) -> Result<Vec<Production<N, T>>, &'static str> {
    let (lhs, mut pos) = nt_parser(line, 0)?;

    pos = read_arrow(line, pos).ok_or("no arrow")?;

    let mut rhssides: Vec<Vec<_>> = vec![vec![]];

    while pos < line.len() {
        match line[pos..].chars().next().unwrap() {
            '\'' | '"' => {
                let (end, xpos) =
                    terminate_str(line, pos).ok_or("could not terminate production")?;
                rhssides.last_mut().unwrap().push(Symbol::Terminal(
                    T::parse_terminal(&line[pos + 1..end - 1]).unwrap(),
                ));
                pos = xpos;
            }
            '|' => {
                pos = eat_disjunction(line, pos).ok_or("no disjunction found")?;
                rhssides.push(vec![]);
            }
            _ => {
                let (rnt, xpos) = nt_parser(line, pos)?;
                rhssides.last_mut().unwrap().push(Symbol::NonTerminal(rnt));
                pos = xpos;
            }
        }
    }

    Ok(rhssides
        .into_iter()
        .map(|prod| Production::new(lhs.clone(), prod))
        .collect())
}

pub fn read_grammar<
    N: Hash + PartialEq + Eq + Clone + ParseNonTerminal,
    T: Hash + PartialEq + Eq + Clone + ParseTerminal,
    F: Fn(&str, usize) -> Result<(N, usize), &'static str>,
>(
    input: &str,
    nt_parser: F,
) -> Result<ContextFreeGrammar<N, T>, &'static str> {
    let mut productions = Vec::new();
    let mut continued_line = String::new();

    for line in input.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some((start, _)) = line.rsplit_once("\\") {
            continued_line += start.trim_end();
            continued_line += " ";
            continue;
        }
        if line.starts_with("%") {
            panic!("https://github.com/nltk/nltk/blob/e4444c9b15762e6d86f5f6c4f5faadb87632c72a/nltk/grammar.py#L1438");
        } else {
            continued_line += line;
            productions.extend(read_production(&continued_line, &nt_parser)?);
            continued_line.clear();
        }
    }

    if productions.is_empty() {
        return Err("empty productions");
    }

    Ok(ContextFreeGrammar::new(
        productions[0].lhs.clone(),
        productions,
    ))
}

pub trait ParseTerminal: Sized {
    type Error: Debug;

    fn parse_terminal(s: &str) -> Result<Self, Self::Error>;
}

pub trait ParseNonTerminal: Sized {
    type Error: Debug;

    fn parse_nonterminal(s: &str) -> Result<Self, Self::Error>;
}

impl ParseTerminal for String {
    type Error = ();

    fn parse_terminal(s: &str) -> Result<Self, Self::Error> {
        Ok(s.to_string())
    }
}

impl ParseNonTerminal for String {
    type Error = ();

    fn parse_nonterminal(s: &str) -> Result<Self, Self::Error> {
        Ok(s.to_string())
    }
}
