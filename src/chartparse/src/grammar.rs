use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::rc::Rc;

use fnv::FnvHashMap;
use unicode_width::UnicodeWidthChar;

use crate::production::{NonTerminal, Production, Symbol, SymbolWrapper, Terminal};

#[derive(Clone)]
struct LeftCorners<S: Hash + Clone + PartialEq + Eq> {
    immediate_categories: HashMap<NonTerminal<S>, HashSet<NonTerminal<S>>>,
    immediate_words: HashMap<NonTerminal<S>, HashSet<Terminal<S>>>,
    left_corners: HashMap<NonTerminal<S>, HashSet<NonTerminal<S>>>,
    parents: HashMap<NonTerminal<S>, HashSet<NonTerminal<S>>>,
    left_corner_words: Option<bool>,
}

#[derive(Clone)]
pub struct ContextFreeGrammar<S: Hash + Clone + PartialEq + Eq> {
    pub start: NonTerminal<S>,
    pub productions: Vec<Rc<Production<S>>>,
    categories: HashSet<NonTerminal<S>>,
    // Grammar forms
    is_lexical: bool,
    is_nonlexical: bool,
    min_len: usize,
    max_len: usize,
    all_unary_are_lexical: bool,
    pub lhs_index: FnvHashMap<NonTerminal<S>, Vec<Rc<Production<S>>>>,
    rhs_index: FnvHashMap<SymbolWrapper<S>, Vec<Rc<Production<S>>>>,
    empty_index: FnvHashMap<NonTerminal<S>, Rc<Production<S>>>,
    lexical_index: FnvHashMap<Terminal<S>, HashSet<Rc<Production<S>>>>,
    leftcorner_relationships: Option<LeftCorners<S>>,
}

impl<S: Hash + Clone + PartialEq + Eq> ContextFreeGrammar<S> {
    fn new(start: NonTerminal<S>, productions: Vec<Production<S>>) -> Self {
        let productions = productions.into_iter().map(Rc::from).collect();
        Self::new_from_rc(start, productions)
    }

    fn new_from_rc(start: NonTerminal<S>, productions: Vec<Rc<Production<S>>>) -> Self {
        let mut lexical_index: FnvHashMap<Terminal<S>, HashSet<Rc<Production<S>>>> =
            FnvHashMap::default();
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
                match &token.inner {
                    Symbol::_NT(_) => continue,
                    Symbol::_T(t) => {
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

impl<S: Hash + Clone + PartialEq + Eq> ContextFreeGrammar<S> {
    pub(crate) fn check_coverage(&self, tokens: &[S]) -> bool {
        tokens.iter().all(|t| {
            self.lexical_index.contains_key(&Terminal {
                symbol: t.to_owned(),
            })
        })
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

    pub(crate) fn start(&self) -> SymbolWrapper<S> {
        self.start.clone().to_symbol()
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
        lhs: Option<NonTerminal<S>>,
        rhs: Option<SymbolWrapper<S>>,
        empty: bool,
    ) -> Result<Vec<Rc<Production<S>>>, &'static str> {
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

impl<S: Hash + Clone + PartialEq + Eq + ParseSymbol> ContextFreeGrammar<S> {
    pub fn fromstring(s: String) -> Result<Self, &'static str> {
        read_grammar(&s, &standard_nonterm_parser)
    }
}

impl ContextFreeGrammar<String> {
    fn eliminate_start(&self) -> ContextFreeGrammar<String> {
        let start = self.start.clone().to_symbol();

        if self.productions.iter().any(|p| p.rhs.contains(&start)) {
            let new_start = NonTerminal {
                symbol: "S0_SIGMA".to_string(),
            };
            let mut productions = self.productions.clone();
            productions.push(Rc::new(Production::new(new_start.clone(), vec![start])));
            ContextFreeGrammar::new_from_rc(new_start, productions)
        } else {
            self.clone()
        }
    }
}

pub fn standard_nonterm_parser<S: Hash + Clone + PartialEq + Eq + ParseSymbol>(
    s: &str,
    pos: usize,
) -> Result<(NonTerminal<S>, usize), &'static str> {
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

    let nt = NonTerminal {
        symbol: S::parse_nonterminal(&s[pos..index]).map_err(|_| "could not parse nonterminal")?,
    };
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
    S: Hash + Clone + PartialEq + Eq + ParseSymbol,
    F: Fn(&str, usize) -> Result<(NonTerminal<S>, usize), &'static str>,
>(
    line: &str,
    nt_parser: &F,
) -> Result<Vec<Production<S>>, &'static str> {
    let (lhs, mut pos) = nt_parser(line, 0)?;

    pos = read_arrow(line, pos).ok_or("no arrow")?;

    let mut rhssides: Vec<Vec<_>> = vec![vec![]];

    while pos < line.len() {
        match line[pos..].chars().next().unwrap() {
            '\'' | '"' => {
                let (end, xpos) =
                    terminate_str(line, pos).ok_or("could not terminate production")?;
                rhssides.last_mut().unwrap().push(SymbolWrapper::terminal(
                    ParseSymbol::parse_terminal(&line[pos + 1..end - 1]).unwrap(),
                ));
                pos = xpos;
            }
            '|' => {
                pos = eat_disjunction(line, pos).ok_or("no disjunction found")?;
                rhssides.push(vec![]);
            }
            _ => {
                let (rnt, xpos) = nt_parser(line, pos)?;
                rhssides
                    .last_mut()
                    .unwrap()
                    .push(SymbolWrapper::nonterminal(rnt.symbol));
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
    S: Hash + Clone + PartialEq + Eq + ParseSymbol,
    F: Fn(&str, usize) -> Result<(NonTerminal<S>, usize), &'static str>,
>(
    input: &str,
    nt_parser: F,
) -> Result<ContextFreeGrammar<S>, &'static str> {
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

pub trait ParseSymbol: Sized {
    type Error: Debug;

    fn parse_terminal(s: &str) -> Result<Self, Self::Error>;

    fn parse_nonterminal(s: &str) -> Result<Self, Self::Error>;
}

impl ParseSymbol for String {
    type Error = ();

    fn parse_terminal(s: &str) -> Result<Self, Self::Error> {
        Ok(s.to_string())
    }

    fn parse_nonterminal(s: &str) -> Result<Self, Self::Error> {
        Ok(s.to_string())
    }
}
