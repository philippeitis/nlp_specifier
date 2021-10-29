#![deny(unused_imports)]
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use std::process::Stdio;
use std::rc::Rc;

use reqwest::blocking::Client;
use reqwest::Url;
use syn::visit_mut::VisitMut;
use syn::{parse_file, Attribute, File, ImplItemMethod, ItemFn};

use chartparse::ChartParser;

use crate::docs::Docs;
use crate::grammar::AsSpec;
use crate::nl_ir::Specification;
use crate::parse_tree::tree::{TerminalSymbol, S};
use crate::parse_tree::Symbol;
use crate::search_tree::SearchTree;
use crate::sentence::{Sentence, Token};

#[derive(Copy, Clone)]
pub enum SpacyModel {
    SM,
    MD,
    LG,
    TRF,
}

impl SpacyModel {
    pub fn spacy_ident(&self) -> &'static str {
        match self {
            SpacyModel::SM => "en_core_web_sm",
            SpacyModel::MD => "en_core_web_md",
            SpacyModel::LG => "en_core_web_lg",
            SpacyModel::TRF => "en_core_web_trf",
        }
    }
}

#[derive(Debug)]
pub enum SpecError {
    Io(std::io::Error),
    Syn(syn::Error),
}

impl From<std::io::Error> for SpecError {
    fn from(e: std::io::Error) -> Self {
        SpecError::Io(e)
    }
}

impl From<syn::Error> for SpecError {
    fn from(e: syn::Error) -> Self {
        SpecError::Syn(e)
    }
}

pub struct Specifier {
    pub file: File,
    pub searcher: SearchTree,
}

pub enum FileOutput {
    Fmt(String),
    NoFmt(String, std::io::Error),
}

impl Specifier {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, SpecError> {
        let s = std::fs::read_to_string(path)?;
        Ok(Self::new(parse_file(&s)?))
    }

    fn new(file: File) -> Self {
        Self {
            searcher: SearchTree::new(&file),
            file,
        }
    }

    pub(crate) fn specify(
        &mut self,
        tokenizer: &Tokenizer,
        parser: &ChartParser<Symbol, TerminalSymbol>,
    ) {
        SpecifierX {
            searcher: &self.searcher,
            tokenizer,
            parser,
        }
        .visit_file_mut(&mut self.file)
    }

    /// Formats the given Rust file (in byte format) using rustfmt if possible.
    fn rustfmt<S: AsRef<[u8]>>(bytes: S) -> std::io::Result<String> {
        let mut cmd = std::process::Command::new("rustfmt")
            .arg("--emit")
            .arg("stdout")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        cmd.stdin.as_mut().unwrap().write_all(bytes.as_ref())?;
        let output = cmd.wait_with_output()?;
        match String::from_utf8(output.stdout) {
            Ok(s) => Ok(s),
            Err(e) => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        }
    }

    pub fn to_fmt_string(&self) -> FileOutput {
        let file = &self.file;
        let output = quote::quote! {#file}.to_string();
        match Self::rustfmt(output.as_bytes()) {
            Ok(output) => FileOutput::Fmt(output),
            Err(e) => FileOutput::NoFmt(output, e),
        }
    }
}

struct SpecifierX<'a, 'b> {
    searcher: &'a SearchTree,
    tokenizer: &'a Tokenizer,
    parser: &'a ChartParser<'b, Symbol, TerminalSymbol>,
}

fn should_specify<A: AsRef<[Attribute]>>(attrs: A) -> bool {
    attrs
        .as_ref()
        .iter()
        .map(|attr| &attr.path)
        .any(|x| quote::quote! {#x}.to_string() == "specify")
}

impl<'a, 'b> SpecifierX<'a, 'b> {
    fn specify_docs(&self, attrs: &mut Vec<Attribute>) {
        if should_specify(&attrs) {
            let docs = Docs::from(&attrs);
            if let Some(section) = docs.sections.first() {
                if let Ok(sents) = self.tokenizer.tokenize_sents(&section.sentences) {
                    for sent in sents {
                        for spec in sentence_to_specifications(&self.parser, &sent) {
                            if let Ok(new_attrs) = spec.as_spec() {
                                attrs.extend(new_attrs);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<'a, 'b> VisitMut for SpecifierX<'a, 'b> {
    fn visit_impl_item_method_mut(&mut self, i: &mut ImplItemMethod) {
        self.specify_docs(&mut i.attrs)
    }

    fn visit_item_fn_mut(&mut self, i: &mut ItemFn) {
        self.specify_docs(&mut i.attrs)
    }
}

/// Tokenizes sentences, using spaCy through pyo3.
/// Tokenizations are cached internally, which provides a speedup of ~700x
/// over using Python-side caching (likely due to unidecode overhead).
pub struct Tokenizer {
    url: Url,
    model: SpacyModel,
    cache: RefCell<HashMap<String, Rc<Sentence>>>,
}

/// Tokenizes sentences, using spaCy through pyo3.
/// Tokenizations are cached internally, which provides a speedup of ~700x
/// over using Python-side caching (likely due to unidecode overhead).
impl Tokenizer {
    pub fn new<S: Into<SpacyModel>>(url: Url, model: S) -> Self {
        Self {
            url,
            model: model.into(),
            cache: Default::default(),
        }
    }

    pub fn tokenize_sents(&self, sents: &[String]) -> Result<Vec<Rc<Sentence>>, ()> {
        fn read_string<R: Read>(reader: &mut R) -> String {
            let str_len = rmp::decode::read_str_len(reader).unwrap();
            let mut buf = vec![0; str_len as usize];
            reader.read_exact(&mut buf).unwrap();
            String::from_utf8(buf).unwrap()
        }

        let sents: Vec<_> = {
            let cache = self.cache.borrow();
            sents.iter().map(|s| (s, cache.get(s).cloned())).collect()
        };

        let to_parse: Vec<_> = sents
            .iter()
            .cloned()
            .filter_map(|(sent, tokens)| match tokens {
                None => Some(sent.to_string()),
                Some(_) => None,
            })
            .collect();

        if !to_parse.is_empty() {
            let client = Client::new();
            let request_json = serde_json::json!({
                "model": self.model.spacy_ident(),
                "sentences": to_parse,
            });

            let bytes = client
                .get(self.url.join("/tokenize").unwrap())
                .json(&request_json)
                .send()
                .unwrap()
                .bytes()
                .unwrap();

            let mut cache = self.cache.borrow_mut();
            let mut bytes = std::io::Cursor::new(bytes);
            let _map_len = rmp::decode::read_map_len(&mut bytes).unwrap();
            assert_eq!(read_string(&mut bytes), "sentences");
            let num_sents = rmp::decode::read_array_len(&mut bytes).unwrap();
            for (_, src) in (0..num_sents).zip(to_parse) {
                let sent_map_len = rmp::decode::read_map_len(&mut bytes).unwrap();
                // can have text, tokens, and vector in that order
                let mut text = None;
                let mut tokens = None;
                let mut vector = None;
                let mut buf = [0; 16];
                for _ in 0..sent_map_len {
                    match rmp::decode::read_str(&mut bytes, &mut buf).unwrap() {
                        "text" => {
                            text = Some(read_string(&mut bytes));
                        }
                        "tokens" => {
                            let num_tokens = rmp::decode::read_array_len(&mut bytes).unwrap();
                            let mut token_vec = Vec::with_capacity(num_tokens as usize);
                            for _ in 0..num_tokens {
                                rmp::decode::read_array_len(&mut bytes).unwrap();
                                let tag = read_string(&mut bytes);
                                let tok_text = read_string(&mut bytes);
                                let lemma = read_string(&mut bytes);
                                token_vec.push(Token::from((tag, tok_text, lemma)));
                            }
                            tokens = Some(token_vec);
                        }
                        "vector" => {
                            let num_bytes = rmp::decode::read_bin_len(&mut bytes).unwrap();
                            let mut float_vec = Vec::with_capacity((num_bytes as usize) / 4);
                            for _ in 0..num_bytes / 4 {
                                let mut float_buf = [0; 4];
                                bytes.read_exact(&mut float_buf).unwrap();
                                float_vec.push(f32::from_le_bytes(float_buf))
                            }
                            vector = Some(float_vec);
                        }
                        _ => {}
                    }
                }
                match (text, tokens) {
                    (Some(sent), Some(tokens)) => {
                        cache.insert(
                            src,
                            Rc::new(Sentence::new(sent, tokens, vector.unwrap_or_default())),
                        );
                    }
                    _ => {}
                }
            }
        }

        let cache = self.cache.borrow();
        Ok(sents
            .into_iter()
            .map(|(sent, res)| res.unwrap_or_else(|| cache.get(sent).unwrap().clone()))
            .collect())
    }

    pub fn persist_cache(&self) {
        let client = Client::new();
        let _ = client.post(self.url.join("/persist_cache").unwrap()).send();
    }

    pub fn explain(&self, s: &str) -> Option<String> {
        use serde::Deserialize;
        #[derive(Deserialize)]
        struct Explanation {
            explanation: Option<String>,
        }

        let client = Client::new();

        let response = client
            .post(self.url.join("/explain").unwrap())
            .body(s.to_string())
            .send()
            .ok()?;

        let json = response.json::<Explanation>().ok()?;
        json.explanation
    }
}

pub struct SimMatcher<'a> {
    tokenizer: &'a Tokenizer,
    sentence: Rc<Sentence>,
    cutoff: f32,
}

impl<'a> SimMatcher<'a> {
    pub fn new(sentence: &str, tokenizer: &'a Tokenizer, cutoff: f32) -> Self {
        SimMatcher {
            sentence: tokenizer
                .tokenize_sents(&[sentence.to_string()])
                .unwrap()
                .remove(0),
            tokenizer,
            cutoff,
        }
    }

    pub fn is_similar(&self, sent: &str) -> Result<bool, ()> {
        let other = self
            .tokenizer
            .tokenize_sents(&[sent.to_string()])?
            .remove(0);
        Ok(self.sentence.similarity(&other) > self.cutoff)
    }

    pub fn any_similar(&self, sents: &[String]) -> Result<bool, ()> {
        let other = self.tokenizer.tokenize_sents(sents)?;
        Ok(other
            .into_iter()
            .any(|sent| self.sentence.similarity(&sent) > self.cutoff))
    }

    pub fn print_seen(&self) {
        println!("{}", 0);
    }
}

pub fn sentence_to_specifications(
    parser: &ChartParser<Symbol, TerminalSymbol>,
    sentence: &Sentence,
) -> Vec<Specification> {
    sentence
        .parse_trees(&parser)
        .into_iter()
        .map(S::from)
        .map(Specification::from)
        .collect()
}
