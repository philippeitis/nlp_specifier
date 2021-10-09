use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::rc::Rc;

use pyo3::types::IntoPyDict;
use pyo3::{PyObject, PyResult, Python, ToPyObject};
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

struct SpecifierX<'a, 'b, 'p> {
    searcher: &'a SearchTree,
    tokenizer: &'a Tokenizer<'p>,
    parser: &'a ChartParser<'b, Symbol, TerminalSymbol>,
}

fn should_specify<A: AsRef<[Attribute]>>(attrs: A) -> bool {
    attrs
        .as_ref()
        .iter()
        .map(|attr| &attr.path)
        .any(|x| quote::quote! {#x}.to_string() == "specify")
}

impl<'a, 'b, 'p> SpecifierX<'a, 'b, 'p> {
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

impl<'a, 'b, 'p> VisitMut for SpecifierX<'a, 'b, 'p> {
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
pub struct Tokenizer<'p> {
    parser: PyObject,
    py: Python<'p>,
    cache: RefCell<HashMap<String, Rc<Sentence>>>,
}

impl<'p> Tokenizer<'p> {
    fn init_with_args<S: Into<SpacyModel>>(
        py: Python<'p>,
        tokenizer_call: &str,
        model: S,
        args: &[(&str, PyObject)],
    ) -> Self {
        let path = std::env::current_dir().unwrap();
        let locals = [
            ("sys", py.import("sys").unwrap().to_object(py)),
            ("pathlib", py.import("pathlib").unwrap().to_object(py)),
            ("root_dir", path.to_object(py)),
        ]
        .into_py_dict(py);
        // TODO: Make sure we fix path handling for the general case.
        let code = "sys.path.extend([str(pathlib.Path(root_dir).parent / 'nlp'), str(pathlib.Path(root_dir).parent)])";
        py.eval(code, None, Some(locals)).unwrap();

        let locals = [
            ("tokenizer", py.import("tokenizer").unwrap().to_object(py)),
            ("model", model.into().spacy_ident().to_object(py)),
        ]
        .iter()
        .chain(args.into_iter())
        .into_py_dict(py);

        let parser: PyObject = py
            .eval(&format!("tokenizer.{}", tokenizer_call), None, Some(locals))
            .unwrap()
            .extract()
            .unwrap();

        Tokenizer {
            parser,
            py,
            cache: RefCell::default(),
        }
    }
    pub fn new<S: Into<SpacyModel>>(py: Python<'p>, model: S) -> Self {
        Self::init_with_args(py, "Tokenizer(model)", model, &[])
    }

    pub fn from_cache<P: AsRef<Path>, S: Into<SpacyModel>>(
        py: Python<'p>,
        path: P,
        model: S,
    ) -> Self {
        Self::init_with_args(
            py,
            "Tokenizer.from_cache(path, model)",
            model,
            &[("path", path.as_ref().to_object(py))],
        )
    }

    pub fn tokenize_sents(&self, sents: &[String]) -> PyResult<Vec<Rc<Sentence>>> {
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

        {
            let mut cache = self.cache.borrow_mut();
            for (doc, src) in self
                .parser
                .call_method1(self.py, "stokenize", (to_parse.to_object(self.py),))?
                .extract::<Vec<PyObject>>(self.py)?
                .into_iter()
                .zip(to_parse)
            {
                let tokens: Vec<(String, String, String)> =
                    doc.getattr(self.py, "metadata")?.extract(self.py)?;
                let tokens: Vec<_> = tokens.into_iter().map(Token::from).collect();
                let doc = doc.getattr(self.py, "doc")?;
                let sent = doc.getattr(self.py, "text")?.extract(self.py)?;
                let vector = doc
                    .getattr(self.py, "vector")?
                    .call_method0(self.py, "tolist")?
                    .extract(self.py)?;
                // (vec<token>, str, vec<f32>
                cache.insert(src, Rc::new(Sentence::new(sent, tokens, vector)));
            }
        }

        let cache = self.cache.borrow();
        Ok(sents
            .into_iter()
            .map(|(sent, res)| res.unwrap_or_else(|| cache.get(sent).unwrap().clone()))
            .collect())
    }

    pub fn write_data<P: AsRef<Path>>(&self, path: P) -> PyResult<()> {
        self.parser
            .call_method1(self.py, "write_data", (path.as_ref().to_object(self.py),))?;
        Ok(())
    }
}

pub struct SimMatcher<'a, 'p> {
    tokenizer: &'a Tokenizer<'p>,
    sentence: Rc<Sentence>,
    cutoff: f32,
}

impl<'a, 'p> SimMatcher<'a, 'p> {
    pub fn new(sentence: &str, tokenizer: &'a Tokenizer<'p>, cutoff: f32) -> Self {
        SimMatcher {
            sentence: tokenizer
                .tokenize_sents(&[sentence.to_string()])
                .unwrap()
                .remove(0),
            tokenizer,
            cutoff,
        }
    }

    pub fn is_similar(&self, sent: &str) -> PyResult<bool> {
        let other = self
            .tokenizer
            .tokenize_sents(&[sent.to_string()])?
            .remove(0);
        Ok(self.sentence.similarity(&other) > self.cutoff)
    }

    pub fn any_similar(&self, sents: &[String]) -> PyResult<bool> {
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
