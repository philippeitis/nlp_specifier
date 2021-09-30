use std::io::Write;
use std::path::Path;
use std::process::Stdio;

use pyo3::types::IntoPyDict;
use pyo3::{PyObject, PyResult, Python, ToPyObject};
use syn::visit_mut::VisitMut;
use syn::{parse_file, Attribute, File, ImplItemMethod, ItemFn};

use chartparse::ChartParser;

use crate::docs::Docs;
use crate::grammar::AsSpec;
use crate::nl_ir::Specification;
use crate::parse_tree::tree::TerminalSymbol;
use crate::parse_tree::tree::S;
use crate::parse_tree::{Symbol, SymbolTree, Terminal};
use crate::search_tree::SearchTree;

pub enum SpacyModel {
    SM,
    MD,
    LG,
    TRF,
}

impl SpacyModel {
    fn spacy_ident(&self) -> &'static str {
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

    pub(crate) fn specify(&mut self, tokenizer: &Tokenizer, parser: &ChartParser<Symbol>) {
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
    parser: &'a ChartParser<'b, Symbol>,
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
                if let Ok(section_tokens) = self.tokenizer.tokenize_sents(&section.sentences) {
                    for sentence_tokens in section_tokens {
                        let (specs, _) = sentence_to_specifications(&self.parser, &sentence_tokens);
                        for spec in specs {
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

pub struct Tokenizer<'p> {
    parser: PyObject,
    py: Python<'p>,
}

impl<'p> Tokenizer<'p> {
    pub fn new<S: Into<SpacyModel>>(py: Python<'p>, model: S) -> Self {
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
            .into_py_dict(py);
        let parser: PyObject = py
            .eval("tokenizer.Tokenizer(model)", None, Some(locals))
            .unwrap()
            .extract()
            .unwrap();
        Tokenizer { parser, py }
    }

    pub fn tokenize_sents(&self, sents: &[String]) -> PyResult<Vec<Vec<(String, String, String)>>> {
        self.parser
            .call_method1(self.py, "stokenize", (sents.to_object(self.py), ))?
            .extract::<Vec<PyObject>>(self.py)?
            .into_iter()
            .map(|x| x.getattr(self.py, "metadata"))
            .collect::<PyResult<Vec<PyObject>>>()?
            .into_iter()
            .map(|x| x.extract::<Vec<(String, String, String)>>(self.py))
            .collect::<PyResult<Vec<Vec<(String, String, String)>>>>()
    }
}

pub struct SimMatcher<'p> {
    sim_matcher: PyObject,
    cutoff: f32,
    py: Python<'p>,
}

impl<'p> SimMatcher<'p> {
    pub fn new(py: Python<'p>, sentence: &str, parser: &Tokenizer<'p>, cutoff: f32) -> Self {
        let locals = [
            ("nlp_query", py.import("nlp_query").unwrap().to_object(py)),
            ("sent", sentence.to_object(py)),
            ("parser", parser.parser.clone()),
            ("cutoff", cutoff.to_object(py)),
        ]
            .into_py_dict(py);
        SimMatcher {
            sim_matcher: py
                .eval("nlp_query.SimPhrase(sent, parser, -1.)", None, Some(locals))
                .unwrap()
                .to_object(py),
            cutoff,
            py,
        }
    }

    pub fn is_similar(&self, sent: &str) -> PyResult<bool> {
        let sim: f32 = self
            .sim_matcher
            .call_method1(self.py, "sent_similarity", (sent, ))?
            .extract(self.py)?;
        Ok(sim > self.cutoff)
    }

    pub fn any_similar(&self, sents: &[String]) -> PyResult<bool> {
        self.sim_matcher
            .call_method(
                self.py,
                "any_similar",
                (sents.to_object(self.py), self.cutoff.to_object(self.py)),
                None,
            )?
            .extract(self.py)
    }

    pub fn print_seen(&self) {
        let sents_seen: usize = self
            .sim_matcher
            .getattr(self.py, "sents_seen")
            .unwrap()
            .extract(self.py)
            .unwrap();
        println!("{}", sents_seen);
    }
}

pub fn sentence_to_trees(
    parser: &ChartParser<Symbol>,
    sentence: &[(String, String, String)],
) -> Vec<SymbolTree> {
    let tokens: Result<Vec<TerminalSymbol>, _> = sentence
        .iter()
        .map(|(t, _, _)| TerminalSymbol::from_terminal(t))
        .collect();
    let tokens: Vec<_> = match tokens {
        Ok(t) => t.into_iter().map(Symbol::from).collect(),
        Err(_) => return Vec::new(),
    };

    let iter: Vec<_> = sentence
        .iter()
        .cloned()
        .map(|(_tok, text, lemma)| Terminal {
            word: text,
            lemma: lemma.to_lowercase(),
        })
        .collect();
    match parser.parse(&tokens) {
        Ok(trees) => trees
            .into_iter()
            .map(|t| SymbolTree::from_iter(t, &mut iter.clone().into_iter()))
            .collect(),
        Err(_) => return Vec::new(),
    }
}

pub fn sentence_to_specifications(
    parser: &ChartParser<Symbol>,
    sentence: &[(String, String, String)],
) -> (Vec<Specification>, usize) {
    let tokens: Result<Vec<TerminalSymbol>, _> = sentence
        .iter()
        .map(|(t, _, _)| TerminalSymbol::from_terminal(t))
        .collect();
    let tokens: Vec<_> = match tokens {
        Ok(t) => t.into_iter().map(Symbol::from).collect(),
        Err(_) => return (Vec::new(), 0),
    };

    let iter: Vec<_> = sentence
        .iter()
        .cloned()
        .map(|(_tok, text, lemma)| Terminal {
            word: text,
            lemma: lemma.to_lowercase(),
        })
        .collect();
    let trees: Vec<_> = match parser.parse(&tokens) {
        Ok(trees) => trees
            .into_iter()
            .map(|t| SymbolTree::from_iter(t, &mut iter.clone().into_iter()))
            .map(S::from)
            .collect(),
        Err(_) => return (Vec::new(), 0),
    };
    let len = trees.len();
    (trees.into_iter().map(Specification::from).collect(), len)
}
