use std::path::{Path, PathBuf};
use std::collections::HashSet;
use std::process::Stdio;
use std::io::Write;

use pyo3::{Python, PyResult, PyObject, ToPyObject, IntoPy};
use pyo3::types::IntoPyDict;
use pyo3::exceptions::{PyStopIteration, PyKeyboardInterrupt};

use syn::visit_mut::VisitMut;
use syn::{File, parse_file, Attribute, ImplItemMethod, ItemFn, PatPath};
use syn::parse::ParseStream;

use chartparse::{TreeWrapper, ChartParser, ContextFreeGrammar};
use chartparse::tree::TreeNode;

use clap::{AppSettings, Clap};

mod search_tree;
mod docs;
mod parse_html;
mod type_match;
mod jsonl;
mod grammar;
mod nl_ir;
mod parse_tree;
mod visualization;


use docs::Docs;
use search_tree::{SearchTree, SearchItem, Depth};
use type_match::{HasFnArg, FnArgLocation};
use parse_html::{toolchain_path_to_html_root, get_toolchain_dirs, file_from_root_dir};
use nl_ir::Specification;
use parse_tree::{SymbolTree, Terminal, Symbol, tree::TerminalSymbol};
use grammar::AsSpec;
use crate::visualization::tree::tag_color;
use itertools::Itertools;


#[macro_use]
extern crate lazy_static;

static TOKENIZER: &str = include_str!("../../nlp/tokenizer.py");
static FIX_TOKENS: &str = include_str!("../../nlp/fix_tokens.py");
static NER: &str = include_str!("../../nlp/ner.py");
static CFG: &str = include_str!("../../nlp/codegrammar.cfg");

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

    fn specify(&mut self, tokenizer: &Tokenizer, parser: &ChartParser<Symbol>) {
        SpecifierX { searcher: &self.searcher, tokenizer, parser }.visit_file_mut(&mut self.file)
    }
}

struct SpecifierX<'a, 'b, 'p> {
    searcher: &'a SearchTree,
    tokenizer: &'a Tokenizer<'p>,
    parser: &'a ChartParser<'b, Symbol>,
}


fn should_specify<A: AsRef<[Attribute]>>(attrs: A) -> bool {
    attrs.as_ref().iter().map(|attr| &attr.path).any(|x| quote::quote! {#x}.to_string() == "specify")
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

struct Tokenizer<'p> {
    parser: PyObject,
    py: Python<'p>,
}

impl<'p> Tokenizer<'p> {
    fn new(py: Python<'p>) -> Self {
        let path = std::env::current_dir().unwrap();
        let locals = [
            ("sys", py.import("sys").unwrap().to_object(py)),
            ("pathlib", py.import("pathlib").unwrap().to_object(py)),
            ("root_dir", path.to_object(py))
        ].into_py_dict(py);
        // TODO: Make sure we fix path handling for the general case.
        let code = "sys.path.extend([str(pathlib.Path(root_dir).parent / 'nlp'), str(pathlib.Path(root_dir).parent)])";
        py.eval(code, None, Some(locals)).unwrap();

        let locals = [("tokenizer", py.import("tokenizer").unwrap())].into_py_dict(py);
        let parser: PyObject = py.eval("tokenizer.Tokenizer()", None, Some(locals)).unwrap().extract().unwrap();
        Tokenizer { parser, py }
    }

    fn tokenize_sents(&self, sents: &[String]) -> PyResult<Vec<Vec<(String, String, String)>>> {
        self.parser.call_method1(self.py, "stokenize", (sents.to_object(self.py), ))?
            .extract::<Vec<PyObject>>(self.py)?
            .into_iter()
            .map(|x| x.getattr(self.py, "metadata"))
            .collect::<PyResult<Vec<PyObject>>>()?
            .into_iter()
            .map(|x| x.extract::<Vec<(String, String, String)>>(self.py))
            .collect::<PyResult<Vec<Vec<(String, String, String)>>>>()
    }
}

struct SimMatcher<'p> {
    sim_matcher: PyObject,
    cutoff: f32,
    py: Python<'p>,
}

impl<'p> SimMatcher<'p> {
    fn new(py: Python<'p>, sentence: &str, parser: &Tokenizer<'p>, cutoff: f32) -> Self {
        let locals = [
            ("nlp_query", py.import("nlp_query").unwrap().to_object(py)),
            ("sent", sentence.to_object(py)),
            ("parser", parser.parser.clone()),
            ("cutoff", cutoff.to_object(py)),
        ].into_py_dict(py);
        SimMatcher {
            sim_matcher: py.eval("nlp_query.SimPhrase(sent, parser, -1.)", None, Some(locals)).unwrap().to_object(py),
            cutoff,
            py,
        }
    }

    fn is_similar(&self, sent: &str) -> PyResult<bool> {
        let sim: f32 = self.sim_matcher.call_method1(self.py, "sent_similarity", (sent, ))?.extract(self.py)?;
        Ok(sim > self.cutoff)
    }

    fn any_similar(&self, sents: &[String]) -> PyResult<bool> {
        self
            .sim_matcher
            .call_method(
                self.py,
                "any_similar",
                (sents.to_object(self.py),
                 self.cutoff.to_object(self.py)),
                None,
            )?.extract(self.py)
    }

    fn print_seen(&self) {
        let sents_seen: usize = self.sim_matcher.getattr(self.py, "sents_seen").unwrap().extract(self.py).unwrap();
        println!("{}", sents_seen);
    }
}

fn sentence_to_trees(parser: &ChartParser<Symbol>, sentence: &[(String, String, String)]) -> Vec<SymbolTree> {
    let tokens: Result<Vec<TerminalSymbol>, _> = sentence.iter().map(|(t, _, _)| TerminalSymbol::from_terminal(t)).collect();
    let tokens: Vec<_> = match tokens {
        Ok(t) => t.into_iter().map(Symbol::from).collect(),
        Err(_) => return Vec::new(),
    };

    let iter: Vec<_> = sentence.iter().cloned().map(|(tok, text, lemma)| Terminal { word: text, lemma: lemma.to_lowercase() }).collect();
    match parser.parse(&tokens) {
        Ok(trees) => trees
            .into_iter()
            .map(|t| SymbolTree::from_iter(t, &mut iter.clone().into_iter()))
            .collect(),
        Err(_) => return Vec::new(),
    }
}

fn sentence_to_specifications(parser: &ChartParser<Symbol>, sentence: &[(String, String, String)]) -> (Vec<Specification>, usize) {
    let tokens: Result<Vec<TerminalSymbol>, _> = sentence.iter().map(|(t, _, _)| TerminalSymbol::from_terminal(t)).collect();
    let tokens: Vec<_> = match tokens {
        Ok(t) => t.into_iter().map(Symbol::from).collect(),
        Err(_) => return (Vec::new(), 0)
    };

    let iter: Vec<_> = sentence.iter().cloned().map(|(tok, text, lemma)| Terminal { word: text, lemma: lemma.to_lowercase() }).collect();
    let trees: Vec<_> = match parser.parse(&tokens) {
        Ok(trees) => trees
            .into_iter()
            .map(|t| SymbolTree::from_iter(t, &mut iter.clone().into_iter()))
            .map(parse_tree::tree::S::from)
            .collect(),
        Err(_) => return (Vec::new(), 0)
    };
    let len = trees.len();
    (trees
         .into_iter()
         .map(Specification::from)
         .collect(),
     len)
}

fn search_demo() {
    let start = std::time::Instant::now();
    let path = toolchain_path_to_html_root(&get_toolchain_dirs().unwrap()[0]);
    let tree = file_from_root_dir(&path).unwrap();
    let end = std::time::Instant::now();
    println!("Parsing Rust stdlib took {}s", (end - start).as_secs_f32());

    Python::with_gil(|py| -> PyResult<()> {
        let parser = Tokenizer::new(py);
        let matcher = SimMatcher::new(py, "The minimum of two values", &parser, 0.85);
        let start = std::time::Instant::now();
        let usize_first = HasFnArg { fn_arg_location: FnArgLocation::Output, fn_arg_type: Box::new("f32") };
        println!("{:?}", tree.search(&|item| usize_first.item_matches(item) && match &item.item {
            SearchItem::Fn(_) | SearchItem::Method(_) => {
                item.docs
                    .sections.first().map(|sect| matcher.any_similar(&sect.sentences).unwrap_or(false)).unwrap_or(false)
            }
            _ => false,
        }, Depth::Infinite).len());
        let end = std::time::Instant::now();
        println!("Search took {}s", (end - start).as_secs_f32());
        matcher.print_seen();
        Ok(())
    }).unwrap();
}

fn specify_docs<P: AsRef<Path>>(path: P) {
    let start = std::time::Instant::now();
    let tree = file_from_root_dir(&path).unwrap();
    let end = std::time::Instant::now();

    println!("Parsing Rust stdlib took {}s", (end - start).as_secs_f32());

    let cfg = ContextFreeGrammar::<Symbol>::fromstring(CFG.to_string()).unwrap();
    let parser = ChartParser::from_grammar(&cfg);

    let tokens = Python::with_gil(|py| -> PyResult<Vec<Vec<(String, String, String)>>> {
        let tokenizer = Tokenizer::new(py);

        let mut sentences = Vec::new();
        for value in tree.search(&|x| matches!(&x.item, SearchItem::Fn(_) | SearchItem::Method(_)) && !x.docs.sections.is_empty(), Depth::Infinite) {
            sentences.extend(&value.docs.sections[0].sentences);
        }

        let sentences: Vec<_> = sentences.into_iter().collect::<HashSet<_>>().into_iter().map(String::from).collect();

        let start = std::time::Instant::now();
        let tokens = tokenizer.tokenize_sents(&sentences)?;
        let end = std::time::Instant::now();
        println!("Time to tokenize sentences: {}", (end - start).as_secs_f32());
        Ok(tokens)
    }).unwrap();

    let mut ntrees = 0;
    let mut nspecs = 0;
    let mut successful_sents = 0;
    let mut unsucessful_sents = 0;
    let mut specified_sents = 0;

    let start = std::time::Instant::now();

    for metadata in tokens.iter() {
        let (specs, trees_len) = sentence_to_specifications(&parser, metadata);

        if !specs.is_empty() {
            // println!("{}", "=".repeat(80));
            // println!("Sentence: {}", sentence);
            // println!("    Tags: ?");
            // println!("{}", "=".repeat(80));

            // for (tree, spec) in trees.iter().zip(specs.iter()) {
            //     println!("{}", tree.call_method0(py, "__str__").unwrap().extract::<String>(py).unwrap());
            //     println!("{:?}", spec);
            // }

            // println!();
            successful_sents += 1;
        } else {
            unsucessful_sents += 1;
        }
        ntrees += trees_len;
        let count = specs.iter().map(Specification::as_spec).filter(Result::is_ok).count();
        if count != 0 {
            specified_sents += 1;
        }
        nspecs += count;
    }
    let end = std::time::Instant::now();
    println!("          Sentences: {}", tokens.len());
    println!("Successfully parsed: {}", successful_sents);
    println!("              Trees: {}", ntrees);
    println!("     Specifications: {}", nspecs);
    println!("Specified Sentences: {}", specified_sents);
    println!("       Time elapsed: {}", (end - start).as_secs_f32());

    //                  Sentences: 4946
    //        Successfully parsed: 284
    //                      Trees: 515
    //             Specifications: 155
    //        Specified Sentences: 114
}

fn specify_sentences(sentences: Vec<String>) {
    let cfg = ContextFreeGrammar::<Symbol>::fromstring(CFG.to_string()).unwrap();
    let parser = ChartParser::from_grammar(&cfg);

    let tokens = Python::with_gil(|py| -> PyResult<Vec<Vec<(String, String, String)>>> {
        Tokenizer::new(py).tokenize_sents(&sentences)
    }).unwrap();

    for (metadata, sentence) in tokens.iter().zip(sentences.iter()) {
        let tokens: Result<Vec<TerminalSymbol>, _> = metadata.iter().map(|(t, _, _)| TerminalSymbol::from_terminal(t)).collect();
        let tokens: Vec<_> = match tokens {
            Ok(t) => t.into_iter().map(Symbol::from).collect(),
            Err(_) => continue,
        };

        let iter: Vec<_> = metadata.iter().cloned().map(|(tok, text, lemma)| Terminal { word: text, lemma: lemma.to_lowercase() }).collect();
        let trees: Vec<_> = match parser.parse(&tokens) {
            Ok(trees) => trees
                .into_iter()
                .map(|t| SymbolTree::from_iter(t, &mut iter.clone().into_iter()))
                .map(parse_tree::tree::S::from)
                .collect(),
            Err(_) => continue,
        };
        let specs: Vec<_> = trees
            .clone()
            .into_iter()
            .map(Specification::from)
            .collect();
        println!("{}", "=".repeat(80));
        println!("Sentence: {}", sentence);
        println!("{}", "=".repeat(80));
        for spec in specs {
            match spec.as_spec() {
                Ok(attrs) => {
                    println!("SUCCESS");
                    for attr in attrs {
                        println!("{}", quote::quote! {#attr}.to_string());
                    }
                }
                Err(e) => {
                    println!("FAILURE: {:?}", e);
                }
            }
        }
    }
}

fn specify_file<P: AsRef<Path>>(path: P) -> File {
    let mut specifier = Specifier::from_path(&path).unwrap();

    let cfg = ContextFreeGrammar::<Symbol>::fromstring(CFG.to_string()).unwrap();
    let parser = ChartParser::from_grammar(&cfg);

    Python::with_gil(|py| -> PyResult<()> {
        let tokenizer = Tokenizer::new(py);

        // Do ahead of time to take advantage of parallelism
        let mut sentences = Vec::new();
        for value in specifier.searcher.search(&|x| matches!(&x.item, SearchItem::Fn(_) | SearchItem::Method(_)) && !x.docs.sections.is_empty(), Depth::Infinite) {
            sentences.extend(&value.docs.sections[0].sentences);
        }

        let sentences: Vec<_> = sentences.into_iter().collect::<HashSet<_>>().into_iter().map(String::from).collect();

        let _ = tokenizer.tokenize_sents(&sentences)?;
        specifier.specify(&tokenizer, &parser);
        Ok(())
    }).unwrap();
    specifier.file
}

#[derive(Clap)]
#[clap(setting = AppSettings::ColoredHelp)]
struct Opts {
    #[clap(subcommand)]
    command: SubCommand,
}

#[derive(Clap)]
enum SubCommand {
    #[clap(name = "end-to-end")]
    EndToEnd(EndToEnd),
    #[clap(name = "specify")]
    Specify {
        #[clap(subcommand)]
        sub_cmd: Specify
    },
    #[clap(name = "render")]
    Render {
        #[clap(subcommand)]
        sub_cmd: Render,
    },
}

/// Demonstrates entire pipeline from start to end on provided file, writing output to terminal.
#[derive(Clap)]
#[clap(setting = AppSettings::ColoredHelp)]
struct EndToEnd {
    /// Source file to specify.
    #[clap(short, long, parse(from_os_str), default_value = "../../data/test3.rs")]
    path: PathBuf,
}

/// Creates specifications for a variety of sources.
#[derive(Clap)]
#[clap(setting = AppSettings::ColoredHelp)]
enum Specify {
    /// Specifies the sentence, and prints the result to the terminal.
    #[clap(setting = AppSettings::ColoredHelp)]
    Sentence {
        sentence: String
    },
    /// Specifies the newline separated sentences in the provided file, and prints the results to the terminal.
    #[clap(setting = AppSettings::ColoredHelp)]
    Testcases {
        #[clap(short, long, parse(from_os_str), default_value = "../../data/base_grammar_test_cases.txt")]
        path: PathBuf,
    },
    /// Specifies the file at `path`, and writes the result to `dest`.
    #[clap(setting = AppSettings::ColoredHelp)]
    File {
        /// The root file to be parsed. If no file is provided, defaults to a test case.
        #[clap(short, long, parse(from_os_str), default_value = "../../data/test3.rs")]
        path: PathBuf,
        /// The path to write the specified file to. If no path is provided, defaults to the provided
        /// path, appending "_specified" file stem.
        #[clap(short, long, parse(from_os_str))]
        dest: Option<PathBuf>,
    },
    /// Specifies each item in the documentation at the provided path. If no path is provided,
    /// defaults to Rust's standard library documentation.
    ///
    /// Documentation can be generated using `cargo doc`, or can be downloaded via `rustup`.
    #[clap(setting = AppSettings::ColoredHelp)]
    Docs {
        /// The root directory of the documentation to be parsed. If no directory is provided,
        /// defaults to the Rust toolchain root.
        #[clap(short, long, parse(from_os_str))]
        path: Option<PathBuf>,
    },
    /// Provides a REPL for specifying sentences repeatedly.
    #[clap(setting = AppSettings::ColoredHelp)]
    Repl,
}

/// Visualization of various components in the pipeline
#[derive(Clap)]
#[clap(setting = AppSettings::ColoredHelp)]
enum Render {
    #[clap(setting = AppSettings::ColoredHelp)]
    ParseTree {
        /// Sentence to specify.
        sentence: String,
        /// Path to write output to.
        #[clap(short, long, parse(from_os_str), default_value = "../../images/parse_tree.pdf")]
        path: PathBuf,
        /// Open file in browser
        #[clap(short, long)]
        open_browser: bool,
    },
}

fn repl(py: Python) -> PyResult<()> {
    use pastel::Color;
    let cfg = ContextFreeGrammar::<Symbol>::fromstring(CFG.to_string()).unwrap();
    let parser = ChartParser::from_grammar(&cfg);
    let brush = pastel::ansi::Brush::from_environment(pastel::ansi::Stream::Stdout);

    println!("Running doc_parser REPL. Type \"exit\" or \"quit\" to terminate the REPL.");
    let tokenizer = Tokenizer::new(py);
    println!("Finished loading parser.");
    loop {
        // Python hijacks stdin
        let sent: String = py.eval("input(\'>>> \')", None, None)?.extract()?;
        if ["exit", "quit"].contains(&sent.as_str()) {
            break;
        }
        let tokens = tokenizer.tokenize_sents(&[sent])?.remove(0);
        println!("Tokens: {}", tokens.iter().map(|(token, _, _)| {
            match TerminalSymbol::from_terminal(token) {
                Ok(sym) => {
                    let sym = Symbol::from(sym);
                    let mut color = pastel::parser::parse_color(tag_color(&sym)).unwrap();
                    if color == Color::black() {
                        color = pastel::parser::parse_color("#24e3dd").unwrap();
                    }
                    brush.paint(token, color)
                }
                Err(_) => {
                    brush.paint(token, Color::white())
                }
            }
        }).join(" "));
        let (specs, _) = sentence_to_specifications(&parser, &tokens);
        if specs.is_empty() {
            println!("{}", brush.paint("No specification generated", Color::red()));
        }
        for (i, spec) in specs.iter().enumerate() {
            println!("Specification {}/{}", i + 1, specs.len());
            match spec.as_spec() {
                Ok(attrs) => {
                    for attr in attrs {
                        println!("{}", quote::quote! {#attr}.to_string());
                    }
                }
                Err(e) => {
                    println!("{}", brush.paint(format!("FAILURE: {:?}", e), Color::red()));
                }
            }
        }
    }
    Ok(())
}

fn main() {
    let opts: Opts = Opts::parse();
    // TODO: Detect duplicate invocations.
    // TODO: keyword in fn name, capitalization?
    // TODO: similarity metrics (capitalization, synonym distance via wordnet)
    // TODO: Decide spurious keywords
    //
    // TODO: Mechanism to evaluate code quality
    // TODO: Add type to CODE item? eg. CODE_USIZE, CODE_BOOL, CODE_STR, then make CODE accept all of these
    //  std::any::type_name_of_val
    // TODO: allow specifying default value in #[invoke]
    //  eg. #[invoke(str, arg1 = 1usize, arg2 = ?, arg3 = ?)]
    //
    // TODO:
    //  1. parse sent into tokens (falliable)
    //  2. parse tokens into trees (infalliable)
    //  3. parse tree in initial type (infalliable)
    //  4. unresolved code blocks (infalliable)
    //  5. resolved code items (falliable)
    //  6. final specification (infalliable)
    match opts.command {
        SubCommand::EndToEnd(EndToEnd { path }) => {
            let file = specify_file(path);
            let output = quote::quote! {#file}.to_string();
            let mut cmd = std::process::Command::new("rustfmt")
                .arg("--emit")
                .arg("stdout")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            cmd.stdin.as_mut().unwrap().write(output.as_bytes()).unwrap();
            let output = cmd.wait_with_output().unwrap();
            println!("{}", String::from_utf8(output.stdout).unwrap());
        }
        SubCommand::Specify { sub_cmd } => match sub_cmd {
            Specify::Sentence { sentence } => specify_sentences(vec![sentence]),
            Specify::File { path, dest } => {
                let dest = match dest {
                    None => {
                        let mut stem = path.file_stem().unwrap().to_os_string();
                        stem.push("_specified.rs");
                        path.with_file_name(stem)
                    }
                    Some(dest) => dest,
                };
                let file = specify_file(path);
                let output = quote::quote! {#file}.to_string();
                std::fs::write(&dest, output).unwrap();

                if std::process::Command::new("rustfmt").arg(&dest).spawn().is_ok() {
                    println!("Formatted output file, result available at {}", dest.display())
                } else {
                    println!("Output file {} could not be formatted", dest.display())
                }
            }
            Specify::Docs { path } => {
                let path = path.unwrap_or_else(||
                    toolchain_path_to_html_root(&get_toolchain_dirs().unwrap()[0])
                );
                specify_docs(path);
            }
            Specify::Testcases { path } => {
                let testcases = std::fs::read_to_string(path).unwrap();
                specify_sentences(testcases.lines().into_iter().map(String::from).collect())
            }
            Specify::Repl => {
                Python::with_gil(|py| match repl(py) {
                    Ok(_) => {}
                    Err(e) if e.is_instance::<PyKeyboardInterrupt>(py) => {
                        println!()
                    }
                    Err(e) => {
                        println!("Process failed due to {:?}", e);
                    }
                });
            }
        },
        SubCommand::Render { sub_cmd } => match sub_cmd {
            Render::ParseTree { sentence, path, open_browser } => {
                let cfg = ContextFreeGrammar::<Symbol>::fromstring(CFG.to_string()).unwrap();
                let parser = ChartParser::from_grammar(&cfg);
                Python::with_gil(|py| -> PyResult<()> {
                    let tokenizer = Tokenizer::new(py);
                    let tokens = tokenizer.tokenize_sents(&vec![sentence])?.remove(0);
                    let trees = sentence_to_trees(&parser, &tokens);
                    match trees.first() {
                        None => println!("No tree generated for the provided sentence."),
                        Some(tree) => visualization::tree::render_tree(tree, &path).unwrap(),
                    }
                    if open_browser {
                        py.run("import webbrowser; webbrowser.open(path)", None, Some([("path", path.to_object(py))].into_py_dict(py)))?;
                    }
                    Ok(())
                }).unwrap();
            }
        }
    }
}

