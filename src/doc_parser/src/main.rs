use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::Parser;

use chartparse::ChartParser;

mod analysis;
mod docs;
mod grammar;
mod jsonl;
mod nl_ir;
mod parse_html;
mod parse_json;
mod parse_tree;
mod search_tree;
mod sentence;
mod specifier;
mod type_match;
mod visualization;

use analysis::count_subsequences_from_tokens;
use grammar::AsSpec;
use itertools::Itertools;
use nl_ir::Specification;
use parse_html::{file_from_root_dir, get_toolchain_dirs, toolchain_path_to_html_root};
use parse_tree::{tree::TerminalSymbol, Symbol};
use reqwest::Url;
use search_tree::{Depth, SearchItem};

use crate::parse_html::DocItem;
use crate::specifier::Tokenizer;
use specifier::{sentence_to_specifications, FileOutput, SimMatcher, SpacyModel, Specifier};
use type_match::{FnArgLocation, HasFnArg};
use visualization::tree::tag_color;

#[macro_use]
extern crate lazy_static;

static CFG: &str = include_str!("../codegrammar.cfg");

type ContextFreeGrammar = chartparse::ContextFreeGrammar<Symbol, TerminalSymbol>;

struct ModelOptions {
    model: SpacyModel,
    url: Url,
}

impl ModelOptions {
    fn new(model: SpacyModelCli, cache: Option<Url>) -> Self {
        let model = SpacyModel::from(model);
        let url = match cache {
            None => Url::parse("http://0.0.0.0:5000").unwrap(),
            Some(url) => url,
        };
        ModelOptions { model, url }
    }
}

#[derive(clap::ArgEnum, Copy, Clone)]
pub enum SpacyModelCli {
    SM,
    MD,
    LG,
    TRF,
}

impl From<SpacyModelCli> for SpacyModel {
    fn from(s: SpacyModelCli) -> Self {
        match s {
            SpacyModelCli::SM => SpacyModel::SM,
            SpacyModelCli::MD => SpacyModel::MD,
            SpacyModelCli::LG => SpacyModel::LG,
            SpacyModelCli::TRF => SpacyModel::TRF,
        }
    }
}

fn parse_url(s: &str) -> Result<Url, url::ParseError> {
    Url::parse(s)
}
#[derive(Parser)]
struct Opts {
    #[clap(short, long, arg_enum, default_value = "lg")]
    model: SpacyModelCli,
    #[clap(short, long, parse(try_from_str = parse_url))]
    url: Option<Url>,
    #[clap(subcommand)]
    command: SubCommand,
}

#[derive(Parser)]
enum SubCommand {
    #[clap(name = "end-to-end")]
    EndToEnd(EndToEnd),
    Specify {
        #[clap(subcommand)]
        sub_cmd: Specify,
    },
    Render {
        #[clap(subcommand)]
        sub_cmd: Render,
    },
    Search,
    Parse {
        path: PathBuf,
    },
    Index {
        path: PathBuf,
    },
}

/// Demonstrates entire pipeline from start to end on provided file, writing output to terminal.
#[derive(Parser)]
struct EndToEnd {
    /// Source file to specify.
    #[clap(parse(from_os_str), default_value = "../../data/test3.rs")]
    path: PathBuf,
}

/// Creates specifications for a variety of sources.
#[derive(Parser)]
enum Specify {
    /// Specifies the sentence, and prints the result to the terminal.
    Sentence { sentence: String },
    /// Specifies the newline separated sentences in the provided file, and prints the results to the terminal.
    Testcases {
        #[clap(
            short,
            long,
            parse(from_os_str),
            default_value = "../../data/base_grammar_test_cases.txt"
        )]
        path: PathBuf,
    },
    /// Specifies the file at `path`, and writes the result to `dest`.
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
    Docs {
        /// The root directory of the documentation to be parsed. If no directory is provided,
        /// defaults to the Rust toolchain root.
        #[clap(short, long, parse(from_os_str))]
        path: Option<PathBuf>,
    },
    /// Provides a REPL for specifying sentences repeatedly.
    Repl,
}

/// Visualization of various components in the pipeline
#[derive(Parser)]
enum Render {
    ParseTree {
        /// Sentence to specify.
        sentence: String,
        /// Path to write output to.
        #[clap(
            short,
            long,
            parse(from_os_str),
            default_value = "../../images/parse_tree.pdf"
        )]
        path: PathBuf,
        /// Open file in browser
        #[clap(short, long)]
        open_browser: bool,
    },
}

fn search_demo(options: &ModelOptions) {
    let start = std::time::Instant::now();
    let path = toolchain_path_to_html_root(&get_toolchain_dirs().unwrap()[0]);
    let tree = file_from_root_dir(&path).unwrap();
    let end = std::time::Instant::now();
    println!("Parsing Rust stdlib took {}s", (end - start).as_secs_f32());

    let parser = Tokenizer::new(options.url.clone(), options.model);
    let matcher = SimMatcher::new("The minimum of two values", &parser, 0.85);
    for _ in 0..2 {
        let start = std::time::Instant::now();
        let usize_first = HasFnArg {
            fn_arg_location: FnArgLocation::Output,
            fn_arg_type: Box::new("f32"),
        };
        let items = tree.search(
            &|item| {
                usize_first.item_matches(item)
                    && match &item.item {
                        SearchItem::Fn(_) | SearchItem::Method(_) => item
                            .docs
                            .sections
                            .first()
                            .map(|sect| matcher.any_similar(&sect.sentences).unwrap_or(false))
                            .unwrap_or(false),
                        _ => false,
                    }
            },
            Depth::Infinite,
        );
        println!("{:?}", items.len());
        for item in items {
            println!("{}", item.docs);
        }
        let end = std::time::Instant::now();
        println!("Search took {}s", (end - start).as_secs_f32());
    }
    matcher.print_seen();
}

fn specify_docs<P: AsRef<Path>>(path: P, options: &ModelOptions) {
    let start = std::time::Instant::now();
    let tree = file_from_root_dir(&path).unwrap();
    let end = std::time::Instant::now();

    println!("Parsing Rust stdlib took {}s", (end - start).as_secs_f32());

    let cfg = ContextFreeGrammar::from_str(CFG).unwrap();
    let parser = ChartParser::from_grammar(&cfg);

    let tokenizer = Tokenizer::new(options.url.clone(), options.model);

    let mut sentences = Vec::new();
    for value in tree.search(
        &|x| {
            matches!(&x.item, SearchItem::Fn(_) | SearchItem::Method(_))
                && !x.docs.sections.is_empty()
        },
        Depth::Infinite,
    ) {
        sentences.extend(&value.docs.sections[0].sentences);
    }

    let sentences: Vec<_> = sentences
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .map(String::from)
        .collect();

    let start = std::time::Instant::now();
    let tokens = tokenizer.tokenize_sents(&sentences).unwrap();
    let end = std::time::Instant::now();
    println!(
        "Time to tokenize sentences: {}",
        (end - start).as_secs_f32()
    );

    let mut ntrees = 0;
    let mut nspecs = 0;
    let mut successful_sents = 0;
    let mut specified_sents = 0;

    let start = std::time::Instant::now();

    let mut unsuccessful_vec = Vec::new();
    for sent in tokens.iter() {
        let specs = sentence_to_specifications(&parser, &sent);

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
            unsuccessful_vec.push(sent);
        }
        ntrees += specs.len();
        let count = specs
            .iter()
            .map(Specification::as_spec)
            .filter(Result::is_ok)
            .count();
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

    count_subsequences_from_tokens(&unsuccessful_vec);
    //                  Sentences: 4946
    //        Successfully parsed: 284
    //                      Trees: 515
    //             Specifications: 155
    //        Specified Sentences: 114
}

fn specify_sentences(sentences: Vec<String>, options: &ModelOptions) {
    let cfg = ContextFreeGrammar::from_str(CFG).unwrap();
    let parser = ChartParser::from_grammar(&cfg);

    let tokens = Tokenizer::new(options.url.clone(), options.model)
        .tokenize_sents(&sentences)
        .unwrap();

    for sent in tokens.iter() {
        let specs: Vec<_> = sentence_to_specifications(&parser, &sent);
        println!("{}", "=".repeat(80));
        println!("Sentence: {}", &sent.text);
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

fn specify_file<P: AsRef<Path>>(path: P, options: &ModelOptions) -> Specifier {
    let mut specifier = Specifier::from_path(&path).unwrap();

    let cfg = ContextFreeGrammar::from_str(CFG).unwrap();
    let parser = ChartParser::from_grammar(&cfg);

    let tokenizer = Tokenizer::new(options.url.clone(), options.model);

    // Do ahead of time to take advantage of parallelism
    let mut sentences = Vec::new();
    for value in specifier.searcher.search(
        &|x| {
            matches!(&x.item, SearchItem::Fn(_) | SearchItem::Method(_))
                && !x.docs.sections.is_empty()
        },
        Depth::Infinite,
    ) {
        sentences.extend(&value.docs.sections[0].sentences);
    }

    let sentences: Vec<_> = sentences
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .map(String::from)
        .collect();

    let _ = tokenizer.tokenize_sents(&sentences).unwrap();
    specifier.specify(&tokenizer, &parser);
    specifier
}

fn repl(options: &ModelOptions) {
    use pastel::ansi::Brush;
    use pastel::Color;

    let cfg = ContextFreeGrammar::from_str(CFG).unwrap();
    let parser = ChartParser::from_grammar(&cfg);
    let brush = pastel::ansi::Brush::from_environment(pastel::ansi::Stream::Stdout);

    let light_red = Color::from_rgb(255, 85, 85);
    fn paint_string(brush: &Brush, text: &str, tag: TerminalSymbol) -> String {
        let sym = Symbol::from(tag);
        let mut color = pastel::parser::parse_color(tag_color(&sym)).unwrap();
        if color == Color::black() {
            color = pastel::parser::parse_color("#24e3dd").unwrap();
        }
        brush.paint(text, color)
    }

    println!("Running doc_parser REPL. Type \"exit\" or \"quit\" to terminate the REPL.");
    println!("Commands:");
    println!("!explain: explain a particular token");
    println!("!lemma: display the lemmas in a particular sentence");

    let tokenizer = Tokenizer::new(options.url.clone(), options.model);
    loop {
        let mut sent = String::new();
        // Python hijacks stdin
        print!(">>> ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut sent).unwrap();
        let sent = sent.trim().to_string();
        if ["exit", "quit"].contains(&sent.as_str()) {
            break;
        }
        if sent.starts_with("!explain") {
            if let Some((_, keyword)) = sent.split_once(' ') {
                if let Some(explanation) = tokenizer.explain(keyword) {
                    println!("{}", explanation);
                } else {
                    println!("No explanation was found for {}", keyword);
                }
            }
            continue;
        } else if sent.starts_with("!lemma") {
            if let Some((_, keyword)) = sent.split_once(' ') {
                let sent = tokenizer
                    .tokenize_sents(&[keyword.to_string()])
                    .unwrap()
                    .remove(0);
                println!(
                    "{}",
                    sent.tokens
                        .iter()
                        .map(|token| paint_string(&brush, &token.lemma, token.tag))
                        .join(" ")
                );
            }
            continue;
        }
        let sent = tokenizer.tokenize_sents(&[sent]).unwrap().remove(0);
        println!(
            "Tokens: {}",
            sent.tokens
                .iter()
                .map(|token| (Symbol::from(token.tag).to_string(), token.tag))
                .map(|(text, tag)| paint_string(&brush, &text, tag))
                .join(" ")
        );
        let specs = sentence_to_specifications(&parser, &sent);

        if specs.is_empty() {
            println!("{}", brush.paint("No specification generated", &light_red));
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
                    println!("{}", brush.paint(format!("FAILURE: {:?}", e), &light_red));
                }
            }
        }
    }
}

fn main() {
    let opts: Opts = Opts::parse();
    let options = ModelOptions::new(opts.model.clone(), opts.url.clone());
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
    //  1. parse sent into tokens (fallible)
    //  2. parse tokens into trees (infallible)
    //  3. parse tree in initial type (infallible)
    //  4. unresolved code blocks (infallible)
    //  5. resolved code items (fallible)
    //  6. final specification (infallible)
    match opts.command {
        SubCommand::EndToEnd(EndToEnd { path }) => {
            match specify_file(path, &options).to_fmt_string() {
                FileOutput::Fmt(output) => println!("{}", output),
                FileOutput::NoFmt(output, _) => println!("{}", output),
            }
        }
        SubCommand::Specify { sub_cmd } => match sub_cmd {
            Specify::Sentence { sentence } => specify_sentences(vec![sentence], &options),
            Specify::File { path, dest } => {
                let dest = match dest {
                    None => {
                        let mut stem = path.file_stem().unwrap().to_os_string();
                        stem.push("_specified.rs");
                        path.with_file_name(stem)
                    }
                    Some(dest) => dest,
                };

                match specify_file(path, &options).to_fmt_string() {
                    FileOutput::Fmt(output) => {
                        std::fs::write(&dest, output).unwrap();
                        println!("Formatted output file written to {}", dest.display())
                    }
                    FileOutput::NoFmt(output, _) => {
                        std::fs::write(&dest, output).unwrap();
                        println!(
                            "Output file written to {} (could not be formatted)",
                            dest.display()
                        )
                    }
                }
            }
            Specify::Docs { path } => {
                let path = path.unwrap_or_else(|| {
                    toolchain_path_to_html_root(&get_toolchain_dirs().unwrap()[0])
                });
                specify_docs(path, &options);
            }
            Specify::Testcases { path } => {
                let testcases = std::fs::read_to_string(path).unwrap();
                specify_sentences(
                    testcases.lines().into_iter().map(String::from).collect(),
                    &options,
                )
            }
            Specify::Repl => repl(&options),
        },
        SubCommand::Render { sub_cmd } => match sub_cmd {
            Render::ParseTree {
                sentence,
                path,
                open_browser,
            } => {
                let cfg = ContextFreeGrammar::from_str(CFG).unwrap();
                let parser = ChartParser::from_grammar(&cfg);
                let tokenizer = Tokenizer::new(options.url.clone(), options.model);
                let sent = tokenizer.tokenize_sents(&[sentence]).unwrap().remove(0);
                let trees = sent.parse_trees(&parser);
                match trees.first() {
                    None => println!("No tree generated for the provided sentence."),
                    Some(tree) => {
                        visualization::tree::render_tree(tree, &path).unwrap();
                        if open_browser {
                            if webbrowser::open(&path).is_err() {
                                println!("Could not open in web browser");
                            }
                        }
                    }
                }
            }
        },
        SubCommand::Parse { path } => {
            use parse_html::parse_file;
            match parse_file(path) {
                Ok(doc_item) => match doc_item {
                    DocItem::Fn(f) => {
                        println!("{}", f.s);
                    }
                    DocItem::Struct(strukt) => {
                        println!("{}", strukt.s);
                        for method in strukt.methods {
                            println!("\n{}", method);
                        }
                    }
                    DocItem::Primitive(primitive) => {
                        println!("{}", primitive.s);
                        for method in primitive.methods {
                            println!("\n{}", method);
                        }
                    }
                },
                Err(_) => println!("Error occured"),
            }
        }
        SubCommand::Index { path } => {
            use parse_json::make_index;

            use roogle_engine::search::Scope;
            let mut own_path = std::env::current_dir().unwrap();
            own_path.push("./index");
            let _ = std::fs::create_dir(&own_path);
            let mut index_dir = own_path.canonicalize().unwrap();
            std::process::Command::new("cargo")
                .arg("+nightly")
                .arg("doc")
                .env("RUSTDOCFLAGS", "-Zunstable-options --output-format=json")
                .env("CARGO_TARGET_DIR", &index_dir)
                .current_dir(path)
                .output();
            index_dir.push("debug");
            let _ = std::fs::remove_dir_all(index_dir);
            let index = make_index().unwrap();
            println!("{}", index.crates.len());
            let (x, query) = roogle_engine::query::parse::parse_query("fn (_) -> _").unwrap();
            println!("{}", x);
            for item in index
                .search(&query, Scope::Crate("aaaa".to_string()), 1.)
                .unwrap()
            {
                println!("{}", item.docs.unwrap_or_default());
                println!("{}", item.name);
            }
        }
        SubCommand::Search => {
            search_demo(&options);
        }
    }
}
