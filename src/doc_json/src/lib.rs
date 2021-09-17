use std::path::{Path, PathBuf};
use std::collections::{HashSet};
use std::process::Command;
use std::fmt::{Display, Formatter};

use scraper::{ElementRef, Html, Node, Selector};
use scraper::node::Element;
use selectors::attr::CaseSensitivity;
use home::rustup_home;

mod docs;
mod lib_py;

use docs::{RawDocs, Docs};

#[macro_use]
extern crate lazy_static;

#[derive(Debug)]
pub enum ParseError {
    Io(std::io::Error),
    RustDoc(&'static str),
    RustUp(String),
    Syn(syn::Error),
}

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Io(e) => write!(f, "Io: {}", e),
            ParseError::RustDoc(s) => write!(f, "RustDoc: {}", s),
            ParseError::RustUp(s) => write!(f, "RustUp: {}", s),
            ParseError::Syn(s) => write!(f, "Syn: {}", s),
        }
    }
}

#[derive(Debug)]
pub enum DocItem {
    Fn(DocFn),
    Struct(DocStruct),
    Primitive(DocStruct),
}

impl From<std::io::Error> for ParseError {
    fn from(e: std::io::Error) -> Self {
        ParseError::Io(e)
    }
}

#[derive(Debug)]
pub struct DocFn {
    pub s: String,
}

#[derive(Debug)]
pub struct DocStruct {
    pub s: String,
    pub methods: Vec<String>,
}

lazy_static! {
    static ref H1_FQN: Selector = Selector::parse("h1.fqn").unwrap();
    static ref DETAILS_DIV: Selector = Selector::parse("details.top-doc>div.docblock").unwrap();
    static ref DIV_DOCBLOCK: Selector = Selector::parse("div.docblock").unwrap();
    static ref STRUCT_IMPL_BLOCK: Selector = Selector::parse("details.implementors-toggle>div.impl-items").unwrap();
    static ref METHOD_NAME: Selector = Selector::parse("summary>div>h4").unwrap();
    static ref FN_NAME: Selector = Selector::parse("pre.fn").unwrap();
    static ref FN_DOC: Selector = Selector::parse("details.rustdoc-toggle>div.docblock").unwrap();
    static ref TITLE: Selector = Selector::parse("head>title").unwrap();
    static ref BODY: Selector = Selector::parse("body").unwrap();
    static ref SECTION: Selector = Selector::parse("section").unwrap();
    static ref RUST_DOC_IGNORE_MAP: HashSet<&'static str> = {
        let mut set = HashSet::new();
        set.insert("rust-by-example");
        set.insert("reference");
        set.insert("embedded-book");
        set.insert("edition-guide");
        set.insert("arch");
        set.insert("core_arch");
        set.insert("book");
        set.insert("nomicon");
        set.insert("unstable-book");
        set.insert("cargo");
        set.insert("rustc");
        set.insert("implementors");
        set.insert("rustdoc");
        set.insert("src");
        set
    };
}

fn has_class(e: &Element, class: &str) -> bool {
    e.has_class(class, CaseSensitivity::AsciiCaseInsensitive)
}

fn stringify(e: &ElementRef) -> String {
    let mut s = String::new();
    for item in e.children() {
        match item.value() {
            Node::Text(ref text) => s.push_str(&*text),
            Node::Element(ref child_element) => match child_element.name() {
                "code" => {
                    s.push('`');
                    s.push_str(&stringify(&ElementRef::wrap(item).unwrap()));
                    s.push('`');
                }
                "span" => {
                    if has_class(child_element, "notable-traits")
                        || has_class(child_element, "since")
                        || has_class(child_element, "render-detail")
                        || has_class(child_element, "out-of-band")
                    {
                        continue;
                    }
                    s.push_str(&stringify(&ElementRef::wrap(item).unwrap()));
                }
                "div" => {
                    if has_class(child_element, "code-attribute") || has_class(child_element, "example-wrap") {
                        continue;
                    }

                    s.push_str(&stringify(&ElementRef::wrap(item).unwrap()));
                }
                _ => {
                    s.push_str(&stringify(&ElementRef::wrap(item).unwrap()));
                }
            },
            _ => continue,
        }
    }
    s
}

fn textify(e: &ElementRef) -> String {
    let mut s = String::new();
    for item in e.children() {
        match item.value() {
            Node::Text(ref text) => s.push_str(&*text),
            Node::Element(_) => {
                s.push_str(&textify(&ElementRef::wrap(item).unwrap()));
            }
            _ => continue,
        }
    }
    s
}

fn parse_list(e: &ElementRef) -> Vec<String> {
    e.children()
        .map(ElementRef::wrap)
        .flatten()
        .map(|e| stringify(&e))
        .collect()
}

fn parse_example(_e: &ElementRef) -> String {
    "```CODE```".to_string()
}

fn parse_doc(e: &ElementRef) -> Docs {
    let mut raw_doc = RawDocs::default();

    for item in e.children().map(ElementRef::wrap).flatten() {
        let e = item.value();
        match e.name() {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                raw_doc.push_lines(format!("{} {}", "#", stringify(&item)));
            }
            "p" => {
                raw_doc.push_lines(stringify(&item));
            }
            "div" => {
                if has_class(e, "example-wrap") {
                    raw_doc.push_lines(parse_example(&item));
                } else if has_class(e, "information") {
                    continue;
                } else {
                    continue;
                }
            }
            "ul" => {
                for list_item in parse_list(&item) {
                    raw_doc.push_lines(format!("* {}", list_item));
                }
            }
            "ol" => {
                for (i, list_item) in parse_list(&item).into_iter().enumerate() {
                    raw_doc.push_lines(format!("{}. {}", i, list_item));
                }
            }
            "blockquote" => {
                raw_doc.push_lines(format!("> {}", stringify(&item)));
            }
            "table" => {}
            _ => continue,
        }
    }

    raw_doc.consolidate()
}

impl DocStruct {
    fn from_primitive_block(block: &ElementRef) -> Result<Self, ParseError> {
        let decl_block = block.select(&H1_FQN).next().ok_or_else(|| ParseError::RustDoc("No header"))?;
        let docs = match block.select(&DETAILS_DIV).next() {
            None => RawDocs::default().consolidate(),
            Some(b) => parse_doc(&b),
        };
        let s = stringify(&decl_block);
        let decl_str = match s.rsplit_once(" ") {
            None => return Err(ParseError::RustDoc("no separator in decl")),
            Some((_, rhs)) => rhs,
        };
        let mut strukt = DocStruct {
            s: format!("{}\nstruct {} {{}}", docs, decl_str),
            methods: vec![],
        };
        strukt.eat_impls(&block);
        Ok(strukt)
    }

    fn from_struct_block(block: &ElementRef) -> Result<Self, ParseError> {
        let decl_block = block.select(&DIV_DOCBLOCK).next().ok_or_else(|| ParseError::RustDoc("No header"))?;
        let docs = match block.select(&DETAILS_DIV).next() {
            None => RawDocs::default().consolidate(),
            Some(b) => parse_doc(&b),
        };
        let decl_str = stringify(&decl_block);
        let mut strukt = DocStruct {
            s: format!("{}\n{}", docs, decl_str),
            methods: vec![],
        };
        strukt.eat_impls(&block);
        Ok(strukt)
    }

    fn eat_impls(&mut self, block: &ElementRef) {
        for doc in block.select(&STRUCT_IMPL_BLOCK) {
            let mut doc_children = doc.children().filter_map(ElementRef::wrap).peekable();
            while let Some(item) = doc_children.next() {
                if item.value().name() == "details" {
                    let name = match item.select(&METHOD_NAME).next() {
                        None => continue,
                        Some(name) => name,
                    };

                    let docs = match item.select(&DIV_DOCBLOCK).next() {
                        None => RawDocs::default().consolidate(),
                        Some(e) => parse_doc(&e),
                    };
                    self.methods
                        .push(format!("{}\n{} {{}}", docs, stringify(&name)));
                } else if has_class(item.value(), "method") {} else {}
            }
        }
    }
}

impl DocFn {
    fn from_block(block: &ElementRef) -> Result<Self, ParseError> {
        let decl_block = block.select(&FN_NAME).next().ok_or_else(|| ParseError::RustDoc("Fn has no name"))?;

        let docs = match block.select(&FN_DOC).next() {
            None => RawDocs::default().consolidate(),
            Some(b) => parse_doc(&b),
        };

        Ok(DocFn {
            s: format!("{}\n{} {{}}", docs, textify(&decl_block)),
        })
    }
}

pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<DocItem, ParseError> {
    let s = std::fs::read_to_string(&path)?;
    let document = Html::parse_document(&s);
    let path_type = match path.as_ref().file_name() {
        None => return Err(ParseError::RustDoc("Path has no name")),
        Some(p) => p.to_string_lossy(),
    };

    let lhs = match path_type.split_once(".") {
        None => return Err(ParseError::RustDoc("Path is invalid")),
        Some((lhs, _)) => lhs
    };

    let block = match document.select(&TITLE).next() {
        None => return Err(ParseError::RustDoc("No title found")),
        Some(er) => {
            let text = er.text().collect::<Vec<_>>().join("");
            if text == "Redirection" {
                return Err(ParseError::RustDoc("Redirection"));
            }

            let body = document
                .select(&BODY)
                .next()
                .ok_or_else(|| ParseError::RustDoc("Should have body"))?;

            if has_class(body.value(), "rustdoc") && has_class(body.value(), "source") {
                return Err(ParseError::RustDoc("Source file"));
            }

            let section = body
                .select(&SECTION)
                .next()
                .ok_or_else(|| ParseError::RustDoc("No section"))?;
            section
        }
    };

    match lhs {
        "fn" => DocFn::from_block(&block).map(DocItem::Fn),
        "struct" => DocStruct::from_struct_block(&block).map(DocItem::Struct),
        "primitive" => DocStruct::from_primitive_block(&block).map(DocItem::Primitive),
        _ => Result::Err(ParseError::RustDoc("did not match")),
    }
}

fn find_all_files_in_root_dir_helper<P: AsRef<Path>>(dir: P, paths: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let dir = dir.as_ref();
    match dir.file_name() {
        None => {}
        Some(s) => match s.to_str() {
            None => {}
            Some(s) => if RUST_DOC_IGNORE_MAP.contains(&s) {
                return Ok(());
            }
        }
    }

    let choices = ["struct.", "fn.", "enum.", "primitive.", "constant.", "macro.", "trait.", "keyword."];

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let _ = find_all_files_in_root_dir_helper(path, paths);
        } else if let Some(file_name) = path.file_name() {
            match file_name.to_str() {
                None => continue,
                Some(path_type) => if choices.iter().any(|choice| path_type.starts_with(choice)) {
                    paths.push(path);
                }
            }
        }
    }

    Ok(())
}

pub fn get_toolchain_dirs() -> Result<Vec<PathBuf>, ParseError> {
    let path = rustup_home()?.join("toolchains");

    match String::from_utf8(Command::new("rustup").args(&["toolchain", "list"]).output()?.stdout) {
        Ok(s) => {
            Ok(s
                .lines()
                .map(|line|
                    line.strip_suffix(" (override)")
                        .unwrap_or(line)
                        .strip_suffix(" (default)")
                        .unwrap_or(line)
                        .trim()
                )
                .map(|s| path.join(s))
                .collect())
        }
        Err(e) => {
            Err(ParseError::RustUp(e.to_string()))
        }
    }
}

pub fn toolchain_path_to_html_root<P: AsRef<Path>>(p: P) -> PathBuf {
    p.as_ref().join("share/doc/rust/html/")
}

pub fn find_all_files_in_root_dir<P: AsRef<Path>>(dir: P) -> std::io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    find_all_files_in_root_dir_helper(dir, &mut paths)?;
    Ok(paths)
}
