use std::path::Path;

use pyo3::{Python, PyResult, PyObject, ToPyObject};
use pyo3::types::{IntoPyDict, PyModule};
use pyo3::exceptions::PyStopIteration;

mod search_tree;
mod docs;
mod parse_html;
mod type_match;

use syn::visit_mut::VisitMut;
use syn::{File, parse_file, Attribute, ImplItemMethod, ItemFn};
use syn::parse::{Parse, ParseStream};
use quote::ToTokens;

use docs::Docs;
use search_tree::{SearchTree, SearchItem, Depth};
use type_match::{HasFnArg, FnArgLocation};
use crate::parse_html::{parse_all_files, toolchain_path_to_html_root, get_toolchain_dirs, file_from_root_dir, DocItem, ItemContainer};


#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate syn;

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

    fn specify(&mut self, parser: &Parser, grammar: &Grammar) {
        SpecifierX { searcher: &self.searcher, parser, grammar }.visit_file_mut(&mut self.file)
    }
}

struct SpecifierX<'a, 'p> {
    searcher: &'a SearchTree,
    parser: &'a Parser<'p>,
    grammar: &'a Grammar<'p>
}


fn should_specify<A: AsRef<[Attribute]>>(attrs: A) -> bool {
    attrs.as_ref().iter().map(|attr| &attr.path).any(|x| quote::quote! {#x}.to_string() == "specify")
}

struct AttrHelper {
    attrs: Vec<Attribute>,
}
impl Parse for AttrHelper {
    fn parse(input: ParseStream) -> Result<Self, syn::Error> {
        Ok(AttrHelper {
            attrs: input.call(Attribute::parse_outer)?,
        })
    }
}

impl<'a, 'p> VisitMut for SpecifierX<'a, 'p> {
    fn visit_impl_item_method_mut(&mut self, i: &mut ImplItemMethod) {
        if should_specify(&i.attrs) {
            let docs = Docs::from(&i.attrs);
            if let Some(section) = docs.sections.first() {
                for sentence in &section.sentences {
                    if let Some(Ok(tree)) = self.parser.parse_trees(sentence).next() {
                        match self.grammar.specify_tree(&tree) {
                            Ok(spec) => {
                                let attr = spec.trim_matches('"');
                                // let e: Result<Vec<Attribute>, syn::Error> = Attribute::parse_inner::parse(&tokens).unwrap();
                                let e: Result<AttrHelper, syn::Error> = syn::parse_str(attr);
                                i.attrs.extend(e.unwrap().attrs);
                            }
                            Err(_) => {}
                        }
                    }
                }
            }
        }
    }

    fn visit_item_fn_mut(&mut self, i: &mut ItemFn) {
        if should_specify(&i.attrs) {
            let docs = Docs::from(&i.attrs);
            if let Some(section) = docs.sections.first() {
                for sentence in &section.sentences {
                    if let Some(Ok(tree)) = self.parser.parse_trees(sentence).next() {
                        match self.grammar.specify_tree(&tree) {
                            Ok(spec) => {
                                let attr = spec.trim_matches('"');
                                // let e: Result<Vec<Attribute>, syn::Error> = Attribute::parse_inner::parse(&tokens).unwrap();
                                let e: Result<AttrHelper, syn::Error> = syn::parse_str(attr);
                                i.attrs.extend(e.unwrap().attrs);
                            }
                            Err(_) => {}
                        }
                    }
                }
            }
        }
    }
}

struct Parser<'p> {
    parser: PyObject,
    py: Python<'p>,
}

impl<'p> Parser<'p> {
    fn new(py: Python<'p>) -> Self {
        let path = std::env::current_dir().unwrap();
        let locals = [
            ("sys", py.import("sys").unwrap().to_object(py)),
            ("pathlib", py.import("pathlib").unwrap().to_object(py)),
            ("root_dir", path.to_object(py))
        ].into_py_dict(py);
        // TODO: Make sure we fix path handling for the general case.
        let code = "sys.path.extend([str(pathlib.Path(root_dir).parent), str(pathlib.Path(root_dir).parent.parent)])";
        py.eval(code, None, Some(locals)).unwrap();

        let locals = [("doc_parser", py.import("doc_parser").unwrap())].into_py_dict(py);
        let parser: PyObject = py.eval("doc_parser.Parser.default()", None, Some(locals)).unwrap().extract().unwrap();
        Parser { parser, py }
    }

    fn parse_trees(&self, s: &str) -> TreeIter<'p> {
        let kwargs = [("attach_tags", false)].into_py_dict(self.py);
        TreeIter { trees: self.parser.call_method(self.py, "parse_tree", (s, ), Some(kwargs)).unwrap(), py: self.py }
    }
}

struct TreeIter<'p> {
    trees: PyObject,
    py: Python<'p>,
}

impl<'p> Iterator for TreeIter<'p> {
    type Item = PyResult<PyObject>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.trees.call_method0(self.py, "__next__") {
            Ok(val) => { Some(Ok(val)) }
            Err(e) if e.is_instance::<PyStopIteration>(self.py) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

struct Grammar<'p> {
    grammar: &'p PyModule,
    py: Python<'p>,
}

impl<'p> Grammar <'p> {
    fn new(py: Python<'p>) -> Self {
        Grammar {
            grammar: py.import("grammar").unwrap(),
            py
        }
    }

    fn specify_tree(&self, tree: &PyObject) -> PyResult<String> {
        let locals = [("grammar", &self.grammar.to_object(self.py)), ("val", tree)].into_py_dict(self.py);
        match self.py.eval("grammar.Specification(val, None).as_spec()", None, Some(locals)) {
            Ok(val) => val.extract(),
            Err(e) => Err(e)
        }
    }
}

struct SimMatcher<'p> {
    sim_matcher: PyObject,
    cutoff: f32,
    py: Python<'p>,
}

impl<'p> SimMatcher <'p> {
    fn new(py: Python<'p>, sentence: &str, parser: &Parser<'p>, cutoff: f32) -> Self {
        let locals = [
            ("nlp_query", py.import("nlp_query").unwrap().to_object(py)),
            ("sent", sentence.to_object(py)),
            ("parser", parser.parser.clone()),
            ("cutoff", cutoff.to_object(py)),
        ].into_py_dict(py);
        SimMatcher {
            sim_matcher: py.eval("nlp_query.SimPhrase(sent, parser, -1.)", None, Some(locals)).unwrap().to_object(py),
            cutoff,
            py
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
                None
            )?.extract(self.py)
    }
    fn print_seen(&self) {
        let sents_seen: usize = self.sim_matcher.getattr(self.py, "sents_seen").unwrap().extract(self.py).unwrap();
        println!("{}", sents_seen);
    }

}


fn main() {
    let mut x = Specifier::from_path("../../data/test.rs").unwrap();

    let start = std::time::Instant::now();
    let path = toolchain_path_to_html_root(&get_toolchain_dirs().unwrap()[0]);
    let tree = file_from_root_dir(&path).unwrap();
    let end = std::time::Instant::now();
    println!("Parsing Rust stdlib took {}s", (end - start).as_secs_f32());
    Python::with_gil(|py| -> PyResult<()> {
        let parser = Parser::new(py);
        let grammar = Grammar::new(py);
        let matcher = SimMatcher::new(py, "The minimum of two values", &parser, 0.85);
        let start = std::time::Instant::now();
        let usize_first = HasFnArg { fn_arg_location: FnArgLocation::Output, fn_arg_type: Box::new("f32")};
        println!("{:?}", tree.search(&|item| usize_first.item_matches(item) && match item {
            SearchItem::Fn(docs, _) | SearchItem::Method(docs, _) => {
                docs
                    .sections.first().map(|sect| matcher.any_similar(&sect.sentences).unwrap_or(false)).unwrap_or(false)
            }
            _ => false,
        }, Depth::Infinite));
        let end = std::time::Instant::now();
        println!("Search took {}s", (end - start).as_secs_f32());
        matcher.print_seen();
        // x.specify(&parser, &grammar);
        Ok(())
    }).unwrap();

    let file = &x.file;
    std::fs::write("../../data/test_specified.rs", file.to_token_stream().to_string()).unwrap();
    std::process::Command::new("rustfmt").arg("../../data/test_specified.rs").spawn().unwrap();
}
