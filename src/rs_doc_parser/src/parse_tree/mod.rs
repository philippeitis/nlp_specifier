use chartparse::ContextFreeGrammar;

mod eir;
pub mod tree;

pub use eir::{SymbolTree, Symbol};

/// Generates the Symbol enum, and adds conversion methods for the chartparse library's structs.
pub fn gen_cfg_enum() {
    let cfg = ContextFreeGrammar::fromstring(std::fs::read_to_string("../doc_parser/codegrammar.cfg").unwrap()).unwrap();
    let mut str_enum = String::new();
    str_enum.push_str("#[derive(Copy, Clone, Debug, Eq, PartialEq)]\n");
    str_enum.push_str("pub enum Symbol {\n");
    let mut symbols: Vec<_> = cfg.lhs_index.keys().map(|x| &x.symbol).collect();
    symbols.sort();

    for lhs in symbols.iter().cloned() {
        str_enum.push_str("    ");
        str_enum.push_str(lhs);
        str_enum.push_str(",\n");
    }

    str_enum.push_str("}\n");
    println!("{}", str_enum);

    let mut from_str_impl = String::new();
    from_str_impl.push_str("impl From<NonTerminal> for Symbol {\n");
    from_str_impl.push_str("    fn from(nt: NonTerminal) -> Self {\n");
    from_str_impl.push_str("        match nt.symbol.as_str() {\n");

    for lhs in symbols.iter().cloned() {
        from_str_impl.push_str(&format!("            \"{}\" => Symbol::{},\n", lhs, lhs));
    }
    from_str_impl.push_str("            x => panic!(\"Unexpected symbol {}\", x),\n");
    from_str_impl.push_str("        }\n");
    from_str_impl.push_str("    }\n");
    from_str_impl.push_str("}\n");
    println!("{}", from_str_impl);
}

#[derive(Debug, Clone)]
pub struct Terminal {
    pub word: String,
    pub lemma: String,
}