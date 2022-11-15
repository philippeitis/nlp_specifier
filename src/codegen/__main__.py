import itertools
import subprocess
from pathlib import Path

import networkx
from nltk import CFG, Nonterminal


class Option:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value.value

    def ident(self, uuid):
        return f"{self.value.lower()}_{uuid}"

    def constructor(self, ident):
        return f"Some({self._value.constructor(ident)})"

    def __str__(self):
        return f"Option<{self._value}>"


class Box:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value.value

    def ident(self, uuid):
        return f"{self.value.lower()}_{uuid}"

    def constructor(self, ident):
        return f"Box::new({self._value.constructor(ident)})"

    def __str__(self):
        return f"Box<{self._value}>"


class Literal:
    def __init__(self, value):
        self.value = value

    def ident(self, uuid):
        return f"{self.value.lower()}_{uuid}"

    def constructor(self, ident):
        return f"{self.value}::from({ident})"

    def __str__(self):
        return str(self.value)


class NoneItem:
    @property
    def value(self):
        return "NO_VALUE_AT_ALL"

    def ident(self, uuid):
        return f"SHOULD_NOT_APPEAR_IN_YOUR_CODE_{uuid}"

    def constructor(self, ident):
        return f"None"


class Terminal:
    def __init__(self, value):
        self.value = value

    def ident(self, uuid):
        return f"{self.value.lower()}_{uuid}"

    def constructor(self, ident):
        return f"{self.value}::from({ident})"

    def __str__(self):
        return self.value


def nt_to_rust(nt: Nonterminal, lhs, terminals):
    if nt in terminals:
        root = Terminal
    else:
        root = Literal
    nt = str(nt)

    if nt.endswith("_Q"):
        nt = nt.removesuffix('_Q')
        if nt in terminals:
            root = Terminal

        if nt == str(lhs):
            return Option(Box(root(nt)))
        return Option(root(nt))
    if nt == str(lhs):
        return Box(root(nt))
    return root(nt)


def read_terminals(path: Path):
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        vals = line.split(" ", 2)
        if len(vals) == 1:
            lines.append((vals[0], vals[0]))
        else:
            lines.append((vals[0], vals[1]))
    return lines


class Variants:
    def __init__(self, rhsv):
        # Generate all possible variants
        variants = []
        counter = 0

        for rhs in rhsv:
            rhsx = [nt_to_rust(r, lhs, terminal_set) for r in rhs._rhs]
            name = None
            if len(rhs._rhs) == 1:
                name = str(rhs._rhs[0]).capitalize()
            variants.append((name, rhsx))

        counter = 0
        names = set(name for name, _ in variants)
        for i, (name, rhs) in enumerate(variants):
            if name is not None:
                continue
            if len([r for r in rhs if isinstance(r, Option)]) == len(rhs) - 1:
                name = [r for r in rhs if not isinstance(r, Option)][0].value
                if name in names:
                    name = None
            if name is None:
                name = f"_{counter}"
                counter += 1
            variants[i] = (name, rhs)

        # Generate all possible variants
        variants_desugared = []

        for name, variant in variants:
            num_opts = len([v for v in variant if isinstance(v, Option)])
            if num_opts == 0:
                variant_desugar = [variant]
            else:
                variant_desugar = []
                for i in range(0, 2 ** num_opts):
                    variant_desugar.append([])
                    for item in variant:
                        if isinstance(item, Option):
                            if i & 1:
                                variant_desugar[-1].append(item)
                            else:
                                variant_desugar[-1].append(NoneItem())
                            i >>= 1
                        else:
                            variant_desugar[-1].append(item)
            variants_desugared += [(name, variant) for variant in variant_desugar]
        self.max_rhs = max(len(rhs) for _, rhs in variants)
        self.variants = variants
        self.desugared_variants = variants_desugared


class RustEnum:
    def __init__(self, name, vis, variants, attributes):
        self.name = name
        self.vis = vis
        self.variants = variants
        self.attributes = attributes

    def __str__(self):
        attrs = "\n".join(f"#[{attr}]" for attr in self.attributes)
        variants = "\n".join(f"    {variant}," for variant in self.variants)
        return f"{attrs}\n{self.vis} enum {self.name} {{\n{variants}\n}}"


def rust_impl_lhs(lhs, variants: Variants):
    impl_t = f"impl From<SymbolTree> for {lhs} {{\n"
    impl_t += "    fn from(tree: SymbolTree) -> Self {\n"
    impl_t += "        let (_symbol, branches) = tree.unwrap_branch();\n"
    impl_t += "        Self::from(branches)\n"
    impl_t += "    }\n"
    impl_t += "}\n"
    impl_l = f"impl From<Vec<SymbolTree>> for {lhs} {{\n"
    impl_l += "    fn from(branches: Vec<SymbolTree>) -> Self {\n"
    impl_l += "        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());\n"

    # Generate From<> impl

    labels = ("labels.next(), " * variants.max_rhs)[:-2]
    impl_l += f"        match ({labels}) {{\n"

    for name, variant in variants.desugared_variants:
        impl_l += " " * 12
        ids = [x.ident(i) for i, x in enumerate(variant)]
        variant_e = [f"Some((Symbol::{x.value}, {idx}))" for x, idx in zip(variant, ids) if
                     not isinstance(x, NoneItem)]
        variant_p = variant_e + ["None"] * (variants.max_rhs - len(variant_e))
        impl_l += f"({', '.join(variant_p)}) => {{\n"
        impl_l += " " * 16
        variant_c = [x.constructor(idx) for x, idx in zip(variant, ids)]
        impl_l += f"{lhs}::{name}({', '.join(variant_c)})\n"
        impl_l += " " * 12
        impl_l += f"}},\n"

    impl_l += "            _ => panic!(\"Unexpected SymbolTree - have you used the code generation with the latest grammar?\"),\n"
    impl_l += "        }\n"
    impl_l += "    }\n"
    impl_l += "}\n"

    # Generate enum
    enum_variants = [f"{name}({', '.join(str(x) for x in rhs)})"
                     for name, rhs in variants.variants]

    return impl_t, impl_l, RustEnum(lhs, "pub", enum_variants, ["derive(Clone)"])


if __name__ == '__main__':
    PWD = Path(__file__).parent
    non_terminals = CFG.fromstring((PWD / Path("nonterminals.cfg")).read_text())
    terminals = read_terminals(PWD / Path("terminals.cfg"))
    terminal_set = {Nonterminal(v) for v, _ in terminals}
    graph = networkx.DiGraph()

    for lhs, rhsv in non_terminals._lhs_index.items():
        rhs = set(r for rhs in rhsv for r in rhs._rhs)
        for r in rhs:
            if isinstance(r, Nonterminal):
                graph.add_edge(str(lhs), str(r).removesuffix("_Q"))
            else:
                graph.add_edge(str(lhs), f"\"{r}\"")

    cycling = set()
    for cycle in networkx.simple_cycles(graph):
        cycling.update(cycle)

    cfg = "# This file is automatically generated by running code_gen.py\n\n"
    tree_rs = "#![allow(non_camel_case_types)]\n"
    tree_rs += "use std::hash::Hash;\n\n"
    tree_rs += "use chartparse::grammar::ParseTerminal;\n\n"
    tree_rs += "use crate::parse_tree::{SymbolTree, Symbol, Terminal};\n\n"

    for lhs, rhsv in non_terminals._lhs_index.items():
        variants = Variants(rhsv)
        impl_t, impl_l, e = rust_impl_lhs(lhs, variants)
        tree_rs += e + "\n"
        tree_rs += impl_t + "\n"
        tree_rs += impl_l + "\n"

        for _, variant in variants.desugared_variants:
            rhs = " ".join(v.value for v in variant if not isinstance(v, NoneItem))
            cfg += f"{lhs} -> {rhs}\n"
        cfg += "\n"

    cfg += "# Terminals\n\n"

    for term, sym in terminals:
        tree_rs += f"#[derive(Clone)]\n"
        tree_rs += f"pub struct {term} {{\n"
        tree_rs += f"    pub word: String,\n"
        tree_rs += f"    pub lemma: String,\n"
        tree_rs += f"}}\n\n"
        tree_rs += f"impl From<Vec<SymbolTree>> for {term} {{\n"
        tree_rs += f"    fn from(mut branches: Vec<SymbolTree>) -> Self {{\n"
        tree_rs += f"        let t = branches.remove(0).unwrap_terminal();"
        tree_rs += f"        Self {{ word: t.word, lemma: t.lemma }}\n"
        tree_rs += f"    }}\n"
        tree_rs += f"}}\n\n"
        tree_rs += f"impl From<{term}> for Terminal {{\n"
        tree_rs += f"    fn from(val: {term}) -> Self {{\n"
        tree_rs += f"        Self {{ word: val.word, lemma: val.lemma }}\n"
        tree_rs += f"    }}\n"
        tree_rs += f"}}\n\n"
        cfg += f"{term} -> \"{term}\"\n"

    terminal = "#[derive(Copy, Clone, Eq, PartialEq, Hash)]\npub enum TerminalSymbol {\n"

    # phf stub - worth considering in future
    # phf_terminal = "static TERMINALSYMBOLS: phf::Map<&'static str, TerminalSymbol> = phf_map! {\n"

    terminal_from = "impl ParseTerminal for TerminalSymbol {\n"
    terminal_from += "    type Error = String;"
    terminal_from += "    fn parse_terminal(s: &str) -> Result<Self, Self::Error> {\n"
    terminal_from += "        match s {\n"
    for term, sym in terminals:
        terminal += f"    {term},\n"

        terminal_from += " " * 12
        terminal_from += f"\"{sym}\" => Ok(TerminalSymbol::{term}),\n"
        # phf_terminal += f"    \"{sym}\" => TerminalSymbol::{term},\n"
        if sym != term:
            terminal_from += " " * 12
            terminal_from += f"\"{term}\" => Ok(TerminalSymbol::{term}),\n"
            # phf_terminal += f"    \"{term}\" => TerminalSymbol::{term},\n"
    # phf_terminal += "};\n"

    terminal += "}\n"
    terminal_from += " " * 12
    terminal_from += f"x => Err(format!(\"Terminal {{}} is not supported.\", x)),\n"
    terminal_from += "        }\n"
    terminal_from += "    }\n"
    terminal_from += "}\n"
    #     terminal_from = """impl TerminalSymbol {
    #     pub fn from_terminal<S: AsRef<str>>(s: S) -> Result<Self, String> {
    #         TERMINALSYMBOLS.get(s.as_ref()).cloned().ok_or(format!("Terminal {} is not supported.", s.as_ref()))
    #     }
    # }"""
    tree_rs += terminal + "\n"
    # tree_rs += phf_terminal + "\n"
    tree_rs += terminal_from + "\n"
    eir_rs = """#![allow(non_camel_case_types)]
use std::hash::Hash;
use std::fmt::Formatter;

use chartparse::Tree;
use chartparse::grammar::{ParseNonTerminal, ParseTerminal};

use crate::parse_tree::Terminal;
use crate::parse_tree::tree::TerminalSymbol;

#[derive(Hash, Copy, Clone, Debug, Eq, PartialEq)]
pub enum Symbol {
"""
    for lhs, _ in non_terminals._lhs_index.items():
        eir_rs += f"   {lhs},\n"
    for term, _ in terminals:
        eir_rs += f"    {term},\n"
    eir_rs += "}\n\n"
    eir_rs += """impl From<TerminalSymbol> for Symbol {
    fn from(t: TerminalSymbol) -> Symbol {
        match t {\n"""
    for term, _ in terminals:
        eir_rs += " " * 12
        eir_rs += f"TerminalSymbol::{term} => Symbol::{term},\n"
    eir_rs += "        }\n"
    eir_rs += "    }\n"
    eir_rs += "}\n\n"

    eir_rs += """impl<S: AsRef<str>> From<S> for Symbol {
    fn from(nt: S) -> Self {
        if let Ok(termsym) = TerminalSymbol::parse_terminal(nt.as_ref()) {
            return termsym.into();
        }

        match nt.as_ref() {
"""
    for lhs, _ in non_terminals._lhs_index.items():
        eir_rs += " " * 12
        eir_rs += f"\"{lhs}\" => Symbol::{lhs},\n"
    eir_rs += " " * 12
    eir_rs += "x => panic!(\"Unexpected symbol {}\", x),\n"

    eir_rs += "        }\n"
    eir_rs += "    }\n"
    eir_rs += "}\n\n"

    eir_rs += """impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
"""
    for lhs, _ in itertools.chain(non_terminals._lhs_index.items(), terminals):
        eir_rs += " " * 12
        eir_rs += f"Symbol::{lhs} => \"{lhs}\",\n"
    eir_rs += "        })\n"
    eir_rs += "    }\n"
    eir_rs += "}\n\n"

    eir_rs += (PWD / Path("./symbol_tree.rs")).read_text()

    (Path(__file__).parent / Path("../doc_parser/codegrammar.cfg")).write_text(cfg)
    tree_rs_path = (Path(__file__).parent / Path("../doc_parser/src/parse_tree/tree.rs"))
    tree_rs_path.write_text(tree_rs)
    subprocess.Popen(["rustfmt", tree_rs_path])
    eir_rs_path = (Path(__file__).parent / Path("../doc_parser/src/parse_tree/eir.rs"))
    eir_rs_path.write_text(eir_rs)
    subprocess.Popen(["rustfmt", eir_rs_path])
