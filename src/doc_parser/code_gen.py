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


if __name__ == '__main__':
    x = CFG.fromstring(Path("./codegrammar_pre.cfg").read_text())
    graph = networkx.DiGraph()

    print("use crate::parse_tree::{SymbolTree, Symbol};\n")
    for lhs, rhsv in x._lhs_index.items():
        rhs = set(r for rhs in rhsv for r in rhs._rhs)
        for r in rhs:
            if isinstance(r, Nonterminal):
                graph.add_edge(str(lhs), str(r).removesuffix("_Q"))
            else:
                graph.add_edge(str(lhs), f"\"{r}\"")

    cycling = set()
    for cycle in networkx.simple_cycles(graph):
        cycling.update(cycle)

    terminals = set()
    for lhs, rhsv in x._lhs_index.items():
        for rhs in rhsv:
            if len(rhs._rhs) == 1 and not isinstance(rhs._rhs[0], Nonterminal):
                terminals.add(lhs)
                terminals.add(str(lhs))
                break


    for lhs, rhsv in x._lhs_index.items():
        if lhs in terminals:
            print(f"pub struct {lhs} {{")
            print(f"    pub word: String,")
            print(f"    pub lemma: String,")
            print(f"}}\n")
            print(f"impl From<Vec<SymbolTree>> for {lhs} {{")
            print(f"    fn from(mut branches: Vec<SymbolTree>) -> Self {{")
            print(f"        let t = branches.remove(0).unwrap_terminal();")
            print(f"        Self {{ word: t.word, lemma: t.lemma }}")
            print(f"    }}")
            print(f"}}\n")
            continue

        e = f"pub enum {lhs} {{\n"
        impl_t = f"impl From<SymbolTree> for {lhs} {{\n"
        impl_t += "    fn from(tree: SymbolTree) -> Self {\n"
        impl_t += "        let (symbol, branches) = tree.unwrap_branch();\n"
        impl_t += "        Self::from(branches)\n"
        impl_t += "    }\n"
        impl_t += "}\n"
        impl_l = f"impl From<Vec<SymbolTree>> for {lhs} {{\n"
        impl_l += "    fn from(branches: Vec<SymbolTree>) -> Self {\n"
        impl_l += "        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());\n"

        variants = []
        counter = 0
        for rhs in rhsv:
            rhsx = [nt_to_rust(r, lhs, terminals) for r in rhs._rhs]
            name = None
            if len(rhs._rhs) == 1:
                name = str(rhs._rhs[0]).capitalize()
            variants.append((name, rhsx))

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
        max_len = max(len(rhs) for _, rhs in variants + [(None, [])])
        if max_len != 0:
            labels = ("labels.next(), " * max_len)[:-2]
            impl_l += f"        match ({labels}) {{\n"

            enum_matches = []
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

                for variant in variant_desugar:
                    impl_l += " " * 12
                    ids = [x.ident(i) for i, x in enumerate(variant)]
                    variant_e = [f"Some((Symbol::{x.value}, {idx}))" for x, idx in zip(variant, ids) if
                                 not isinstance(x, NoneItem)]
                    variant_p = variant_e + ["None"] * (max_len - len(variant_e))
                    impl_l += f"({', '.join(variant_p)}) => {{\n"
                    impl_l += " " * 16
                    variant_c = [x.constructor(idx) for x, idx in zip(variant, ids)]
                    impl_l += f"{lhs}::{name}({', '.join(variant_c)})\n"
                    impl_l += " " * 12
                    impl_l += f"}},\n"

            impl_l += "            _ => panic!(\"Unexpected SymbolTree - have you used the code generation with the last grammar?\"),\n"
            impl_l += "        }\n"
            impl_l += "    }\n"
            impl_l += "}\n"

        for name, rhs in variants:
            e += f"    {name}({', '.join(str(x) for x in rhs)}),\n"
        e += "}\n"

        if variants:
            print(e)
            print(impl_t)
            print(impl_l)
