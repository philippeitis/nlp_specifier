#[derive(Clone, Debug)]
pub enum SymbolTree {
    Terminal(Terminal),
    Branch(Symbol, Vec<SymbolTree>),
}

impl SymbolTree {
    pub(crate) fn from_iter<I: Iterator<Item=Terminal>, N: Eq + Clone + Hash + PartialEq + Into<Symbol>, T>(tree: Tree<N, T>, iter: &mut I) -> Self {
        match tree {
            Tree::Terminal(_) => SymbolTree::Terminal(iter.next().unwrap()),
            Tree::Branch(nt, rest) => {
                let mut sym_trees = Vec::with_capacity(rest.len());
                for item in rest {
                    sym_trees.push(SymbolTree::from_iter(item, iter));
                }
                SymbolTree::Branch(
                    nt.into(),
                    sym_trees
                )
            }
        }
    }
}

impl SymbolTree {
    pub fn unwrap_terminal(self) -> Terminal {
        match self {
            SymbolTree::Terminal(t) => t,
            SymbolTree::Branch(_, _) => panic!("Called unwrap_terminal with non-terminal Tree"),
        }
    }

    pub fn unwrap_branch(self) -> (Symbol, Vec<SymbolTree>) {
        match self {
            SymbolTree::Terminal(_) => panic!("Called unwrap_branch with terminal Tree"),
            SymbolTree::Branch(sym, trees) => (sym, trees),
        }
    }
}

impl ParseNonTerminal for Symbol {
    type Error = ();

    fn parse_nonterminal(s: &str) -> Result<Self, Self::Error> {
        Ok(Self::from(s))
    }
}