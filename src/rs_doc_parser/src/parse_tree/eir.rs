use chartparse::TreeWrapper;
use chartparse::production::{NonTerminal, Terminal as CTerminal};
use chartparse::tree::TreeNode;
use crate::parse_tree::Terminal;


#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Symbol {
    ARITH,
    ARITHOP,
    ASSERT,
    ASSIGN,
    BITOP,
    BOOL_EXPR,
    CC,
    CD,
    CHAR,
    CODE,
    COMMA,
    COND,
    DOT,
    DT,
    EQTO,
    EVENT,
    EXCL,
    FOR,
    HASSERT,
    HYPH,
    IF,
    IFF,
    IN,
    JJ,
    JJR,
    JJS,
    LIT,
    LRB,
    MD,
    MJJ,
    MNN,
    MREL,
    MRET,
    MVB,
    NN,
    NNP,
    NNPS,
    NNS,
    OBJ,
    OBJV,
    OP,
    PROP,
    PROP_OF,
    PRP,
    QASSERT,
    QUANT,
    QUANT_EXPR,
    RANGE,
    RANGEMOD,
    RB,
    REL,
    RET,
    RETIF,
    RRB,
    RSEP,
    S,
    SHIFT,
    SHIFTOP,
    SIDE,
    STR,
    TJJ,
    TO,
    VB,
    VBD,
    VBG,
    VBN,
    VBP,
    VBZ,
    WDT,
    X,
}

impl From<NonTerminal> for Symbol {
    fn from(nt: NonTerminal) -> Self {
        match nt.symbol.as_str() {
            "ARITH" => Symbol::ARITH,
            "ARITHOP" => Symbol::ARITHOP,
            "ASSERT" => Symbol::ASSERT,
            "ASSIGN" => Symbol::ASSIGN,
            "BITOP" => Symbol::BITOP,
            "BOOL_EXPR" => Symbol::BOOL_EXPR,
            "CC" => Symbol::CC,
            "CD" => Symbol::CD,
            "CHAR" => Symbol::CHAR,
            "CODE" => Symbol::CODE,
            "COMMA" => Symbol::COMMA,
            "COND" => Symbol::COND,
            "DOT" => Symbol::DOT,
            "DT" => Symbol::DT,
            "EQTO" => Symbol::EQTO,
            "EVENT" => Symbol::EVENT,
            "EXCL" => Symbol::EXCL,
            "FOR" => Symbol::FOR,
            "HASSERT" => Symbol::HASSERT,
            "HYPH" => Symbol::HYPH,
            "IF" => Symbol::IF,
            "IFF" => Symbol::IFF,
            "IN" => Symbol::IN,
            "JJ" => Symbol::JJ,
            "JJR" => Symbol::JJR,
            "JJS" => Symbol::JJS,
            "LIT" => Symbol::LIT,
            "LRB" => Symbol::LRB,
            "MD" => Symbol::MD,
            "MJJ" => Symbol::MJJ,
            "MNN" => Symbol::MNN,
            "MREL" => Symbol::MREL,
            "MRET" => Symbol::MRET,
            "MVB" => Symbol::MVB,
            "NN" => Symbol::NN,
            "NNP" => Symbol::NNP,
            "NNPS" => Symbol::NNPS,
            "NNS" => Symbol::NNS,
            "OBJ" => Symbol::OBJ,
            "OBJV" => Symbol::OBJV,
            "OP" => Symbol::OP,
            "PROP" => Symbol::PROP,
            "PROP_OF" => Symbol::PROP_OF,
            "PRP" => Symbol::PRP,
            "QASSERT" => Symbol::QASSERT,
            "QUANT" => Symbol::QUANT,
            "QUANT_EXPR" => Symbol::QUANT_EXPR,
            "RANGE" => Symbol::RANGE,
            "RANGEMOD" => Symbol::RANGEMOD,
            "RB" => Symbol::RB,
            "REL" => Symbol::REL,
            "RET" => Symbol::RET,
            "RETIF" => Symbol::RETIF,
            "RRB" => Symbol::RRB,
            "RSEP" => Symbol::RSEP,
            "S" => Symbol::S,
            "SHIFT" => Symbol::SHIFT,
            "SHIFTOP" => Symbol::SHIFTOP,
            "SIDE" => Symbol::SIDE,
            "STR" => Symbol::STR,
            "TJJ" => Symbol::TJJ,
            "TO" => Symbol::TO,
            "VB" => Symbol::VB,
            "VBD" => Symbol::VBD,
            "VBG" => Symbol::VBG,
            "VBN" => Symbol::VBN,
            "VBP" => Symbol::VBP,
            "VBZ" => Symbol::VBZ,
            "WDT" => Symbol::WDT,
            "X" => Symbol::X,
            x => panic!("Unexpected symbol {}", x),
        }
    }
}

#[derive(Clone, Debug)]
pub enum SymbolTree {
    Terminal(Terminal),
    Branch(Symbol, Vec<SymbolTree>),
}

impl SymbolTree {
    pub(crate) fn from_iter<I: Iterator<Item=Terminal>>(tree: TreeWrapper, iter: &mut I) -> Self {
        match tree.inner {
            TreeNode::Terminal(t) => SymbolTree::Terminal(iter.next().unwrap()),
            TreeNode::Branch(nt, rest) => {
                let mut sym_trees = Vec::with_capacity(rest.len());
                for item in rest {
                    sym_trees.push(SymbolTree::from_iter(item, iter));
                }
                SymbolTree::Branch(
                    Symbol::from(nt),
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