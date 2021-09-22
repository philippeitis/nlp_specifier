use std::fmt::{Display, Formatter};
use std::str::FromStr;

use crate::parse_tree::{SymbolTree, Symbol};
use crate::parse_tree::tree::{S, MRET, OBJ, OP, BITOP, ARITHOP, SHIFTOP, LIT, CODE};


pub struct Code {
    pub(crate) code: String,
}

impl Code {
    fn new(s: &str) -> Self {
        Code {
            code: s.to_string()
        }
    }
}

impl From<CODE> for Code {
    fn from(c: CODE) -> Self {
        Code {
            code: c.word
        }
    }
}

pub struct Literal {
    pub(crate) s: String,
}

impl From<LIT> for Literal {
    fn from(l: LIT) -> Self {
        Literal {
            s: l.word
        }
    }
}

impl Literal {
    fn new(s: &str) -> Self {
        Literal {
            s: s.to_string()
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    SubFrom,
    Div,
    Mul,
    Rem,
    BitXor,
    And,
    BitAnd,
    Or,
    BitOr,
    Shr,
    Shl,
}

impl BinOp {
    fn apply_jj(self, jj: &str) -> BinOp {
        match (self, jj.to_lowercase().as_str()) {
            (BinOp::And, "logical") => BinOp::And,
            (BinOp::And, "boolean") => BinOp::BitAnd,
            (BinOp::And, "bitwise") => BinOp::BitAnd,
            (BinOp::Or, "logical") => BinOp::Or,
            (BinOp::Or, "boolean") => BinOp::BitOr,
            (BinOp::Or, "bitwise") => BinOp::BitOr,
            _ => self
        }
    }

    fn shift_with_dir(dir: &str) -> Option<Self> {
        match dir.to_lowercase().as_str() {
            "right" => Some(BinOp::Shr),
            "left" => Some(BinOp::Shl),
            _ => None,
        }
    }

    /// Should
    fn from_tree(tree: SymbolTree) -> Self {
        OP::from(tree).into()
    }
}

impl From<OP> for BinOp {
    fn from(op: OP) -> Self {
        match op {
            OP::Bitop(bitop) => match bitop {
                BITOP::_0(jj, cc) => {
                    BinOp::from_str(&cc.lemma).unwrap().apply_jj(&jj.lemma)
                }
                BITOP::_1(nn, cc) => {
                    BinOp::from_str(&cc.lemma).unwrap().apply_jj(&nn.lemma)
                }
            },
            OP::Arithop(a) => match a {
                ARITHOP::ARITH(arith, Some(inx)) => if inx.lemma == "from" {
                    let op = BinOp::from_str(&arith.lemma).unwrap();
                    if op == BinOp::Sub {
                        BinOp::SubFrom
                    } else {
                        op
                    }
                } else {
                    BinOp::from_str(&arith.lemma).unwrap()
                }
                ARITHOP::ARITH(arith, None) => {
                    BinOp::from_str(&arith.lemma).unwrap()
                }
            }
            OP::Shiftop(shift) => match shift {
                SHIFTOP::_0(_, _, _, nn, _) => {
                        BinOp::shift_with_dir(&nn.lemma).unwrap()
                }
                SHIFTOP::_1(jj, _) => {
                    BinOp::shift_with_dir(&jj.lemma).unwrap()
                }
            }
        }
    }
}

impl FromStr for BinOp {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "add" => BinOp::Add,
            "added" => BinOp::Add,
            "increment" => BinOp::Add,
            "incremented" => BinOp::Add,
            "plus" => BinOp::Add,
            "sub" => BinOp::Sub,
            "decremented" => BinOp::Sub,
            "decrement" => BinOp::Sub,
            "subtract" => BinOp::Sub,
            "subtracted" => BinOp::Sub,
            "div" => BinOp::Div,
            "divide" => BinOp::Div,
            "divided" => BinOp::Div,
            "mul" => BinOp::Mul,
            "multiply" => BinOp::Mul,
            "multiplied" => BinOp::Mul,
            "rem" => BinOp::Rem,
            "remainder" => BinOp::Rem,
            "xor" => BinOp::BitXor,
            "and" => BinOp::And,
            "or" => BinOp::Or,
            _ => return Err(())
        })
    }
}

impl Display for BinOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::SubFrom => "-",
            BinOp::Div => "/",
            BinOp::Mul => "*",
            BinOp::Rem => "%",
            BinOp::BitXor => "^",
            BinOp::And => "&&",
            BinOp::BitAnd => "&",
            BinOp::Or => "||",
            BinOp::Shr => ">>",
            BinOp::Shl => "<<",
            BinOp::BitOr => "|",
        })
    }
}

pub struct Op {
    pub(crate) lhs: Box<Object>,
    pub(crate) op: BinOp,
    pub(crate) rhs: Box<Object>,
}

impl Op {
    pub fn new(mut lhs: Object, mut op: BinOp, mut rhs: Object) -> Self {
        if op == BinOp::SubFrom {
            std::mem::swap(&mut lhs, &mut rhs);
            op = BinOp::Sub;
        }

        Self {
            lhs: Box::new(lhs),
            op,
            rhs: Box::new(rhs),
        }
    }
}

pub enum Object {
    Code(Code),
    Lit(Literal),
    Op(Op),
    PropOf(PropertyOf),
}

impl From<OBJ> for Object {
    fn from(obj: OBJ) -> Self {
        match obj {
            OBJ::Code(code) => Object::Code(code.into()),
            OBJ::LIT(_, lit) => Object::Lit(lit.into()),
            OBJ::_0(lhs, op, rhs) => {
                Object::Op(Op::new(lhs.into(), op.into(), rhs.into()))
            }
            _ => unimplemented!(),
        }
    }
}

impl From<Box<OBJ>> for Object {
    fn from(obj: Box<OBJ>) -> Self {
        Self::from(*obj)
    }
}

impl Object {
    /// Tree with OBJ root
    pub(crate) fn from_tree(tree: SymbolTree) -> Self {
        OBJ::from(tree).into()
    }

    /// Tree without OBJ root
    pub(crate) fn from_symbol_trees(branches: Vec<SymbolTree>) -> Self {
        OBJ::from(branches).into()
    }
}

enum Property {
    Mnn(MNN),
    Mjj(MJJ),
}

pub struct PropertyOf {
    prop: Property,
    object: Box<Object>,
}

impl PropertyOf {}

enum Comparator {
    Lt,
    Gt,
    Lte,
    Gte,
    Eq,
    Neq,
}

impl FromStr for Comparator {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "less" => Comparator::Lt,
            "smaller" => Comparator::Lt,
            "greater" => Comparator::Gt,
            "larger" => Comparator::Gt,
            "equal" => Comparator::Eq,
            "unequal" => Comparator::Neq,
            "same" => Comparator::Eq,
            _ => return Err(()),
        })
    }
}

impl Comparator {
    fn apply_eq(self) -> Self {
        match self {
            Comparator::Lt => Comparator::Lte,
            Comparator::Gt => Comparator::Gte,
            x => x
        }
    }

    fn negate(self) -> Self {
        match self {
            Comparator::Lt => Comparator::Gte,
            Comparator::Gt => Comparator::Lte,
            Comparator::Lte => Comparator::Gt,
            Comparator::Gte => Comparator::Lt,
            Comparator::Eq => Comparator::Neq,
            Comparator::Neq => Comparator::Eq,
        }
    }
}

enum IfExpr {
    If,
    Iff,
}

struct BoolCond {
    if_expr: IfExpr,

}

pub enum Event {
    Overflow,
    Panic,
    NoOverflow,
    NoPanic,
    Other(String, bool),
}

impl Event {
    fn new(nn: SymbolTree, vb: SymbolTree) {}
}


pub enum Specification {
    RetIf,
    HAssert,
    QAssert,
    Mret(MReturn),
    Side,
    FnCall,
}

impl From<S> for Specification {
    fn from(spec_tree: S) -> Self {
        match spec_tree {
            S::Mret(m) => Specification::Mret(m.into()),
            _ => unimplemented!(),
        }
    }
}

pub struct MReturn {
    pub(crate) ret_val: Object,
}

impl From<MRET> for MReturn {
    fn from(mret: MRET) -> Self {
        match mret {
            MRET::_0(_, ret_val) => {
                MReturn { ret_val: ret_val.into() }
            }
            MRET::_1(ret_val, vbz, _) => {
                assert_eq!(vbz.lemma, "is");
                MReturn { ret_val: ret_val.into() }
            }
            MRET::_2(ret_val, _) => {
                MReturn { ret_val: ret_val.into() }
            }
        }
    }
}

impl MReturn {
    pub(crate) fn from_symbol_trees(branches: Vec<SymbolTree>) -> Self {
        MRET::from(branches).into()
    }
}

enum UnaryOp {
    Negate,
}

impl FromStr for UnaryOp {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "negate" => UnaryOp::Negate,
            "negated" => UnaryOp::Negate,
            _ => return Err(()),
        })
    }
}

struct QuantAssert {
    quant_expr: QuantExpr,
    assertion: bool,
}

struct HardAssert {
    md: Md,
    assert: Assert,
}

struct Md {
    word: String,
}

struct QuantExpr {
    quant: Quantifier,
    range: RangeMod,
    range_conds: Vec<(CC, ModRelation)>,
}

enum CC {
    And,
    Or,
}

impl FromStr for CC {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "or" => CC::Or,
            "and" => CC::And,
            _ => return Err(()),
        })
    }
}

struct ModRelation {}

struct Relation {
    objs: Vec<(CC, Object)>,
    op: Comparator,
    negated: bool,
}

enum UpperBound {
    Inclusive,
    Exclusive,
}

impl FromStr for UpperBound {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "inclusive" => UpperBound::Inclusive,
            "exclusive" => UpperBound::Exclusive,
            _ => return Err(()),
        })
    }
}

impl UpperBound {
    fn lt(self) -> Comparator {
        match self {
            UpperBound::Inclusive => Comparator::Lte,
            UpperBound::Exclusive => Comparator::Lt,
        }
    }
}

struct Range {
    ident: Option<Object>,
    start: Option<Object>,
    end: Option<Object>,
}

impl Range {
    pub(crate) fn from_symbol_trees(branches: Vec<SymbolTree>) -> Self {
        let mut labels = branches.into_iter().map(|x| x.unwrap_branch());
        println!("MReturn");
        match (labels.next(), labels.next(), labels.next(), labels.next(), labels.next()) {
            (
                Some((Symbol::OBJ, mut ident)),
                Some((Symbol::IN, _)),
                Some((Symbol::OBJ, mut start)),
                Some((Symbol::RSEP, _)),
                Some((Symbol::OBJ, mut end)),
            ) => {
                Self {
                    ident: Some(Object::from_symbol_trees(ident)),
                    start: Some(Object::from_symbol_trees(start)),
                    end: Some(Object::from_symbol_trees(end)),
                }
            }
            (
                Some((Symbol::IN, _)),
                Some((Symbol::OBJ, mut start)),
                Some((Symbol::RSEP, _)),
                Some((Symbol::OBJ, mut end)),
                None,
            ) => {
                Self {
                    ident: None,
                    start: Some(Object::from_symbol_trees(start)),
                    end: Some(Object::from_symbol_trees(end)),
                }
            }
            (
                Some((Symbol::IN, _)),
                Some((Symbol::IN, _)),
                Some((Symbol::OBJ, mut end)),
                None,
                None,
            ) => {
                Self {
                    ident: None,
                    start: None,
                    end: Some(Object::from_symbol_trees(end)),
                }
            }
            _ => unimplemented!()
        }
    }
}

struct RangeMod {
    range: Range,
    upper_bound: UpperBound,
}

impl RangeMod {
    pub(crate) fn from_symbol_trees(mut branches: Vec<SymbolTree>) -> Self {
        let range = Range::from_symbol_trees(branches.remove(0).unwrap_branch().1);
        let upper_bound = match branches.pop() {
            None => UpperBound::Exclusive,
            Some(t) => {
                let (sym, mut branches) = t.unwrap_branch();
                if sym == Symbol::JJ {
                    UpperBound::from_str(&branches.remove(0).unwrap_terminal().word).unwrap()
                } else {
                    UpperBound::Exclusive
                }
            }
        };
        RangeMod {
            range,
            upper_bound,
        }
    }
}

struct Quantifier {
    universal: bool,
    obj: Object,
}

impl Quantifier {
    pub(crate) fn from_symbol_trees(mut branches: Vec<SymbolTree>) -> Self {
        let obj = Object::from_tree(branches.remove(0));
        let universal = match branches.remove(0).unwrap_terminal().word.as_str() {
            "all" | "each" | "any" => true,
            _ => false,
        };

        Quantifier {
            obj,
            universal,
        }
    }
}

pub enum Negatable {
    Assert(Assert),
    Code(Code),
    Event(Event),
}

pub struct Negated {
    iff: IfExpr,
    pub expr: Negatable,
}

struct MNN {}

struct MJJ {}

pub struct Assert {}