use std::fmt::{Display, Formatter};
use std::str::FromStr;

use crate::parse_tree::tree::{
    ARITHOP, ASSERT, ASSIGN, BITOP, BOOL_EXPR, CODE, COND, EVENT, HASSERT, HQASSERT, JJ, LIT, MD,
    MJJ, MNN, MREL, MRET, MVB, OBJ, OP, PROP, PROP_OF, PRP, QASSERT, QUANT, QUANT_EXPR, RANGE,
    RANGEMOD, RB, REL, RETIF, RETIF_, RETIF_TERM, S, SHIFTOP, SPEC_ATOM, SPEC_CHAIN,
    SPEC_CHAIN_PRE, SPEC_COND, SPEC_ITEM, SPEC_TERM, TJJ, VBD, VBG, VBN, VBZ,
};
use crate::parse_tree::Terminal;

#[derive(Clone)]
pub struct Code {
    pub(crate) code: String,
}

impl From<CODE> for Code {
    fn from(c: CODE) -> Self {
        Code { code: c.word }
    }
}

#[derive(Clone)]
pub struct Literal {
    pub(crate) s: String,
}

impl From<LIT> for Literal {
    fn from(l: LIT) -> Self {
        Literal { s: l.word }
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
            _ => self,
        }
    }

    fn shift_with_dir(dir: &str) -> Option<Self> {
        match dir.to_lowercase().as_str() {
            "right" => Some(BinOp::Shr),
            "left" => Some(BinOp::Shl),
            _ => None,
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
            "+" => BinOp::Add,
            "sub" => BinOp::Sub,
            "decremented" => BinOp::Sub,
            "decrement" => BinOp::Sub,
            "subtract" => BinOp::Sub,
            "subtracted" => BinOp::Sub,
            "-" => BinOp::Sub,
            "div" => BinOp::Div,
            "divide" => BinOp::Div,
            "divided" => BinOp::Div,
            "/" => BinOp::Div,
            "mul" => BinOp::Mul,
            "multiply" => BinOp::Mul,
            "multiplied" => BinOp::Mul,
            "*" => BinOp::Mul,
            "rem" => BinOp::Rem,
            "remainder" => BinOp::Rem,
            "%" => BinOp::Rem,
            "xor" => BinOp::BitXor,
            "^" => BinOp::BitXor,
            "and" => BinOp::And,
            "&" => BinOp::And,
            "or" => BinOp::Or,
            "|" => BinOp::Or,
            _ => return Err(()),
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

impl From<OP> for BinOp {
    fn from(op: OP) -> Self {
        match op {
            OP::Bitop(bitop) => match bitop {
                BITOP::_0(jj, cc) => BinOp::from_str(&cc.lemma).unwrap().apply_jj(&jj.lemma),
                BITOP::_1(nn, cc) => BinOp::from_str(&cc.lemma).unwrap().apply_jj(&nn.lemma),
            },
            OP::Arithop(a) => match a {
                ARITHOP::ARITH(arith, Some(inx)) => {
                    if inx.lemma == "from" {
                        let op = BinOp::from_str(&arith.lemma).unwrap();
                        if op == BinOp::Sub {
                            BinOp::SubFrom
                        } else {
                            op
                        }
                    } else {
                        BinOp::from_str(&arith.lemma).unwrap()
                    }
                }
                ARITHOP::ARITH(arith, None) => BinOp::from_str(&arith.lemma).expect(&arith.lemma),
            },
            OP::Shiftop(shift) => match shift {
                SHIFTOP::_0(_, _, _, nn, _) => BinOp::shift_with_dir(&nn.lemma).unwrap(),
                SHIFTOP::_1(jj, _) => BinOp::shift_with_dir(&jj.lemma).unwrap(),
            },
        }
    }
}

#[derive(Clone)]
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

#[derive(Clone)]
pub enum Object {
    Code(Code),
    Lit(Literal),
    Op(Op),
    PropOf(PropertyOf, Box<Object>),
    Mnn(Mnn),
    VbgMnn(VBG, Mnn),
    Prp(PRP),
    Str(String),
    Char(char),
}

impl From<OBJ> for Object {
    fn from(obj: OBJ) -> Self {
        match obj {
            OBJ::Code(code) => Object::Code(code.into()),
            OBJ::LIT(_, lit) => Object::Lit(lit.into()),
            OBJ::_0(lhs, op, rhs) => Object::Op(Op::new(lhs.into(), op.into(), rhs.into())),
            OBJ::MNN(_, mnn) => Object::Mnn(mnn.into()),
            OBJ::_1(_, vbg, mnn) => Object::VbgMnn(vbg, mnn.into()),
            OBJ::_2(prop, obj) => {
                let prop = PropertyOf::from(prop);
                match Object::from(obj) {
                    Object::Op(op) => match op.op {
                        BinOp::Sub => match prop.prop.lemma() {
                            "remainder" => Object::Op(op),
                            _ => Object::PropOf(prop, Box::new(Object::Op(op))),
                        },
                        BinOp::Div => match prop.prop.lemma() {
                            "remainder" => Object::Op(Op::new(*op.lhs, BinOp::Rem, *op.rhs)),
                            _ => Object::PropOf(prop, Box::new(Object::Op(op))),
                        },
                        _ => Object::PropOf(prop, Box::new(Object::Op(op))),
                    },
                    x => Object::PropOf(prop, Box::new(x)),
                }
            }
            OBJ::Prp(prp) => Object::Prp(prp),
            OBJ::Str(s) => Object::Str(s.word),
            OBJ::Char(c) => Object::Char(c.word.chars().skip(1).next().unwrap()),
        }
    }
}

impl From<Box<OBJ>> for Object {
    fn from(obj: Box<OBJ>) -> Self {
        Self::from(*obj)
    }
}

#[derive(Clone)]
pub enum PropOfMod {
    Mnn(Mnn),
    Mjj(MJJ),
}

impl PropOfMod {
    pub fn lemma(&self) -> &str {
        match self {
            PropOfMod::Mnn(mnn) => mnn.root_lemma(),
            PropOfMod::Mjj(mjj) => match mjj {
                MJJ::JJ(_, jj) => &jj.lemma,
                MJJ::JJR(_, jj) => &jj.lemma,
                MJJ::JJS(_, jj) => &jj.lemma,
            },
        }
    }
}

#[derive(Clone)]
pub struct PropertyOf {
    pub prop: PropOfMod,
}

impl From<PROP_OF> for PropertyOf {
    fn from(prop_of: PROP_OF) -> Self {
        match prop_of {
            PROP_OF::_0(_, mnn, _, _) | PROP_OF::_1(_, mnn, _) => PropertyOf {
                prop: PropOfMod::Mnn(mnn.into()),
            },
            PROP_OF::_2(_, mjj, _) | PROP_OF::_3(_, mjj, _, _) => PropertyOf {
                prop: PropOfMod::Mjj(mjj),
            },
        }
    }
}

#[derive(Clone)]
pub struct IsProperty {
    pub mvb: MVB,
    pub prop_type: IsPropMod,
}

#[derive(Clone)]
pub enum IsPropMod {
    Mjj(MJJ),
    Rel(Relation),
    Obj(Object),
    RangeMod(RangeMod),
    None,
}

impl From<PROP> for IsProperty {
    fn from(prop: PROP) -> Self {
        match prop {
            PROP::_0(mvb, mjj) => IsProperty {
                mvb,
                prop_type: IsPropMod::Mjj(mjj.into()),
            },
            PROP::_1(mvb, mrel) => IsProperty {
                mvb,
                prop_type: IsPropMod::Rel(mrel.into()),
            },
            PROP::_2(mvb, obj) => IsProperty {
                mvb,
                prop_type: IsPropMod::Obj(obj.into()),
            },
            PROP::Mvb(mvb) => IsProperty {
                mvb,
                prop_type: IsPropMod::None,
            },
            PROP::_3(mvb, rangemod) => IsProperty {
                mvb,
                prop_type: IsPropMod::RangeMod(rangemod.into()),
            },
        }
    }
}

#[derive(Copy, Clone)]
pub enum Comparator {
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
            "small" => Comparator::Lt,
            "greater" => Comparator::Gt,
            "great" => Comparator::Gt,
            "larger" => Comparator::Gt,
            "large" => Comparator::Gt,
            "equal" => Comparator::Eq,
            "unequal" => Comparator::Neq,
            "same" => Comparator::Eq,
            _ => return Err(()),
        })
    }
}

impl Display for Comparator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Comparator::Lt => "<",
            Comparator::Gt => ">",
            Comparator::Lte => "<=",
            Comparator::Gte => ">=",
            Comparator::Eq => "==",
            Comparator::Neq => "!=",
        })
    }
}

impl Comparator {
    fn apply_eq(self) -> Self {
        match self {
            Comparator::Lt => Comparator::Lte,
            Comparator::Gt => Comparator::Gte,
            x => x,
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

#[derive(Copy, Clone)]
pub enum IfExpr {
    If,
    Iff,
}

#[derive(Clone)]
pub struct BoolCond {
    pub if_expr: IfExpr,
    pub negated: bool,
    pub value: BoolValue,
}

enum BoolCondIntermediate {
    Complete(BoolCond),
    Incomplete {
        if_expr: IfExpr,
        negated: bool,
        value: BoolValueIncomplete,
    },
}

impl From<COND> for BoolCondIntermediate {
    fn from(cond: COND) -> Self {
        let (if_expr, value) = match cond {
            COND::_0(_if, value) => (IfExpr::If, BoolValueIntermediate::from(value)),
            COND::_1(_iff, value) => (IfExpr::Iff, BoolValueIntermediate::from(value)),
        };
        match value {
            BoolValueIntermediate::Complete(value) => BoolCondIntermediate::Complete(BoolCond {
                if_expr,
                negated: false,
                value,
            }),
            BoolValueIntermediate::Incomplete(value) => BoolCondIntermediate::Incomplete {
                if_expr,
                negated: false,
                value,
            },
        }
    }
}

#[derive(Clone)]
pub enum BoolValue {
    Assert(Assert),
    QAssert(QuantAssert),
    Code(Code),
    Event(Event),
}

enum BoolValueIncomplete {
    Event(EventIntermediate),
}

enum BoolValueIntermediate {
    Complete(BoolValue),
    Incomplete(BoolValueIncomplete),
}

impl From<BOOL_EXPR> for BoolValueIntermediate {
    fn from(boolexpr: BOOL_EXPR) -> Self {
        use BoolValueIntermediate::*;
        match boolexpr {
            BOOL_EXPR::Assert(a) => Complete(BoolValue::Assert(a.into())),
            BOOL_EXPR::Qassert(q) => Complete(BoolValue::QAssert(q.into())),
            BOOL_EXPR::Code(c) => Complete(BoolValue::Code(c.into())),
            BOOL_EXPR::Event(e) => Incomplete(BoolValueIncomplete::Event(e.into())),
        }
    }
}

#[derive(Clone)]
pub enum EventVB {
    VBZ(VBZ),
    VBD(VBD),
}

struct EventIntermediate {
    mnn: Mnn,
    vb: EventVB,
}

impl From<EVENT> for EventIntermediate {
    fn from(e: EVENT) -> Self {
        match e {
            EVENT::_0(mnn, vbd) => EventIntermediate {
                mnn: mnn.into(),
                vb: EventVB::VBD(vbd),
            },
            EVENT::_1(mnn, vbz) => EventIntermediate {
                mnn: mnn.into(),
                vb: EventVB::VBZ(vbz),
            },
        }
    }
}

impl EventIntermediate {
    fn complete(self, inner: EventTarget) -> Event {
        Event {
            mnn: self.mnn,
            vb: self.vb,
            inner,
        }
    }
}

#[derive(Clone)]
pub enum EventTarget {
    Object(Object),
    This,
}

#[derive(Clone)]
pub struct Event {
    pub mnn: Mnn,
    pub vb: EventVB,
    pub inner: EventTarget,
}

pub enum Specification {
    RetIf(ReturnIf),
    Mret(MReturn),
    SpecAtom(SpecAtom),
    SpecCond(SpecCond),
    SpecTerm(SpecTerm),
}

pub enum SpecAtom {
    HAssert(HardAssert),
    QAssert(HardQuantAssert),
    Action(ActionObj),
    Side,
}

impl From<SPEC_ATOM> for SpecAtom {
    fn from(atom: SPEC_ATOM) -> Self {
        match atom {
            SPEC_ATOM::Hassert(hassert) => SpecAtom::HAssert(hassert.into()),
            SPEC_ATOM::Hqassert(hqassert) => SpecAtom::QAssert(hqassert.into()),
            SPEC_ATOM::Assign(action_obj) => SpecAtom::Action(action_obj.into()),
            SPEC_ATOM::Side(_) => SpecAtom::Side,
        }
    }
}

impl From<S> for Specification {
    fn from(spec_tree: S) -> Self {
        match spec_tree {
            S::Mret(m) => Specification::Mret(m.into()),
            S::Retif(retif) => Specification::RetIf(retif.into()),
            S::Spec_atom(atom) => Specification::SpecAtom(atom.into()),
            S::Spec_cond(cond) => Specification::SpecCond(cond.into()),
            S::Spec_term(spec_term) => Specification::SpecTerm(spec_term.into()),
        }
    }
}

pub struct SpecCond {
    pub cond: BoolCond,
    pub atom: SpecAtom,
}

impl From<SPEC_COND> for SpecCond {
    fn from(cond: SPEC_COND) -> Self {
        match cond {
            SPEC_COND::_0(atom, cond) | SPEC_COND::_1(cond, _, atom) => {
                let atom = SpecAtom::from(atom);
                let cond = match BoolCondIntermediate::from(cond) {
                    BoolCondIntermediate::Complete(cond) => cond,
                    BoolCondIntermediate::Incomplete {
                        if_expr,
                        negated,
                        value,
                    } => match value {
                        BoolValueIncomplete::Event(e) => BoolCond {
                            if_expr,
                            negated,
                            value: BoolValue::Event(e.complete(unimplemented!())),
                        },
                    },
                };

                SpecCond { cond, atom }
            }
        }
    }
}

pub enum SpecAtom_ {
    HAssert(HardAssert),
    QAssert(HardQuantAssert),
    Action(ActionObj),
    Side,
    Mret(MReturn),
}

pub struct SpecTermVal {
    atom: SpecAtom_,
    cond: Option<BoolCond>,
}

struct SpecTermValIntermediate {
    atom: SpecAtom_,
    cond: Option<BoolCondIntermediate>,
}

pub struct SpecTerm {
    vals: Vec<Vec<SpecTermVal>>,
}

pub struct SpecTermIntermediate {
    vals: Vec<Vec<SpecTermValIntermediate>>,
}

impl From<SPEC_ATOM> for SpecAtom_ {
    fn from(s: SPEC_ATOM) -> Self {
        match s {
            SPEC_ATOM::Hassert(h) => SpecAtom_::HAssert(h.into()),
            SPEC_ATOM::Hqassert(h) => SpecAtom_::QAssert(h.into()),
            SPEC_ATOM::Side(_s) => SpecAtom_::Side,
            SPEC_ATOM::Assign(a) => SpecAtom_::Action(a.into()),
        }
    }
}

impl From<MRET> for SpecAtom_ {
    fn from(m: MRET) -> Self {
        SpecAtom_::Mret(m.into())
    }
}

impl From<SPEC_ITEM> for SpecTermValIntermediate {
    fn from(s: SPEC_ITEM) -> Self {
        let (atom, cond) = match s {
            SPEC_ITEM::Spec_cond(c) => match c {
                SPEC_COND::_0(atom, cond) | SPEC_COND::_1(cond, _, atom) => {
                    (SpecAtom_::from(atom), BoolCondIntermediate::from(cond))
                }
            },
            SPEC_ITEM::Retif_(r) => match r {
                RETIF_::_0(mret, cond) | RETIF_::_1(cond, _, mret) => {
                    (SpecAtom_::from(mret), BoolCondIntermediate::from(cond))
                }
            },
        };
        SpecTermValIntermediate {
            atom,
            cond: Some(cond),
        }
    }
}

impl From<SPEC_ATOM> for SpecTermValIntermediate {
    fn from(atom: SPEC_ATOM) -> Self {
        SpecTermValIntermediate {
            atom: atom.into(),
            cond: None,
        }
    }
}

impl From<MRET> for SpecTermValIntermediate {
    fn from(atom: MRET) -> Self {
        SpecTermValIntermediate {
            atom: atom.into(),
            cond: None,
        }
    }
}

struct SpecChainPre {
    spec_term: SpecTermIntermediate,
    rb: RB,
}

impl From<SPEC_CHAIN_PRE> for SpecChainPre {
    fn from(s: SPEC_CHAIN_PRE) -> Self {
        match s {
            SPEC_CHAIN_PRE::_0(chain, _, rb) => SpecChainPre {
                spec_term: chain.into(),
                rb,
            },
        }
    }
}

impl From<Box<SPEC_CHAIN>> for SpecTermIntermediate {
    fn from(s: Box<SPEC_CHAIN>) -> Self {
        Self::from(*s)
    }
}

impl From<SPEC_CHAIN> for SpecTermIntermediate {
    fn from(s: SPEC_CHAIN) -> Self {
        match s {
            SPEC_CHAIN::Spec_item(i) => SpecTermIntermediate {
                vals: vec![vec![i.into()]],
            },
            SPEC_CHAIN::_0(chain, _, rb, spec_item) => {
                assert_eq!(rb.lemma, "otherwise");
                let mut chain = SpecTermIntermediate::from(chain);
                chain.vals.push(vec![spec_item.into()]);
                chain
            }
            SPEC_CHAIN::_1(chain, _, cc, spec_item) => {
                if let Some(cc) = cc {
                    assert_eq!(cc.lemma, "and");
                }
                let mut chain = SpecTermIntermediate::from(chain);
                chain.vals.last_mut().unwrap().push(spec_item.into());
                chain
            }
        }
    }
}

impl From<SPEC_TERM> for SpecTerm {
    fn from(s: SPEC_TERM) -> Self {
        fn from_chain(
            chain: Option<SPEC_CHAIN_PRE>,
            vals: Vec<Vec<SpecTermValIntermediate>>,
        ) -> SpecTerm {
            let spec_term_i = match chain {
                None => SpecTermIntermediate { vals },
                Some(chain) => {
                    let mut chain = SpecChainPre::from(chain);
                    assert_eq!(chain.rb.lemma, "otherwise");
                    chain.spec_term.vals.extend(vals);
                    chain.spec_term
                }
            };

            // If we see an event:
            //  Look back - if we find a target in a previous OW, use that
            //  If we find a target in a previous item in the current AND body, use that
            let mut spec_term_vals = Vec::with_capacity(spec_term_i.vals.len());
            for val in spec_term_i.vals {
                let mut conjunction_vals: Vec<SpecTermVal> = Vec::with_capacity(val.len());
                for item in val {
                    let cond = match item.cond {
                        None => None,
                        Some(BoolCondIntermediate::Complete(cond)) => Some(cond),
                        Some(BoolCondIntermediate::Incomplete {
                            if_expr,
                            negated,
                            value,
                        }) => Some(match value {
                            BoolValueIncomplete::Event(e) => match conjunction_vals.last() {
                                None => BoolCond {
                                    if_expr,
                                    negated,
                                    value: BoolValue::Event(e.complete(EventTarget::This)),
                                },
                                Some(item) => match &item.atom {
                                    SpecAtom_::Mret(mret) => BoolCond {
                                        if_expr,
                                        negated,
                                        value: BoolValue::Event(
                                            e.complete(EventTarget::Object(mret.ret_val.clone())),
                                        ),
                                    },
                                    _ => unimplemented!(),
                                },
                            },
                        }),
                    };
                    conjunction_vals.push(SpecTermVal {
                        atom: item.atom,
                        cond,
                    });
                }
                spec_term_vals.push(conjunction_vals);
            }
            SpecTerm {
                vals: spec_term_vals,
            }
        }

        match s {
            SPEC_TERM::RETIF_TERM(chain, retif_term) => from_chain(
                chain,
                match retif_term {
                    RETIF_TERM::_0(retif, _, rb, val) => {
                        assert_eq!(rb.lemma, "otherwise");
                        vec![
                            vec![SPEC_ITEM::Retif_(retif).into()],
                            vec![SpecTermValIntermediate {
                                atom: SpecAtom_::Mret(MReturn {
                                    ret_val: val.into(),
                                }),
                                cond: None,
                            }],
                        ]
                    }
                },
            ),
            SPEC_TERM::SPEC_ATOM(chain, item) => from_chain(chain, vec![vec![item.into()]]),
            SPEC_TERM::MRET(chain, item) => from_chain(chain, vec![vec![item.into()]]),
            SPEC_TERM::SPEC_ITEM(chain, item) => from_chain(chain, vec![vec![item.into()]]),
        }
    }
}

#[derive(Clone)]
pub struct MReturn {
    pub(crate) ret_val: Object,
}

impl From<MRET> for MReturn {
    fn from(mret: MRET) -> Self {
        match mret {
            MRET::_0(_, ret_val) => MReturn {
                ret_val: ret_val.into(),
            },
            MRET::_1(ret_val, vbz, _) => {
                assert!(["is", "be"].contains(&vbz.lemma.as_str()));
                MReturn {
                    ret_val: ret_val.into(),
                }
            }
            MRET::_2(ret_val, _) => MReturn {
                ret_val: ret_val.into(),
            },
        }
    }
}

#[derive(Copy, Clone)]
pub enum UnaryOp {
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

#[derive(Clone)]
pub enum QuantItem<V: Clone> {
    Code(Code),
    Assert(V),
}

#[derive(Clone)]
pub struct QuantAssert {
    pub quant_expr: QuantExpr,
    pub assertion: QuantItem<Assert>,
}

impl From<QASSERT> for QuantAssert {
    fn from(qassert: QASSERT) -> Self {
        match qassert {
            QASSERT::_0(quant_expr, _, assert) | QASSERT::_1(assert, quant_expr) => QuantAssert {
                quant_expr: quant_expr.into(),
                assertion: QuantItem::Assert(assert.into()),
            },
            QASSERT::_2(code, quant_expr) => QuantAssert {
                quant_expr: quant_expr.into(),
                assertion: QuantItem::Code(code.into()),
            },
        }
    }
}

#[derive(Clone)]
pub struct HardQuantAssert {
    pub quant_expr: QuantExpr,
    pub assertion: QuantItem<HardAssert>,
}

impl From<HQASSERT> for HardQuantAssert {
    fn from(qassert: HQASSERT) -> Self {
        match qassert {
            HQASSERT::_0(quant_expr, _, hassert) | HQASSERT::_1(hassert, quant_expr) => {
                HardQuantAssert {
                    quant_expr: quant_expr.into(),
                    assertion: QuantItem::Assert(hassert.into()),
                }
            }
            HQASSERT::_2(code, quant_expr) => HardQuantAssert {
                quant_expr: quant_expr.into(),
                assertion: QuantItem::Code(code.into()),
            },
        }
    }
}

impl HardQuantAssert {
    pub(crate) fn spec_scope(&self) -> ConditionModifier {
        match &self.assertion {
            QuantItem::Code(_c) => ConditionModifier::Post,
            QuantItem::Assert(h) => {
                ConditionModifier::from_str(&h.md.lemma).unwrap_or(ConditionModifier::Post)
            }
        }
    }
}

#[derive(Clone)]
pub struct HardAssert {
    pub md: MD,
    pub assert: Assert,
}

impl From<HASSERT> for HardAssert {
    fn from(hassert: HASSERT) -> Self {
        match hassert {
            HASSERT::_0(obj, md, prop) => HardAssert {
                md,
                assert: Assert::from(ASSERT::_0(obj, prop)),
            },
            HASSERT::_1(obj, cc, hassert) => {
                let mut hassert = HardAssert::from(hassert);
                match CC::from_str(&cc.lemma).unwrap() {
                    CC::And => hassert.assert.objects.last_mut().unwrap().push(obj.into()),
                    CC::Or => hassert.assert.objects.push(vec![obj.into()]),
                }
                hassert
            }
        }
    }
}

impl From<Box<HASSERT>> for HardAssert {
    fn from(hassert: Box<HASSERT>) -> Self {
        Self::from(*hassert)
    }
}

#[derive(Clone)]
pub struct QuantExpr {
    pub quant: Quantifier,
    pub range: Option<RangeMod>,
    pub range_conds: Vec<Vec<Relation>>,
}

impl From<QUANT_EXPR> for QuantExpr {
    fn from(qexpr: QUANT_EXPR) -> Self {
        match qexpr {
            QUANT_EXPR::QUANT(quant, rangemod) => QuantExpr {
                quant: quant.into(),
                range: rangemod.map(RangeMod::from),
                range_conds: vec![vec![]],
            },
            QUANT_EXPR::_0(quantexpr, _, cc, mrel) => {
                let mut quantexpr = QuantExpr::from(quantexpr);
                match CC::from_str(&cc.lemma).unwrap() {
                    CC::And => quantexpr.range_conds.last_mut().unwrap().push(mrel.into()),
                    CC::Or => quantexpr.range_conds.push(vec![mrel.into()]),
                }

                quantexpr
            }
            QUANT_EXPR::_1(quantexpr, _, mrel) => {
                let mut quantexpr = QuantExpr::from(quantexpr);
                quantexpr.range_conds.last_mut().unwrap().push(mrel.into());
                quantexpr
            }
        }
    }
}

impl From<Box<QUANT_EXPR>> for QuantExpr {
    fn from(qexpr: Box<QUANT_EXPR>) -> Self {
        Self::from(*qexpr)
    }
}

#[derive(Copy, Clone)]
pub enum CC {
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

#[derive(Clone)]
pub struct Relation {
    pub objects: Vec<Vec<Object>>,
    pub op: Comparator,
    pub modifier: Option<RB>,
}

impl From<REL> for Relation {
    fn from(rel: REL) -> Self {
        match rel {
            REL::_0(tjj, _in, obj) => Relation {
                objects: vec![vec![obj.into()]],
                op: Comparator::from_str(
                    match tjj {
                        TJJ::JJ(_, jj) => jj.lemma,
                        TJJ::JJR(_, jjr) => jjr.lemma,
                        TJJ::JJS(_, jjs) => jjs.lemma,
                    }
                    .as_str(),
                )
                .unwrap_or(Comparator::Neq),
                modifier: None,
            },
            REL::_1(tjj, _eqto, obj) => Relation {
                objects: vec![vec![obj.into()]],
                op: Comparator::from_str(
                    match tjj {
                        TJJ::JJ(_, jj) => jj.lemma,
                        TJJ::JJR(_, jjr) => jjr.lemma,
                        TJJ::JJS(_, jjs) => jjs.lemma,
                    }
                    .as_str(),
                )
                .unwrap_or(Comparator::Neq)
                .apply_eq(),
                modifier: None,
            },
            REL::_2(_in, obj) => Relation {
                objects: vec![vec![obj.into()]],
                op: Comparator::Lt,
                modifier: None,
            },
            REL::_3(rel, cc, obj) => {
                let mut rel = Relation::from(rel);
                match CC::from_str(&cc.lemma).unwrap() {
                    CC::And => rel.objects.last_mut().unwrap().push(obj.into()),
                    CC::Or => rel.objects.push(vec![obj.into()]),
                }

                rel
            }
        }
    }
}

impl From<MREL> for Relation {
    fn from(mrel: MREL) -> Self {
        match mrel {
            MREL::REL(Some(rb), rel) => {
                let mut rel = Relation::from(rel);
                rel.modifier = Some(rb);
                rel
            }
            MREL::REL(None, rel) => rel.into(),
        }
    }
}

impl From<Box<REL>> for Relation {
    fn from(rel: Box<REL>) -> Self {
        Self::from(*rel)
    }
}

#[derive(Copy, Clone)]
pub enum UpperBound {
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
    pub fn lt(self) -> Comparator {
        match self {
            UpperBound::Inclusive => Comparator::Lte,
            UpperBound::Exclusive => Comparator::Lt,
        }
    }
}

#[derive(Clone)]
pub struct Range {
    pub ident: Option<Object>,
    pub start: Option<Object>,
    pub end: Option<Object>,
}

impl From<RANGE> for Range {
    fn from(r: RANGE) -> Self {
        match r {
            RANGE::_0(ident, _, start, _, end) => Self {
                ident: Some(ident.into()),
                start: Some(start.into()),
                end: Some(end.into()),
            },
            RANGE::_1(_, start, _, end) => Self {
                ident: None,
                start: Some(start.into()),
                end: Some(end.into()),
            },
            RANGE::_2(_, _, end) => Self {
                ident: None,
                start: None,
                end: Some(end.into()),
            },
        }
    }
}

#[derive(Clone)]
pub struct RangeMod {
    pub range: Range,
    pub upper_bound: UpperBound,
}

impl From<RANGEMOD> for RangeMod {
    fn from(rmod: RANGEMOD) -> Self {
        match rmod {
            RANGEMOD::Range(range) => Self {
                range: range.into(),
                upper_bound: UpperBound::Exclusive,
            },
            RANGEMOD::_0(range, _, jj) => Self {
                range: range.into(),
                upper_bound: UpperBound::from_str(&jj.lemma).unwrap(),
            },
        }
    }
}

#[derive(Clone)]
pub struct Quantifier {
    pub universal: bool,
    pub obj: Object,
}

impl From<QUANT> for Quantifier {
    fn from(quant: QUANT) -> Self {
        match quant {
            QUANT::_0(_a, dt, obj) => Quantifier {
                universal: ["all", "each", "any"].contains(&dt.lemma.as_str()),
                obj: obj.into(),
            },
        }
    }
}

#[derive(Clone)]
pub enum MnnMod {
    Jj(JJ),
    Vbn(VBN),
}

impl MnnMod {
    pub(crate) fn lemma(&self) -> &str {
        match self {
            MnnMod::Jj(jj) => &jj.lemma,
            MnnMod::Vbn(vbn) => &vbn.lemma,
        }
    }
}

#[derive(Clone)]
pub struct Mnn {
    pub adjs: Vec<MnnMod>,
    pub root: Terminal,
}

impl From<MNN> for Mnn {
    fn from(mnn: MNN) -> Self {
        match mnn {
            MNN::Nn(nn) => Mnn {
                adjs: Vec::with_capacity(0),
                root: nn.into(),
            },
            MNN::Nns(nn) => Mnn {
                adjs: Vec::with_capacity(0),
                root: nn.into(),
            },
            MNN::Nnp(nn) => Mnn {
                adjs: Vec::with_capacity(0),
                root: nn.into(),
            },
            MNN::Nnps(nn) => Mnn {
                adjs: Vec::with_capacity(0),
                root: nn.into(),
            },
            MNN::_0(jj, mnn) => {
                let mut mnn = Mnn::from(mnn);
                mnn.adjs.insert(0, MnnMod::Jj(jj));
                mnn
            }
            MNN::_1(vbn, mnn) => {
                let mut mnn = Mnn::from(mnn);
                mnn.adjs.insert(0, MnnMod::Vbn(vbn));
                mnn
            }
        }
    }
}

impl Mnn {
    pub fn root_lemma(&self) -> &str {
        &self.root.lemma
    }
}

impl From<Box<MNN>> for Mnn {
    fn from(mnn: Box<MNN>) -> Self {
        Self::from(*mnn)
    }
}

#[derive(Clone)]
pub struct Assert {
    pub property: IsProperty,
    // (a and b and c) or (d and e and f)
    pub objects: Vec<Vec<Object>>,
}

impl From<ASSERT> for Assert {
    fn from(a: ASSERT) -> Self {
        match a {
            ASSERT::_0(obj, prop) => Assert {
                objects: vec![vec![obj.into()]],
                property: prop.into(),
            },
            ASSERT::_1(obj, cc, assert) => {
                let mut assert = Assert::from(assert);
                match CC::from_str(&cc.lemma).unwrap() {
                    CC::And => assert.objects.last_mut().unwrap().push(obj.into()),
                    CC::Or => assert.objects.push(vec![obj.into()]),
                }
                assert
            }
        }
    }
}

impl From<Box<ASSERT>> for Assert {
    fn from(assert: Box<ASSERT>) -> Self {
        Self::from(*assert)
    }
}

#[derive(Clone)]
pub struct ReturnIf {
    pub ret_pred: Vec<(BoolCond, Object)>,
}

// TODO: Add "compiler" pass which detects overflow / panic / event occur syntax
//  and pulls in relevant object from previous item
impl From<RETIF> for ReturnIf {
    fn from(retif: RETIF) -> Self {
        match retif {
            RETIF::_0(mret, cond) | RETIF::_1(cond, _, mret) => {
                let mret = MReturn::from(mret);
                let cond = match BoolCondIntermediate::from(cond) {
                    BoolCondIntermediate::Complete(cond) => cond,
                    BoolCondIntermediate::Incomplete {
                        if_expr,
                        negated,
                        value,
                    } => match value {
                        BoolValueIncomplete::Event(e) => BoolCond {
                            if_expr,
                            negated,
                            value: BoolValue::Event(
                                e.complete(EventTarget::Object(mret.ret_val.clone())),
                            ),
                        },
                    },
                };
                ReturnIf {
                    ret_pred: vec![(cond, mret.ret_val)],
                }
            }
            RETIF::_2(retif1, _, rb, retif2) => {
                assert_eq!(rb.lemma, "otherwise");
                let mut retif1 = ReturnIf::from(retif1);
                let mut retif2 = ReturnIf::from(retif2);
                retif1.ret_pred.append(&mut retif2.ret_pred);
                retif1
            }
            RETIF::_3(retif, _, rb, obj) => {
                assert_eq!(rb.lemma, "otherwise");
                let mut retif = ReturnIf::from(retif);
                let mut pred = retif.ret_pred.last().unwrap().0.clone();
                pred.negated = true;
                retif.ret_pred.push((pred, obj.into()));
                retif
            }
            RETIF::_4(mret, _, ow_mret, cond) => {
                let mret = MReturn::from(mret);
                let ow_mret = MReturn::from(ow_mret);
                let ow_cond = match BoolCondIntermediate::from(cond) {
                    BoolCondIntermediate::Complete(cond) => cond,
                    BoolCondIntermediate::Incomplete {
                        if_expr,
                        negated,
                        value,
                    } => match value {
                        BoolValueIncomplete::Event(e) => BoolCond {
                            if_expr,
                            negated,
                            value: BoolValue::Event(
                                e.complete(EventTarget::Object(mret.ret_val.clone())),
                            ),
                        },
                    },
                };

                let mut cond = ow_cond.clone();
                cond.negated = true;
                ReturnIf {
                    ret_pred: vec![(ow_cond, ow_mret.ret_val), (cond, mret.ret_val)],
                }
            }
        }
    }
}

impl From<Box<RETIF>> for ReturnIf {
    fn from(retif: Box<RETIF>) -> Self {
        Self::from(*retif)
    }
}

pub trait Lemma {
    fn root_lemma(&self) -> &str;
}

impl Lemma for MVB {
    fn root_lemma(&self) -> &str {
        match self {
            MVB::VB(_, vb) => &vb.lemma,
            MVB::VBZ(_, vb) => &vb.lemma,
            MVB::VBP(_, vb) => &vb.lemma,
            MVB::VBN(_, vb) => &vb.lemma,
            MVB::VBG(_, vb) => &vb.lemma,
            MVB::VBD(_, vb) => &vb.lemma,
        }
    }
}

pub enum ActionObj {
    // Action with one object
    // eg. print item
    Action1(VBZ, Object),
    // eg. add x to y
    // eg. divide x by y
    // action with two objects
    // Where the target is ambiguous
    Action2Resolved {
        vbz: VBZ,
        target: Object,
        value: Object,
    },
    // Where the target is resolved
    Action2Ambiguous(VBZ, Object, Object),
    // target, value
    Set {
        target: Object,
        value: Object,
    },
}

/// TODO: Target resolution scheme / loading from file? Default to first object being target?
///    Investigate NN confusion issue and possible fixes
impl From<ASSIGN> for ActionObj {
    fn from(a: ASSIGN) -> Self {
        match a {
            ASSIGN::_0(vbz, obj0, xin, obj1) => match (vbz.lemma.as_str(), xin.lemma.as_str()) {
                ("store", "in") | ("subtract", "from") | ("divide", "from") | ("add", "to") => {
                    ActionObj::Action2Resolved {
                        vbz,
                        target: obj1.into(),
                        value: obj0.into(),
                    }
                }
                ("divide", "by") | ("subtract", "by") | ("multiply", "by") => {
                    ActionObj::Action2Resolved {
                        vbz,
                        target: obj0.into(),
                        value: obj1.into(),
                    }
                }
                ("assign", "to") => ActionObj::Set {
                    target: obj1.into(),
                    value: obj0.into(),
                },
                ("set", "to") => ActionObj::Set {
                    target: obj0.into(),
                    value: obj1.into(),
                },
                _ => ActionObj::Action2Ambiguous(vbz, obj0.into(), obj1.into()),
            },
            ASSIGN::_1(vbz, obj0, _to, obj1) => match vbz.lemma.as_str() {
                "set" => ActionObj::Set {
                    target: obj0.into(),
                    value: obj1.into(),
                },
                "assign" | "add" => ActionObj::Set {
                    target: obj1.into(),
                    value: obj0.into(),
                },
                _ => ActionObj::Action2Ambiguous(vbz, obj0.into(), obj1.into()),
            },
            ASSIGN::_2(vbz, _dt, _jj, obj) => {
                if _dt.is_some() {
                    println!("{}", vbz.word);
                }
                ActionObj::Action1(vbz, obj.into())
            }
        }
    }
}

pub enum ConditionModifier {
    Pre,
    Post,
}

impl FromStr for ConditionModifier {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "will" => Ok(ConditionModifier::Post),
            "does" => Ok(ConditionModifier::Post),
            "must" => Ok(ConditionModifier::Pre),
            _ => Err(()),
        }
    }
}

impl ConditionModifier {
    pub(crate) fn keyword(&self) -> &'static str {
        match self {
            ConditionModifier::Pre => "requires",
            ConditionModifier::Post => "ensures",
        }
    }
}
