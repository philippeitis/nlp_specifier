use std::fmt::{Display, Formatter};
use std::str::FromStr;

use itertools::Itertools;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Expr};

use crate::nl_ir::{
    ActionObj, Assert, BoolCond, BoolValue, Code, ConditionModifier, Event, EventTarget,
    HardAssert, HardQuantAssert, IfExpr, IsPropMod, IsProperty, Lemma, Literal, MReturn, MnnMod,
    Object, Op, QuantAssert, QuantExpr, QuantItem, Relation, ReturnIf, SpecAtom, SpecCond,
    Specification,
};
use crate::parse_tree::tree::{MJJ, MVB};

fn is_negation<S: AsRef<str>>(s: S) -> bool {
    let s = s.as_ref();
    s == "not" || s == "n't"
}

#[derive(Debug)]
pub enum SpecificationError {
    Syn(syn::Error),
    UnsupportedSpec(&'static str),
    Unimplemented,
}

impl From<syn::Error> for SpecificationError {
    fn from(err: syn::Error) -> Self {
        SpecificationError::Syn(err)
    }
}

pub trait AsCode {
    fn as_code(&self) -> Result<Expr, SpecificationError>;
}

/// Temporary workaround until we create a custom enum for this task
pub trait AsCodeValue {
    type Output;
    fn as_code_value(&self) -> Result<Self::Output, SpecificationError>;
}

impl AsCode for Code {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        Ok(syn::parse_str(self.code.trim_matches('`'))?)
    }
}

impl AsCode for Literal {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        Ok(syn::parse_str(&self.s)?)
    }
}

impl AsCode for Op {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        let lhs = self.lhs.as_code()?;
        let op = syn::parse_str::<syn::BinOp>(&self.op.to_string())?;
        let rhs = self.rhs.as_code()?;

        Ok(syn::parse_str(
            &quote::quote! {(#lhs) #op (#rhs)}.to_string(),
        )?)
    }
}

impl AsCode for Object {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        match self {
            Object::Code(c) => c.as_code(),
            Object::Lit(l) => l.as_code(),
            Object::Op(o) => o.as_code(),
            Object::PropOf(prop, obj) => {
                // TODO: Make this an error / resolve cases such as this.
                let obj = obj.as_code()?;
                Ok(syn::parse_str(&format!(
                    "{}.{}()",
                    prop.prop.lemma(),
                    quote::quote! {#obj}.to_string()
                ))?)
            }
            Object::Mnn(mnn) => {
                // TODO: Make this an error / resolve cases such as this.
                Ok(syn::parse_str(
                    &mnn.adjs
                        .iter()
                        .map(MnnMod::lemma)
                        .chain(std::iter::once(mnn.root_lemma()))
                        .join("_"),
                )?)
            }
            Object::VbgMnn(vbg, mnn) => {
                // TODO: Make this an error / resolve cases such as this.
                Ok(syn::parse_str(&format!(
                    "{}.{}()",
                    mnn.adjs
                        .iter()
                        .map(MnnMod::lemma)
                        .chain(std::iter::once(mnn.root_lemma()))
                        .join("_"),
                    vbg.lemma,
                ))?)
            }
            Object::Prp(prp) => {
                // personal pronoun - what is being referenced here?
                println!("Object::Prp({})", prp.lemma);
                Err(SpecificationError::Unimplemented)
            }
            Object::Str(s) => Ok(syn::parse_str(s)?),
            Object::Char(c) => Ok(syn::parse_str(&format!("'{}'", c))?),
        }
    }
}

impl AsCode for MReturn {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        let val = self.ret_val.as_code()?;
        Ok(syn::parse_str(
            &quote::quote! {result == (#val)}.to_string(),
        )?)
    }
}

impl AsCode for Event {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        let negations = self
            .mnn
            .adjs
            .iter()
            .filter(|x| match x {
                MnnMod::Jj(jj) => is_negation(&jj.lemma),
                MnnMod::Vbn(_) => false,
            })
            .count();
        match &self.inner {
            EventTarget::Object(inner) => {
                let body = inner.as_code()?;
                if negations % 2 == 0 {
                    Ok(syn::parse_str(&format!(
                        "{}s!({})",
                        self.mnn.root_lemma(),
                        quote::quote! {#body}.to_string()
                    ))?)
                } else {
                    Ok(syn::parse_str(&format!(
                        "!{}s!({})",
                        self.mnn.root_lemma(),
                        quote::quote! {#body}.to_string()
                    ))?)
                }
            }
            EventTarget::This => {
                return Ok(syn::parse_str(&format!("{}!()", self.mnn.root_lemma()))?)
            }
        }
    }
}

trait Apply {
    fn apply(&self, lhs: &ExprString) -> Result<ExprString, SpecificationError>;
}

impl Apply for IsProperty {
    /// TODO: This is somewhat expensive, and when properties are tested repeatedly, it would be nice to
    ///     avoid repeating work. Benchmark whether this is an issue in reality.
    fn apply(&self, lhs: &ExprString) -> Result<ExprString, SpecificationError> {
        let s = match &self.prop_type {
            IsPropMod::Obj(rhs) => {
                if !["is", "be"].contains(&self.mvb.root_lemma()) {
                    return Err(SpecificationError::UnsupportedSpec(
                        "Relationships between objects must be of the is or be format",
                    ));
                };

                let rhs = rhs.as_code()?;
                format!("({} == {})", lhs, quote::quote! {#rhs}.to_string())
            }
            IsPropMod::Mjj(mjj) => {
                let (rb, lemma) = match mjj {
                    MJJ::JJ(rb, jj) => (rb.as_ref(), &jj.lemma),
                    MJJ::JJR(rb, jj) => (rb.as_ref(), &jj.lemma),
                    MJJ::JJS(rb, jj) => (rb.as_ref(), &jj.lemma),
                };
                let sym = if rb.map(|x| is_negation(&x.lemma)).unwrap_or(false) {
                    "!"
                } else {
                    ""
                };
                if ["modified", "altered", "changed"].contains(&lemma.as_str()) {
                    format!("{}({} != old({}))", sym, lhs, lhs)
                } else if ["modify", "alter", "change"].contains(&lemma.as_str()) {
                    format!("{}({} == old({}))", sym, lhs, lhs)
                } else {
                    // TODO: Make this an error / resolve cases such as this.
                    format!("{}({}).{}()", sym, lhs, lemma)
                }
            }
            IsPropMod::Rel(rel) => rel.apply(lhs)?.to_string(),
            IsPropMod::RangeMod(range) => {
                let rangex = &range.range;
                match (
                    rangex.ident.as_ref(),
                    rangex.start.as_ref(),
                    rangex.end.as_ref(),
                ) {
                    (None, Some(start), Some(end)) => {
                        let start = start.as_code()?;
                        let end = end.as_code()?;
                        let upper_bound = range.upper_bound.lt();

                        format!(
                            "({} <= {} && {} {} {})",
                            quote::quote! {#start}.to_string(),
                            lhs,
                            lhs,
                            upper_bound,
                            quote::quote! {#end}.to_string(),
                        )
                    }
                    _ => return Err(SpecificationError::UnsupportedSpec("PROP: RANGE case")),
                }
            }
            IsPropMod::None => {
                if ["modify", "alter", "change"].contains(&self.mvb.root_lemma()) {
                    format!("({} != old({}))", lhs, lhs)
                } else if self.mvb.root_lemma() == "overflow" {
                    format!("overflows!({})", lhs)
                } else if self.mvb.root_lemma() == "panic" {
                    format!("panics!({})", lhs)
                } else if self.mvb.root_lemma() == "occur" {
                    // TODO: nn vbd - needs to be applied backwards
                    //  Overlaps with EVENT case
                    format!("{}s!()", lhs)
                } else {
                    // TODO: Make this an error / resolve cases such as this.
                    format!("({}).{}()", lhs, self.mvb.root_lemma())
                }
            }
        };

        let rb = match &self.mvb {
            MVB::VB(rb, _) => rb.as_ref().map(|x| &x.lemma),
            MVB::VBZ(rb, _) => rb.as_ref().map(|x| &x.lemma),
            MVB::VBP(rb, _) => rb.as_ref().map(|x| &x.lemma),
            MVB::VBN(rb, _) => rb.as_ref().map(|x| &x.lemma),
            MVB::VBG(rb, _) => rb.as_ref().map(|x| &x.lemma),
            MVB::VBD(rb, _) => rb.as_ref().map(|x| &x.lemma),
        };

        if rb.map(|x| is_negation(x)).unwrap_or(false) ^ (is_negation(self.mvb.root_lemma())) {
            Ok(ExprString::String(format!("!{}", s)))
        } else {
            Ok(ExprString::String(s))
        }
    }
}

impl AsCode for Assert {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        let s = self
            .objects
            .iter()
            .map(|objs| {
                objs.iter()
                    .map(Object::as_code)
                    .filter_map(Result::ok)
                    .map(|rhs| self.property.apply(&ExprString::Expr(rhs)))
                    .filter_map(Result::ok)
                    .map(|x| x.to_string())
                    .join(" && ")
            })
            .join(") || (");
        if s.is_empty() {
            return Err(SpecificationError::UnsupportedSpec(
                "No valid elements found in specification",
            ));
        }
        Ok(syn::parse_str(&format!("({})", s))?)
    }
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

pub trait AsSpec {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError>;
}

impl AsSpec for Specification {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        match self {
            Specification::Mret(m) => {
                let e = m.as_code()?;
                let attrs = format!("#[ensures({})]", quote::quote! {#e}.to_string());
                let e: AttrHelper = syn::parse_str(&attrs)?;
                Ok(e.attrs)
            }
            Specification::RetIf(returnif) => returnif.as_spec(),
            Specification::SpecAtom(atom) => atom.as_spec(),
            Specification::SpecCond(cond) => cond.as_spec(),
            Specification::SpecTerm(_) => Err(SpecificationError::Unimplemented),
        }
    }
}

impl AsSpec for SpecAtom {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        match self {
            SpecAtom::HAssert(hassert) => hassert.as_spec(),
            SpecAtom::QAssert(qassert) => qassert.as_spec(),
            SpecAtom::Action(action_obj) => action_obj.as_spec(),
            SpecAtom::Side => Err(SpecificationError::Unimplemented),
        }
    }
}

impl AsSpec for SpecCond {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        let spec = match &self.atom {
            SpecAtom::HAssert(hassert) => hassert.as_code_value(),
            SpecAtom::QAssert(qassert) => qassert.as_code_value(),
            SpecAtom::Action(action_obj) => action_obj.as_code_value(),
            SpecAtom::Side => return Err(SpecificationError::Unimplemented),
        }?;

        let exprs: Result<Vec<_>, _> = spec
            .exprs
            .into_iter()
            .map(|(cmod, e)| self.cond.apply(&e).map(|val| (cmod, val)))
            .collect();
        SpecIntermediate::new(exprs?).as_spec()
    }
}

impl Apply for BoolCond {
    fn apply(&self, lhs: &ExprString) -> Result<ExprString, SpecificationError> {
        let expr = lhs.to_string();
        let s = match &self.value {
            BoolValue::Assert(assert) => {
                let assert = assert.as_code()?;
                let pred = quote::quote! {#assert}.to_string();
                match self.if_expr {
                    IfExpr::If => format!("({} ==> {})", pred, expr),
                    IfExpr::Iff => {
                        format!("({} ==> {}) && ({} ==> {})", pred, expr, expr, pred)
                    }
                }
            }
            BoolValue::QAssert(q) => {
                let mut body = q.as_code_value()?;
                let new_body = match &mut body.body {
                    ExprQBody::String(s) => {
                        let s = std::mem::take(s);
                        ExprQBody::IffExpr(IffExpr {
                            lhs: s,
                            rhs: expr.clone(),
                        })
                    }
                    ExprQBody::IffExpr(IffExpr { lhs, rhs }) => ExprQBody::IffExpr(IffExpr {
                        lhs: std::mem::take(lhs),
                        rhs: format!("{} && {}", rhs, expr),
                    }),
                };
                body.body = new_body;

                match self.if_expr {
                    IfExpr::If => format!("#({})", body.as_code_value()?),
                    IfExpr::Iff => {
                        let mut s = format!("({})", body.as_code_value()?);
                        let mut body = q.as_code_value()?;
                        let new_body = match &mut body.body {
                            ExprQBody::String(s) => ExprQBody::IffExpr(IffExpr {
                                lhs: expr,
                                rhs: std::mem::take(s),
                            }),
                            ExprQBody::IffExpr(IffExpr { lhs, rhs }) => {
                                ExprQBody::IffExpr(IffExpr {
                                    lhs: expr,
                                    rhs: format!("({} && {})", lhs, rhs),
                                })
                            }
                        };
                        body.body = new_body;
                        s.push_str(&format!(" && ({})", body.as_code_value()?));
                        s
                    }
                }
            }
            BoolValue::Code(code) => {
                let code = code.as_code()?;
                let pred = quote::quote! {#code}.to_string();
                match self.if_expr {
                    IfExpr::If => format!("({} ==> {})", pred, expr),
                    IfExpr::Iff => {
                        format!("#({} ==> {}) && ({} ==> {})", pred, expr, expr, pred)
                    }
                }
            }
            BoolValue::Event(event) => {
                let event = event.as_code()?;
                let pred = quote::quote! {#event}.to_string();
                match self.if_expr {
                    IfExpr::If => format!("({} ==> {})", pred, expr),
                    IfExpr::Iff => {
                        format!("({} ==> {}) && ({} ==> {})", pred, expr, expr, pred)
                    }
                }
            }
        };

        Ok(ExprString::String(s))
    }
}

impl AsSpec for ReturnIf {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        let mut exprs = Vec::new();
        for (cond, expr) in &self.ret_pred {
            let ret_val = expr.as_code()?;
            let ret_assert = format!("result == {}", quote::quote! {#ret_val}.to_string());
            exprs.push((
                ConditionModifier::Post,
                cond.apply(&ExprString::Expr(syn::parse_str(&ret_assert)?))?,
            ));
        }
        SpecIntermediate::new(exprs).as_spec()
    }
}

pub enum ExprString {
    Expr(Expr),
    String(String),
}

impl Display for ExprString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprString::Expr(e) => f.write_str(&quote::quote! {#e}.to_string()),
            ExprString::String(s) => f.write_str(s),
        }
    }
}

pub struct SpecIntermediate {
    pub exprs: Vec<(ConditionModifier, ExprString)>,
}

impl SpecIntermediate {
    fn new(exprs: Vec<(ConditionModifier, ExprString)>) -> Self {
        Self { exprs }
    }
}

impl AsSpec for SpecIntermediate {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        let e: AttrHelper = syn::parse_str(
            &self
                .exprs
                .iter()
                .map(|(cmod, e)| match e {
                    ExprString::Expr(e) => {
                        format!("#[{}({})]", cmod.keyword(), quote::quote! {#e}.to_string())
                    }
                    ExprString::String(s) => format!("#[{}({})]", cmod.keyword(), s),
                })
                .join("\n"),
        )?;
        Ok(e.attrs)
    }
}

impl AsCodeValue for HardAssert {
    type Output = SpecIntermediate;

    fn as_code_value(&self) -> Result<Self::Output, SpecificationError> {
        let spec_scope = ConditionModifier::from_str(&self.md.lemma)
            .map_err(|_| SpecificationError::UnsupportedSpec("Bad condition"))?;
        let assert = self.assert.as_code()?;
        Ok(SpecIntermediate::new(vec![(
            spec_scope,
            ExprString::Expr(assert),
        )]))
    }
}

impl AsSpec for HardAssert {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        self.as_code_value()?.as_spec()
    }
}

impl AsCodeValue for HardQuantAssert {
    type Output = SpecIntermediate;

    fn as_code_value(&self) -> Result<Self::Output, SpecificationError> {
        let expr = match &self.assertion {
            QuantItem::Code(c) => c.as_code()?,
            QuantItem::Assert(h) => h.assert.as_code()?,
        };

        Ok(SpecIntermediate::new(vec![(
            self.spec_scope(),
            ExprString::String(assert_as_code_value(&self.quant_expr, &expr)?.as_code_value()?),
        )]))
    }
}

impl AsSpec for HardQuantAssert {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        self.as_code_value()?.as_spec()
    }
}

pub struct IffExpr {
    lhs: String,
    rhs: String,
}

pub enum ExprQBody {
    String(String),
    IffExpr(IffExpr),
}

pub struct ExprQuantifierU {
    universal: bool,
    bound_vars: Vec<(String, String)>,
    body: ExprQBody,
}

impl ExprQBody {
    fn flip(&mut self) {
        match self {
            ExprQBody::String(_s) => {}
            ExprQBody::IffExpr(IffExpr { lhs, rhs }) => std::mem::swap(lhs, rhs),
        }
    }
}

impl ExprQuantifierU {
    fn flip(&mut self) {
        self.body.flip()
    }
}

fn assert_as_code_value(
    quant_expr: &QuantExpr,
    expr: &Expr,
) -> Result<ExprQuantifierU, SpecificationError> {
    Ok(match &quant_expr.range {
        None => {
            let ident = quant_expr.quant.obj.as_code()?;
            ExprQuantifierU {
                universal: quant_expr.quant.universal,
                bound_vars: vec![(quote::quote! {#ident}.to_string(), "int".to_string())],
                body: ExprQBody::String(quote::quote! {#expr}.to_string()),
            }
        }
        Some(range) => {
            let ident = match &range.range.ident {
                Some(o) => o,
                None => &quant_expr.quant.obj,
            }
            .as_code()?;
            let start = match &range.range.start {
                // TODO: Type resolution and appropriate minimum detection here.
                //     Ensure that we check specifically for numerical types which Prusti supports.
                //     Also, check that start < end
                None => syn::parse_str("0")?,
                Some(x) => x.as_code()?,
            };
            let cmp = range.upper_bound.lt().to_string();
            let end = range.range.end.as_ref().unwrap().as_code()?;
            let conditions = vec![vec![
                format!(
                    "{} <= {}",
                    quote::quote! {#start}.to_string(),
                    quote::quote! {#ident}.to_string()
                ),
                format!(
                    "{} {} {}",
                    quote::quote! {#ident}.to_string(),
                    cmp,
                    quote::quote! {#end}.to_string()
                ),
            ]];

            let conditions = conditions
                .into_iter()
                .map(|x| x.join(" && "))
                .join(") || (");
            ExprQuantifierU {
                universal: quant_expr.quant.universal,
                bound_vars: vec![(quote::quote! {#ident}.to_string(), "int".to_string())],
                body: ExprQBody::IffExpr(IffExpr {
                    lhs: format!("({})", conditions),
                    rhs: quote::quote! {#expr}.to_string(),
                }),
            }
        }
    })
}

impl AsCodeValue for QuantAssert {
    type Output = ExprQuantifierU;

    fn as_code_value(&self) -> Result<Self::Output, SpecificationError> {
        let expr = match &self.assertion {
            QuantItem::Code(c) => c.as_code()?,
            QuantItem::Assert(assert) => assert.as_code()?,
        };
        assert_as_code_value(&self.quant_expr, &expr)
    }
}

impl AsCodeValue for ExprQuantifierU {
    type Output = String;

    fn as_code_value(&self) -> Result<Self::Output, SpecificationError> {
        let quant = if self.universal { "forall" } else { "forsome" };
        let idents = self
            .bound_vars
            .iter()
            .map(|(name, ty)| format!("{}: {}", name, ty))
            .join(", ");
        Ok(match &self.body {
            ExprQBody::String(s) => format!("{}(|{}| {})", quant, idents, s),
            ExprQBody::IffExpr(IffExpr { lhs, rhs }) => {
                format!("{}(|{}| ({}) ==> ({}))", quant, idents, lhs, rhs)
            }
        })
    }
}

impl AsCodeValue for ActionObj {
    type Output = SpecIntermediate;
    fn as_code_value(&self) -> Result<Self::Output, SpecificationError> {
        match self {
            ActionObj::Action1(action, target) => {
                let target = target.as_code()?;
                Ok(SpecIntermediate::new(vec![(
                    ConditionModifier::Post,
                    ExprString::String(format!(
                        "(called!({}({})))",
                        action.lemma,
                        quote::quote! {#target}.to_string()
                    )),
                )]))
            }
            ActionObj::Action2Ambiguous(action, target, value) => {
                // Should be performing compiler steps here to determine what is target and what
                // is value
                let target = target.as_code()?;
                let value = value.as_code()?;
                Ok(SpecIntermediate::new(vec![(
                    ConditionModifier::Post,
                    ExprString::String(format!(
                        "(called!(({}).{}({})))",
                        quote::quote! { #target }.to_string(),
                        action.lemma,
                        quote::quote! { #value }.to_string()
                    )),
                )]))
            }
            ActionObj::Action2Resolved { vbz, target, value } => {
                let target = target.as_code()?;
                let value = value.as_code()?;
                Ok(SpecIntermediate::new(vec![(
                    ConditionModifier::Post,
                    ExprString::String(format!(
                        "(called!(({}).{}({})))",
                        quote::quote! { #target }.to_string(),
                        vbz.lemma,
                        quote::quote! { #value }.to_string()
                    )),
                )]))
            }
            ActionObj::Set { target, value } => {
                let target = target.as_code()?;
                let value = value.as_code()?;
                Ok(SpecIntermediate::new(vec![(
                    ConditionModifier::Post,
                    ExprString::String(quote::quote! {(#target) == (#value)}.to_string()),
                )]))
            }
        }
    }
}

impl AsSpec for ActionObj {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        self.as_code_value()?.as_spec()
    }
}

impl Apply for Relation {
    fn apply(&self, lhs: &ExprString) -> Result<ExprString, SpecificationError> {
        let op = self.op.to_string();
        let s = self
            .objects
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(Object::as_code)
                    .filter_map(Result::ok)
                    .map(|rhs| format!("(({}) {} ({}))", lhs, op, quote::quote! {#rhs}.to_string()))
                    .join(" && ")
            })
            .join(") || (");
        if self
            .modifier
            .as_ref()
            .map(|x| is_negation(&x.lemma))
            .unwrap_or(false)
        {
            Ok(ExprString::String(format!("!(({}))", s)))
        } else {
            Ok(ExprString::String(format!("(({}))", s)))
        }
    }
}

// TODO: Build a tool to simplify brackets and !
