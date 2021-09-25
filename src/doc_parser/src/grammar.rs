use itertools::Itertools;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Expr};

use crate::nl_ir::{
    Assert, BoolValue, Code, Event, HardAssert, IfExpr, IsPropMod, IsProperty, Lemma, Literal,
    MReturn, MnnMod, Object, Op, QuantAssert, QuantItem, ReturnIf, Specification,
};
use crate::parse_tree::tree::{MJJ, MVB};

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
            Object::Prp(_) => {
                println!("Prp");
                Err(SpecificationError::Unimplemented)
            }
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
        println!("Event");
        Err(SpecificationError::Unimplemented)
    }
}

trait Apply {
    fn apply(&self, rhs: &Expr) -> Result<String, SpecificationError>;
}

impl Apply for IsProperty {
    /// TODO: This is somewhat expensive, and when properties are tested repeatedly, it would be nice to
    ///     avoid repeating work. Benchmark whether this is an issue in reality.
    fn apply(&self, lhs: &Expr) -> Result<String, SpecificationError> {
        let s = match &self.prop_type {
            IsPropMod::Obj(rhs) => {
                if !["is", "be"].contains(&self.mvb.root_lemma()) {
                    return Err(SpecificationError::UnsupportedSpec(
                        "Relationships between objects must be of the is or be format",
                    ));
                };

                let rhs = rhs.as_code()?;
                let s = quote::quote! {(#lhs == #rhs)}.to_string();
                s
            }
            IsPropMod::Mjj(mjj) => {
                let (rb, lemma) = match mjj {
                    MJJ::JJ(rb, jj) => (rb.as_ref(), &jj.lemma),
                    MJJ::JJR(rb, jj) => (rb.as_ref(), &jj.lemma),
                    MJJ::JJS(rb, jj) => (rb.as_ref(), &jj.lemma),
                };
                let sym = if rb.map(|x| x.lemma == "not").unwrap_or(false) {
                    "!"
                } else {
                    ""
                };
                if ["modified", "altered", "changed"].contains(&lemma.as_str()) {
                    let lhs = quote::quote! {#lhs}.to_string();
                    format!("{}({} != old({}))", sym, lhs, lhs)
                } else if ["modify", "alter", "change"].contains(&lemma.as_str()) {
                    let lhs = quote::quote! {#lhs}.to_string();
                    format!("{}({} == old({}))", sym, lhs, lhs)
                } else {
                    // TODO: Make this an error / resolve cases such as this.
                    format!("{}({}).{}()", sym, quote::quote! {#lhs}.to_string(), lemma)
                }
            }
            IsPropMod::Rel(_rel) => {
                // return ModRelation(self.tree[-1]).as_code(lhs)
                println!("PropMod::REL");
                return Err(SpecificationError::Unimplemented);
            }
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
                            quote::quote! {#lhs}.to_string(),
                            quote::quote! {#lhs}.to_string(),
                            upper_bound,
                            quote::quote! {#end}.to_string(),
                        )
                    }
                    _ => return Err(SpecificationError::UnsupportedSpec("PROP: RANGE case")),
                }
            }
            IsPropMod::None => {
                if ["modify", "alter", "change"].contains(&self.mvb.root_lemma()) {
                    let lhs = quote::quote! {#lhs}.to_string();
                    format!("({} != old({}))", lhs, lhs)
                } else if self.mvb.root_lemma() == "overflow" {
                    format!("overflows!({})", quote::quote! {#lhs}.to_string())
                } else {
                    // TODO: Make this an error / resolve cases such as this.
                    format!(
                        "({}).{}()",
                        quote::quote! {#lhs}.to_string(),
                        self.mvb.root_lemma()
                    )
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

        if rb.map(|x| x == "not").unwrap_or(false) ^ (self.mvb.root_lemma() == "not") {
            Ok(format!("!{}", s))
        } else {
            Ok(s)
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
                    .map(|rhs| self.property.apply(&rhs))
                    .filter_map(Result::ok)
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
            Specification::HAssert(hassert) => hassert.as_spec(),
            Specification::QAssert(qassert) => qassert.as_spec(),
            Specification::RetIf(returnif) => returnif.as_spec(),
            Specification::Side => Err(SpecificationError::Unimplemented),
        }
    }
}

impl AsSpec for HardAssert {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        let cond = if self.md.lemma == "will" {
            "ensures"
        } else {
            "requires"
        };
        let assert = self.assert.as_code()?;
        let e: AttrHelper = syn::parse_str(&format!(
            "#[{}({})]",
            cond,
            quote::quote! {#assert}.to_string()
        ))?;
        Ok(e.attrs)
    }
}

impl AsSpec for QuantAssert {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        let scope = if self.is_precond() {
            "requires"
        } else {
            "ensures"
        };

        let attr = format!("#[{}({})]", scope, self.as_code_value()?.as_code_value()?);
        let e: AttrHelper = syn::parse_str(&attr)?;
        Ok(e.attrs)
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

impl AsCodeValue for QuantAssert {
    type Output = ExprQuantifierU;

    fn as_code_value(&self) -> Result<Self::Output, SpecificationError> {
        let expr = match &self.assertion {
            QuantItem::Code(c) => c.as_code()?,
            QuantItem::HAssert(h) => h.assert.as_code()?,
        };

        Ok(match &self.quant_expr.range {
            None => {
                let ident = self.quant_expr.quant.obj.as_code()?;
                ExprQuantifierU {
                    universal: self.quant_expr.quant.universal,
                    bound_vars: vec![(quote::quote! {#ident}.to_string(), "int".to_string())],
                    body: ExprQBody::String(quote::quote! {#expr}.to_string()),
                }
            }
            Some(range) => {
                let ident = match &range.range.ident {
                    Some(o) => o,
                    None => &self.quant_expr.quant.obj,
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
                    universal: self.quant_expr.quant.universal,
                    bound_vars: vec![(quote::quote! {#ident}.to_string(), "int".to_string())],
                    body: ExprQBody::IffExpr(IffExpr {
                        lhs: format!("({})", conditions),
                        rhs: quote::quote! {#expr}.to_string(),
                    }),
                }
            }
        })
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

impl AsSpec for ReturnIf {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        let mut attrs = String::new();
        for (cond, val) in &self.ret_pred {
            let ret_val = val.as_code()?;
            let ret_assert = format!("result == {}", quote::quote! {#ret_val}.to_string());
            let s = match &cond.value {
                BoolValue::Assert(assert) => {
                    let assert = assert.as_code()?;
                    let pred = quote::quote! {#assert}.to_string();
                    match cond.if_expr {
                        IfExpr::If => format!("#[ensures({} ==> {})]", pred, ret_assert),
                        IfExpr::Iff => {
                            format!(
                                "#[ensures({} ==> {})]\n#[ensures({} ==> {})]",
                                pred, ret_assert, ret_assert, pred
                            )
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
                                rhs: ret_assert.clone(),
                            })
                        }
                        ExprQBody::IffExpr(IffExpr { lhs, rhs }) => ExprQBody::IffExpr(IffExpr {
                            lhs: std::mem::take(lhs),
                            rhs: format!("{} && {}", rhs, ret_assert),
                        }),
                    };
                    body.body = new_body;

                    match cond.if_expr {
                        IfExpr::If => format!("#[ensures({})]", body.as_code_value()?),
                        IfExpr::Iff => {
                            let mut s = format!("#[ensures({})]", body.as_code_value()?);
                            let mut body = q.as_code_value()?;
                            let new_body = match &mut body.body {
                                ExprQBody::String(s) => ExprQBody::IffExpr(IffExpr {
                                    lhs: ret_assert,
                                    rhs: std::mem::take(s),
                                }),
                                ExprQBody::IffExpr(IffExpr { lhs, rhs }) => {
                                    ExprQBody::IffExpr(IffExpr {
                                        lhs: ret_assert,
                                        rhs: format!("({} && {})", lhs, rhs),
                                    })
                                }
                            };
                            body.body = new_body;
                            s.push_str(&format!("\n#[ensures({})]", body.as_code_value()?));
                            s
                        }
                    }
                }
                BoolValue::Code(code) => {
                    let code = code.as_code()?;
                    let pred = quote::quote! {#code}.to_string();
                    match cond.if_expr {
                        IfExpr::If => format!("#[ensures({} ==> {})]", pred, ret_assert),
                        IfExpr::Iff => {
                            format!(
                                "#[ensures({} ==> {})]\n#[ensures({} ==> {})]",
                                pred, ret_assert, ret_assert, pred
                            )
                        }
                    }
                }
                BoolValue::Event(_) => {
                    // s = "!" if isinstance(pred, Negated) else ""
                    // overflow_item = pred.expr.root or ret_val
                    // if pred.expr.resolve() == EventType.OVERFLOW:
                    //     ret_assert = f"{s}overflows!({overflow_item.as_code()}) ==> (result == {ret_val})"
                    // elif pred.expr.resolve() == EventType.NO_OVERFLOW:
                    //     ret_assert = f"{s}!overflows!({overflow_item.as_code()}) ==> (result == {ret_val})"
                    // return f"#[ensures({ret_assert})]"
                    println!("EVENT");
                    return Err(SpecificationError::Unimplemented);
                }
            };
            attrs.push_str(&s);
            attrs.push('\n');
        }
        let e: AttrHelper = syn::parse_str(&attrs)?;
        Ok(e.attrs)
    }
}

// TODO: Build a tool to simplify brackets and !
