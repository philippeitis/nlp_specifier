use itertools::Itertools;

use syn::{Expr, Attribute, Error};
use syn::parse::{Parse, ParseStream};
use serde_json::to_string;

use crate::nl_ir::{Op, Code, Literal, Object, MReturn, Assert, Event, Specification, QuantAssert, QuantItem, IsPropMod, Lemma, BinOp, HardAssert, RangeMod};
use crate::parse_tree::tree::MVB;

#[derive(Debug)]
pub enum SpecificationError {
    Syn(syn::Error),
    UnsupportedSpec(&'static str),
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

        Ok(syn::parse_str(&quote::quote! {(#lhs) #op (#rhs)}.to_string())?)
    }
}


impl AsCode for Object {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        match self {
            Object::Code(c) => c.as_code(),
            Object::Lit(l) => l.as_code(),
            Object::Op(o) => o.as_code(),
            _ => return Err(SpecificationError::UnsupportedSpec("Other types of object not supported"))
        }
    }
}


impl AsCode for MReturn {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        let val = self.ret_val.as_code()?;
        Ok(syn::parse_str(&quote::quote! {result == (#val)}.to_string())?)
    }
}


impl AsCode for Event {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        Err(SpecificationError::UnsupportedSpec("Events not supported"))
    }
}

impl AsCode for Assert {
    fn as_code(&self) -> Result<Expr, SpecificationError> {
        match &self.property.prop_type {
            IsPropMod::Obj(obj) => {
                if !["is", "be"].contains(&self.property.mvb.root_lemma()) {
                    panic!("unexpected string in assert {}", self.property.mvb.root_lemma());
                };

                let (rb, lemma) = match &self.property.mvb {
                    MVB::VB(rb, vb) => (rb.as_ref().map(|x| &x.lemma), &vb.lemma),
                    MVB::VBZ(rb, vb) => (rb.as_ref().map(|x| &x.lemma), &vb.lemma),
                    MVB::VBP(rb, vb) => (rb.as_ref().map(|x| &x.lemma), &vb.lemma),
                    MVB::VBN(rb, vb) => (rb.as_ref().map(|x| &x.lemma), &vb.lemma),
                    MVB::VBG(rb, vb) => (rb.as_ref().map(|x| &x.lemma), &vb.lemma),
                    MVB::VBD(rb, vb) => (rb.as_ref().map(|x| &x.lemma), &vb.lemma),
                };
                // if only 1 negation, result is true, ow false.
                let is_negated = rb.map(|x| x == "not").unwrap_or(false)
                    ^ (self.property.mvb.root_lemma() == "not");
                let rhs = obj.as_code()?;
                let op = syn::parse_str::<syn::BinOp>(if is_negated {
                    &"!="
                } else {
                    &"=="
                })?;

                let s = self.objects.iter().map(|objs|
                    objs
                        .iter()
                        .map(Object::as_code)
                        .filter_map(Result::ok)
                        .map(|lhs| quote::quote! {(#lhs) #op (#rhs)}.to_string()).join(" && ")
                ).join(") || (");
                Ok(syn::parse_str(&s)?)
            }
            _ => Err(SpecificationError::UnsupportedSpec("Assert variants not supported")),
        }
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
            _ => unimplemented!(),
        }
    }
}

impl AsSpec for HardAssert {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        let cond = if self.md.lemma == "will" {
            "requires"
        } else {
            "ensures"
        };
        let assert = self.assert.as_code()?;
        let e: AttrHelper = syn::parse_str(&format!("#[{}({})]", cond, quote::quote! {#assert}.to_string()))?;
        Ok(e.attrs)
    }
}

impl AsSpec for QuantAssert {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        let is_precond = match &self.assertion {
            QuantItem::Code(c) => false,
            QuantItem::HAssert(h) => h.md.lemma == "must",
        };

        let scope = if is_precond {
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
    IffExpr(IffExpr)
}

pub struct ExprQuantifierU {
    universal: bool,
    bound_vars: Vec<(String, String)>,
    body: ExprQBody,
}

impl ExprQBody {
    fn flip(&mut self) {
        match self {
            ExprQBody::String(s) => {},
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
                    body: ExprQBody::String(quote::quote! {#expr}.to_string())
                }
            }
            Some(range) => {
                let ident = match &range.range.ident {
                    Some(o) => o,
                    None => &self.quant_expr.quant.obj,
                }.as_code()?;
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
                    format!("{} <= {}", quote::quote! {#start}.to_string(), quote::quote! {#ident}.to_string()),
                    format!("{} {} {}", quote::quote! {#ident}.to_string(), cmp, quote::quote! {#end}.to_string())
                ]];

                let conditions = conditions.into_iter().map(|x| x.join(" && ")).join(") || (");
                ExprQuantifierU {
                    universal: self.quant_expr.quant.universal,
                    bound_vars: vec![(quote::quote! {#ident}.to_string(), "int".to_string())],
                    body: ExprQBody::IffExpr(IffExpr {
                        lhs: conditions,
                        rhs: quote::quote! {#expr}.to_string()
                    })
                }
            },
        })
    }
}

impl AsCodeValue for ExprQuantifierU {
    type Output = String;

    fn as_code_value(&self) -> Result<Self::Output, SpecificationError> {
        let quant = if self.universal {
            "forall"
        } else {
            "forsome"
        };
        let idents = self.bound_vars.iter().map(|(name, ty)| format!("{}: {}", name, ty)).join(", ");
        Ok(match &self.body {
            ExprQBody::String(s) => format!("{}(|{}| {})", quant, idents, s),
            ExprQBody::IffExpr(IffExpr { lhs, rhs }) => format!("{}(|{}| ({}) ==> ({}))", quant, idents, lhs, rhs),
        })
    }
}
