use syn::{Expr, Attribute, Error};

use crate::sir::{Op, Code, Literal, Object, MReturn, Assert, Event, Specification, QuantAssert, QuantItem};
use syn::parse::{Parse, ParseStream};

pub trait AsCode {
    fn as_code(&self) -> Expr;
}

impl AsCode for Code {
    fn as_code(&self) -> Expr {
        syn::parse_str(self.code.trim_matches('`')).unwrap()
    }
}

impl AsCode for Literal {
    fn as_code(&self) -> Expr {
        syn::parse_str(self.s.trim_matches('`')).unwrap()
    }
}


impl AsCode for Op {
    fn as_code(&self) -> Expr {
        let lhs = self.lhs.as_code();
        let op = syn::parse_str::<syn::BinOp>(&self.op.to_string()).unwrap();
        let rhs = self.rhs.as_code();

        syn::parse_str(&quote::quote! {(#lhs) #op (#rhs)}.to_string()).unwrap()
    }
}


impl AsCode for Object {
    fn as_code(&self) -> Expr {
        match self {
            Object::Code(c) => c.as_code(),
            Object::Lit(l) => l.as_code(),
            Object::Op(o) => o.as_code(),
            _ => unimplemented!()
        }
    }
}


impl AsCode for MReturn {
    fn as_code(&self) -> Expr {
        let val = self.ret_val.as_code();
        syn::parse_str(&quote::quote! {result == (#val)}.to_string()).unwrap()
    }
}



impl AsCode for Event {
    fn as_code(&self) -> Expr {
        unimplemented!()
    }
}

impl AsCode for Assert {
    fn as_code(&self) -> Expr {
        unimplemented!()
    }
}

struct AttrHelper {
    attrs: Vec<Attribute>,
}

pub trait AsSpec {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError>;
}

impl Parse for AttrHelper {
    fn parse(input: ParseStream) -> Result<Self, syn::Error> {
        Ok(AttrHelper {
            attrs: input.call(Attribute::parse_outer)?,
        })
    }
}

#[derive(Debug)]
pub enum SpecificationError {
    Syn(syn::Error)
}

impl From<syn::Error> for SpecificationError {
    fn from(err: syn::Error) -> Self {
        SpecificationError::Syn(err)
    }
}

impl AsSpec for Specification {
    fn as_spec(&self) -> Result<Vec<Attribute>, SpecificationError> {
        Ok(match self {
            Specification::Mret(m) => {
                let e = m.as_code();
                let attrs = format!("#[ensures({})]", quote::quote! {#e}.to_string());
                let e: AttrHelper = syn::parse_str(&attrs)?;
                e.attrs
            },
            _ => unimplemented!(),
        })
    }
}