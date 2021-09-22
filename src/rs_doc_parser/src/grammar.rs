use syn::{Expr, Attribute};

use crate::sir::{Op, Code, Literal, Object, MReturn, Negated, Negatable, Assert, Event, Specification};
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
            Object::PropOf(_) => unimplemented!()
        }
    }
}


impl AsCode for MReturn {
    fn as_code(&self) -> Expr {
        let val = self.ret_val.as_code();
        syn::parse_str(&quote::quote! {result == (#val)}.to_string()).unwrap()
    }
}

impl AsCode for Negatable {
    fn as_code(&self) -> Expr {
        match self {
            Negatable::Assert(a) => a.as_code(),
            Negatable::Code(c) => c.as_code(),
            Negatable::Event(e) => e.as_code(),
        }
    }
}

impl AsCode for Negated {
    fn as_code(&self) -> Expr {
        let val = self.expr.as_code();
        syn::parse_str(&quote::quote! {!(#val)}.to_string()).unwrap()
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
    fn as_spec(&self) -> Vec<Attribute>;
}

impl Parse for AttrHelper {
    fn parse(input: ParseStream) -> Result<Self, syn::Error> {
        Ok(AttrHelper {
            attrs: input.call(Attribute::parse_outer)?,
        })
    }
}


impl AsSpec for Specification {
    fn as_spec(&self) -> Vec<Attribute> {
        match self {
            Specification::Mret(m) => {
                let e = m.as_code();
                let attrs = format!("#[ensures({})]", quote::quote! {#e}.to_string());
                let e: AttrHelper = syn::parse_str(&attrs).unwrap();
                e.attrs
            },
            _ => unimplemented!(),
        }
    }
}