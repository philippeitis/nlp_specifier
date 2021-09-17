use syn::{Type, FnArg, ReturnType};
use crate::search_tree::{SearchItem, SearchValue};

pub enum FnArgLocation {
    OutputIndex(usize),
    InputIndex(usize),
    Output,
    Input,
}

pub trait TypeMatch {
    fn ty_matches(&self, ty: &Type) -> bool;
}

impl TypeMatch for &str {
    fn ty_matches(&self, ty: &Type) -> bool {
        match ty {
            Type::Reference(ty) => if self.ty_matches(&ty.elem) {
                return true
            }
            _ => {},
        }
        self == &(quote::quote! {#ty}.to_string())
    }
}

impl TypeMatch for Type {
    fn ty_matches(&self, ty: &Type) -> bool {
        self == ty
    }
}

pub struct HasFnArg {
    pub fn_arg_location: FnArgLocation,
    pub fn_arg_type: Box<dyn TypeMatch>,
}

impl HasFnArg {
    pub(crate) fn item_matches(&self, item: &SearchValue) -> bool {
        match &item.item {
            SearchItem::Fn(item) => match self.fn_arg_location {
                FnArgLocation::OutputIndex(i) => {}
                FnArgLocation::InputIndex(i) => {
                    if i >= item.sig.inputs.len() {
                        return false;
                    }
                    return match &item.sig.inputs[i] {
                        FnArg::Receiver(_) => false,
                        FnArg::Typed(val) => self.fn_arg_type.ty_matches(&val.ty),
                    };
                }
                FnArgLocation::Output => return match &item.sig.output {
                    ReturnType::Default => false,
                    ReturnType::Type(_, ty) => if self.fn_arg_type.ty_matches(ty.as_ref()) {
                        true
                    } else if let Type::Tuple(val) = ty.as_ref() {
                        val.elems.iter().any(|t| self.fn_arg_type.ty_matches(t))
                    } else {
                        false
                    }
                },
                FnArgLocation::Input => for arg in item.sig.inputs.iter() {
                    match &arg {
                        FnArg::Receiver(x) => {}
                        FnArg::Typed(val) => {
                            if self.fn_arg_type.ty_matches(&val.ty) {
                                return true;
                            }
                        }
                    }
                }
            }
            SearchItem::Method(item) => match self.fn_arg_location {
                FnArgLocation::OutputIndex(i) => {}
                FnArgLocation::InputIndex(i) => {
                    if i >= item.sig.inputs.len() {
                        return false;
                    }
                    match &item.sig.inputs[i] {
                        FnArg::Receiver(_) => {}
                        FnArg::Typed(val) => {
                            if self.fn_arg_type.ty_matches(&val.ty) {
                                return true;
                            }
                        }
                    }
                }
                FnArgLocation::Output => return match &item.sig.output {
                    ReturnType::Default => false,
                    ReturnType::Type(_, ty) => if self.fn_arg_type.ty_matches(ty.as_ref()) {
                        true
                    } else if let Type::Tuple(val) = ty.as_ref() {
                        val.elems.iter().any(|t| self.fn_arg_type.ty_matches(t))
                    } else {
                        false
                    }
                },
                FnArgLocation::Input => for arg in item.sig.inputs.iter() {
                    match &arg {
                        FnArg::Receiver(x) => {}
                        FnArg::Typed(val) => {
                            if self.fn_arg_type.ty_matches(&val.ty) {
                                return true;
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        return false;
    }
}

