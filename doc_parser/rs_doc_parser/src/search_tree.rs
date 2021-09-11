use crate::docs::Docs;
use std::convert::TryFrom;

use std::num::NonZeroUsize;
use syn::{ItemConst, ItemEnum, ItemFn, ItemStruct, ImplItemConst, ImplItemMethod, Item, Attribute, Generics, Path, Type, ImplItem, ItemImpl, Visibility, ItemMod, File, FnArg};
use std::fmt::{Display, Formatter, Debug};
use quote::ToTokens;

pub enum SearchItem {
    Const(Docs, ItemConst),
    Enum(Docs, ItemEnum),
    // ExternCrate(ItemExternCrate),
    Fn(Docs, ItemFn),
    // ForeignMod(Doc, ItemForeignMod),
    Impl(Docs, SearchItemImpl),
    // Macro(ItemMacro),
    // Macro2(ItemMacro2),
    Mod(Docs, SearchItemMod),
    // Static(ItemStatic),
    Struct(Docs, ItemStruct),
    // Trait(ItemTrait),
    // TraitAlias(ItemTraitAlias),
    // Type(ItemType),
    // Union(ItemUnion),
    // Use(ItemUse),
    // Verbatim(TokenStream),
    ImplConst(Docs, ImplItemConst),
    Method(Docs, ImplItemMethod),
    // ImplType(Docs, ImplItemType),
    // ImplMacro(Docs, ImplItemMacro),
    // ImplVerbatim(Docs, TokenStream),
    // some variants omitted
}

impl Display for SearchItem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchItem::Const(_, item) => {
                write!(f, "{}", item.ident)
            }
            SearchItem::Enum(_, item) => {
                write!(f, "{}", item.ident)
            }
            SearchItem::Fn(_, item) => {
                write!(f, "{}", item.sig.to_token_stream().to_string())
            }
            SearchItem::Impl(_, item) => {
                write!(f, "{}", item.self_ty.to_token_stream().to_string())
            }
            SearchItem::Mod(_, item) => {
                write!(f, "{}", item.ident.to_string())
            }
            SearchItem::Struct(_, item) => {
                write!(f, "{}", item.ident.to_string())
            }
            SearchItem::ImplConst(_, item) => {
                write!(f, "{}", item.ident.to_string())
            }
            SearchItem::Method(_, item) => {
                write!(f, "{}", item.sig.to_token_stream().to_string())
            }
        }
    }
}

impl Debug for SearchItem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
impl TryFrom<&Item> for SearchItem {
    type Error = ();

    fn try_from(item: &Item) -> Result<Self, Self::Error> {
        match item {
            Item::Const(item) => {
                Ok(SearchItem::Const(Docs::from(&item.attrs), item.clone()))
            }
            Item::Enum(item) => {
                Ok(SearchItem::Enum(Docs::from(&item.attrs), item.clone()))
            }
            // Item::ExternCrate(_) => {}
            Item::Fn(item) => {
                Ok(SearchItem::Fn(Docs::from(&item.attrs), item.clone()))
            }
            // Item::ForeignMod(_) => {}
            Item::Impl(item) => {
                Ok(SearchItem::Impl(Docs::from(&item.attrs), SearchItemImpl::from(item)))
                // Ok(SearchItem::Impl(Docs::from(&item.attrs), item.clone()))
            }
            // Item::Macro(_) => {}
            // Item::Macro2(_) => {}
            Item::Mod(item) => {
                Ok(SearchItem::Mod(Docs::from(&item.attrs), SearchItemMod::from(item)))
            }
            // Item::Static(_) => {}
            Item::Struct(item) => {
                Ok(SearchItem::Struct(Docs::from(&item.attrs), item.clone()))
            }
            // Item::Trait(_) => {}
            // Item::TraitAlias(_) => {}
            // Item::Type(_) => {}
            // Item::Union(_) => {}
            // Item::Use(_) => {}
            // Item::Verbatim(_) => {}
            // Item::__TestExhaustive(_) => {}
            _ => Err(())
        }
    }
}

impl TryFrom<Item> for SearchItem {
    type Error = ();

    fn try_from(item: Item) -> Result<Self, Self::Error> {
        match item {
            Item::Const(item) => {
                Ok(SearchItem::Const(Docs::from(&item.attrs), item))
            }
            Item::Enum(item) => {
                Ok(SearchItem::Enum(Docs::from(&item.attrs), item))
            }
            // Item::ExternCrate(_) => {}
            Item::Fn(item) => {
                Ok(SearchItem::Fn(Docs::from(&item.attrs), item))
            }
            // Item::ForeignMod(_) => {}
            Item::Impl(item) => {
                Ok(SearchItem::Impl(Docs::from(&item.attrs), SearchItemImpl::from(&item)))
                // Ok(SearchItem::Impl(Docs::from(&item.attrs), item.clone()))
            }
            // Item::Macro(_) => {}
            // Item::Macro2(_) => {}
            Item::Mod(item) => {
                Ok(SearchItem::Mod(Docs::from(&item.attrs), SearchItemMod::from(&item)))
            }
            // Item::Static(_) => {}
            Item::Struct(item) => {
                Ok(SearchItem::Struct(Docs::from(&item.attrs), item))
            }
            // Item::Trait(_) => {}
            // Item::TraitAlias(_) => {}
            // Item::Type(_) => {}
            // Item::Union(_) => {}
            // Item::Use(_) => {}
            // Item::Verbatim(_) => {}
            // Item::__TestExhaustive(_) => {}
            _ => Err(())
        }
    }
}


#[derive(Debug)]
pub struct SearchItemImpl {
    pub attrs: Vec<Attribute>,
    pub defaultness: bool,
    pub unsafety: bool,
    pub generics: Generics,
    pub trait_: Option<Path>,
    pub self_ty: Box<Type>,
    pub items: Vec<SearchItem>,
}

impl TryFrom<&ImplItem> for SearchItem {
    type Error = ();

    fn try_from(item: &ImplItem) -> Result<Self, Self::Error> {
        match item {
            ImplItem::Const(item) => {
                Ok(SearchItem::ImplConst(Docs::from(&item.attrs), item.clone()))
            }
            ImplItem::Method(item) => {
                Ok(SearchItem::Method(Docs::from(&item.attrs), item.clone()))
            }
            _ => Err(())
        }
    }
}

impl TryFrom<ImplItem> for SearchItem {
    type Error = ();

    fn try_from(item: ImplItem) -> Result<Self, Self::Error> {
        match item {
            ImplItem::Const(item) => {
                Ok(SearchItem::ImplConst(Docs::from(&item.attrs), item))
            }
            ImplItem::Method(item) => {
                Ok(SearchItem::Method(Docs::from(&item.attrs), item))
            }
            _ => Err(())
        }
    }
}



impl From<&ItemImpl> for SearchItemImpl {
    fn from(impl_item: &ItemImpl) -> Self {
        SearchItemImpl {
            attrs: impl_item.attrs.clone(),
            defaultness: impl_item.defaultness.is_some(),
            unsafety: impl_item.unsafety.is_some(),
            generics: impl_item.generics.clone(),
            trait_: impl_item.trait_.as_ref().map(|(_, path, _)| path.clone()),
            self_ty: impl_item.self_ty.clone(),
            items: impl_item.items.iter().map(SearchItem::try_from).filter_map(Result::ok).collect(),
        }
    }
}

#[derive(Debug)]
pub struct SearchItemMod {
    pub docs: Docs,
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub ident: String,
    pub content: Option<Vec<SearchItem>>,
    pub semi: bool,
}

impl From<&ItemMod> for SearchItemMod {
    fn from(mod_item: &ItemMod) -> Self {
        let docs = Docs::from(&mod_item.attrs);
        let content = mod_item.content.as_ref().map(
            |(_, content)| content.iter().map(SearchItem::try_from).filter_map(Result::ok).collect()
        );
        SearchItemMod {
            docs,
            attrs: mod_item.attrs.clone(),
            vis: mod_item.vis.clone(),
            ident: mod_item.ident.to_string(),
            content,
            semi: mod_item.semi.is_some(),
        }
    }
}

pub struct SearchTree {
    pub docs: Docs,
    pub attrs: Vec<Attribute>,
    pub items: Vec<SearchItem>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Depth {
    Infinite,
    Remaining(NonZeroUsize),
    Done,
}

impl Depth {
    fn enter(self) -> Depth {
        if let Depth::Remaining(val) = self {
            match NonZeroUsize::new(usize::from(val) - 1) {
                None => Depth::Done,
                Some(val) => Depth::Remaining(val),
            }
        } else {
            self
        }
    }
}

impl SearchTree {
    pub fn new(file: &File) -> Self {
        SearchTree {
            docs: Docs::from(&file.attrs),
            attrs: file.attrs.clone(),
            items: file.items.iter().map(SearchItem::try_from).filter_map(Result::ok).collect(),
        }
    }

    pub fn search<Q: Fn(&SearchItem) -> bool>(&self, query: &Q, depth: Depth) -> Vec<&SearchItem> {
        let mut values = Vec::new();
        search(&self.items, query, depth, &mut values);
        values
    }
}

pub fn search<'a, Q: Fn(&SearchItem) -> bool>(items: &'a [SearchItem], query: &Q, depth: Depth, values: &mut Vec<&'a SearchItem>) {
    if depth == Depth::Done {
        return;
    }

    for item in items.iter() {
        match item {
            SearchItem::Impl(_, item) => {
                search(&item.items, query, depth.enter(), values);
            }
            SearchItem::Mod(_, item) => match item.content.as_deref() {
                None => {}
                Some(mod_items) => {
                    search(mod_items, query, depth.enter(), values);
                }
            }
            _ => {}
        }
        if query(item) {
            values.push(item);
        }
    }
}

// pub struct SearchAst {
//     doc_cache: RefCell<HashMap<*const u8, Rc<Docs>>>,
//     ast: File,
// }
//
// impl SearchAst {
//     pub fn new(file: &File) -> Self {
//         SearchAst {
//             doc_cache: RefCell::default(),
//             ast: file.clone(),
//         }
//     }
//
//     pub fn get_docs_item(&self, item: &Item) -> Rc<Docs> {
//         let ptr: *const u8 = unsafe { item as *const Item as *const u8 };
//         let attrs = match item {
//             Item::Const(item) => &item.attrs,
//             Item::Enum(item) => &item.attrs,
//             Item::ExternCrate(item) => &item.attrs,
//             Item::Fn(item) => &item.attrs,
//             Item::ForeignMod(item) => &item.attrs,
//             Item::Impl(item) => &item.attrs,
//             Item::Macro(item) => &item.attrs,
//             Item::Macro2(item) => &item.attrs,
//             Item::Mod(item) => &item.attrs,
//             Item::Static(item) => &item.attrs,
//             Item::Struct(item) => &item.attrs,
//             Item::Trait(item) => &item.attrs,
//             Item::TraitAlias(item) => &item.attrs,
//             Item::Type(item) => &item.attrs,
//             Item::Union(item) => &item.attrs,
//             Item::Use(item) => &item.attrs,
//             _ => unimplemented!(),
//         };
//
//         unsafe { self.doc_for_ptr(ptr, attrs) }
//     }
//
//     pub fn get_docs_impl_item(&self, item: &ImplItem) -> Rc<Docs> {
//         let ptr: *const u8 = unsafe { item as *const ImplItem as *const u8 };
//         let attrs = match item {
//             ImplItem::Const(item) => &item.attrs,
//             ImplItem::Method(item) => &item.attrs,
//             ImplItem::Type(item) => &item.attrs,
//             ImplItem::Macro(item) => &item.attrs,
//             _ => unimplemented!(),
//         };
//         unsafe { self.doc_for_ptr(ptr, attrs) }
//     }
//
//     pub fn get_docs_file(&self, item: &File) -> Rc<Docs> {
//         let ptr: *const u8 = unsafe { item as *const File as *const u8 };
//         unsafe { self.doc_for_ptr(ptr, &item.attrs) }
//     }
//
//     unsafe fn doc_for_ptr(&self, ptr: *const u8, attrs: &[Attribute]) -> Rc<Docs> {
//         if !self.doc_cache.borrow().contains_key(&ptr) {
//             self.doc_cache.borrow_mut().insert(ptr, Rc::new(Docs::from(attrs)));
//         };
//
//         self.doc_cache.borrow().get(&ptr).unwrap().clone()
//     }
// }
//
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
    pub(crate) fn item_matches(&self, item: &SearchItem) -> bool {
        match item {
            SearchItem::Fn(_, item) => match self.fn_arg_location {
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
                FnArgLocation::Output => {}
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
            SearchItem::Method(_, item) => match self.fn_arg_location {
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
                FnArgLocation::Output => {}
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

