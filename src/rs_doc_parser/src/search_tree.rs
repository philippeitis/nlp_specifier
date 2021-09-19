use crate::docs::Docs;
use std::convert::TryFrom;

use std::num::NonZeroUsize;
use syn::{ItemConst, ItemEnum, ItemFn, ItemStruct, ImplItemConst, ImplItemMethod, Item, Attribute, Generics, Path, Type, ImplItem, ItemImpl, Visibility, ItemMod, File};
use std::fmt::{Display, Formatter, Debug};
use quote::ToTokens;

pub enum SearchItem {
    Const(ItemConst),
    Enum(ItemEnum),
    // ExternCrate(ItemExternCrate),
    Fn(ItemFn),
    // ForeignMod(Doc, ItemForeignMod),
    Impl(SearchItemImpl),
    // Macro(ItemMacro),
    // Macro2(ItemMacro2),
    Mod(SearchItemMod),
    // Static(ItemStatic),
    Struct(ItemStruct),
    // Trait(ItemTrait),
    // TraitAlias(ItemTraitAlias),
    // Type(ItemType),
    // Union(ItemUnion),
    // Use(ItemUse),
    // Verbatim(TokenStream),
    ImplConst(ImplItemConst),
    Method(ImplItemMethod),
    // ImplType(Docs, ImplItemType),
    // ImplMacro(Docs, ImplItemMacro),
    // ImplVerbatim(Docs, TokenStream),
    // some variants omitted
}

impl SearchItem {
    fn attrs(&self) -> &[Attribute] {
        match self {
            SearchItem::Const(i) => &i.attrs,
            SearchItem::Enum(i) => &i.attrs,
            SearchItem::Fn(i) => &i.attrs,
            SearchItem::Impl(i) => &i.attrs,
            SearchItem::Mod(i) => &i.attrs,
            SearchItem::Struct(i) => &i.attrs,
            SearchItem::ImplConst(i) => &i.attrs,
            SearchItem::Method(i) => &i.attrs,
        }
    }
}

pub struct SearchValue {
    pub(crate) docs: Docs,
    pub(crate) item: SearchItem,
}

impl From<SearchItem> for SearchValue {
    fn from(item: SearchItem) -> Self {
        Self {
            docs: Docs::from(item.attrs()),
            item,
        }
    }
}

impl Display for SearchItem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            SearchItem::Const(item) => {
                write!(f, "{}", item.ident)
            }
            SearchItem::Enum(item) => {
                write!(f, "{}", item.ident)
            }
            SearchItem::Fn(item) => {
                write!(f, "{}", item.sig.to_token_stream().to_string())
            }
            SearchItem::Impl(item) => {
                write!(f, "{}", item.self_ty.to_token_stream().to_string())
            }
            SearchItem::Mod(item) => {
                write!(f, "{}", item.ident.to_string())
            }
            SearchItem::Struct(item) => {
                write!(f, "{}", item.ident.to_string())
            }
            SearchItem::ImplConst(item) => {
                write!(f, "{}", item.ident.to_string())
            }
            SearchItem::Method(item) => {
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

impl Debug for SearchValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.item, f)
    }
}

impl Display for SearchValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.item, f)
    }
}

impl TryFrom<&Item> for SearchValue {
    type Error = ();

    fn try_from(item: &Item) -> Result<Self, Self::Error> {
        Ok(match item {
            Item::Const(item) => SearchItem::Const(item.clone()),
            Item::Enum(item) => SearchItem::Enum(item.clone()),
            // Item::ExternCrate(_) => {}
            Item::Fn(item) => SearchItem::Fn(item.clone()),
            // Item::ForeignMod(_) => {}
            Item::Impl(item) => SearchItem::Impl(SearchItemImpl::from(item)),
            // Item::Macro(_) => {}
            // Item::Macro2(_) => {}
            Item::Mod(item) => SearchItem::Mod(SearchItemMod::from(item)),
            // Item::Static(_) => {}
            Item::Struct(item) => SearchItem::Struct(item.clone()),
            // Item::Trait(_) => {}
            // Item::TraitAlias(_) => {}
            // Item::Type(_) => {}
            // Item::Union(_) => {}
            // Item::Use(_) => {}
            // Item::Verbatim(_) => {}
            // Item::__TestExhaustive(_) => {}
            _ => return Err(())
        }.into())
    }
}

impl TryFrom<Item> for SearchValue {
    type Error = ();

    fn try_from(item: Item) -> Result<Self, Self::Error> {
        Ok(match item {
            Item::Const(item) => SearchItem::Const(item),
            Item::Enum(item) => SearchItem::Enum(item),

            // Item::ExternCrate(_) => {}
            Item::Fn(item) => SearchItem::Fn(item),
            // Item::ForeignMod(_) => {}
            Item::Impl(item) => SearchItem::Impl(SearchItemImpl::from(&item)),
            // Item::Macro(_) => {}
            // Item::Macro2(_) => {}
            Item::Mod(item) => SearchItem::Mod(SearchItemMod::from(&item)),
            // Item::Static(_) => {}
            Item::Struct(item) => SearchItem::Struct(item),
            // Item::Trait(_) => {}
            // Item::TraitAlias(_) => {}
            // Item::Type(_) => {}
            // Item::Union(_) => {}
            // Item::Use(_) => {}
            // Item::Verbatim(_) => {}
            // Item::__TestExhaustive(_) => {}
            _ => return Err(())
        }.into())
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
    pub items: Vec<SearchValue>,
}

impl TryFrom<&ImplItem> for SearchValue {
    type Error = ();

    fn try_from(item: &ImplItem) -> Result<Self, Self::Error> {
        Ok(match item {
            ImplItem::Const(item) => SearchItem::ImplConst(item.clone()),
            ImplItem::Method(item) => SearchItem::Method(item.clone()),
            _ => return Err(()),
        }.into())
    }
}

impl TryFrom<ImplItem> for SearchValue {
    type Error = ();

    fn try_from(item: ImplItem) -> Result<Self, Self::Error> {
        Ok(match item {
            ImplItem::Const(item) => SearchItem::ImplConst(item),
            ImplItem::Method(item) => SearchItem::Method(item),
            _ => return Err(()),
        }.into())
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
            items: impl_item.items.iter().map(SearchValue::try_from).filter_map(Result::ok).collect(),
        }
    }
}

#[derive(Debug)]
pub struct SearchItemMod {
    pub docs: Docs,
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub ident: String,
    pub content: Option<Vec<SearchValue>>,
    pub semi: bool,
}

impl From<&ItemMod> for SearchItemMod {
    fn from(mod_item: &ItemMod) -> Self {
        let docs = Docs::from(&mod_item.attrs);
        let content = mod_item.content.as_ref().map(
            |(_, content)| content.iter().map(SearchValue::try_from).filter_map(Result::ok).collect()
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
    pub items: Vec<SearchValue>,
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
            items: file.items.iter().map(SearchValue::try_from).filter_map(Result::ok).collect(),
        }
    }

    pub fn search<Q: Fn(&SearchValue) -> bool>(&self, query: &Q, depth: Depth) -> Vec<&SearchValue> {
        let mut values = Vec::new();
        search(&self.items, query, depth, &mut values);
        values
    }
}

pub fn search<'a, Q: Fn(&SearchValue) -> bool>(items: &'a [SearchValue], query: &Q, depth: Depth, values: &mut Vec<&'a SearchValue>) {
    if depth == Depth::Done {
        return;
    }

    for item in items.iter() {
        match &item.item {
            SearchItem::Impl(item) => {
                search(&item.items, query, depth.enter(), values);
            }
            SearchItem::Mod(item) => match item.content.as_deref() {
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
