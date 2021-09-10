use syn::{File, ItemConst, ItemEnum, ItemImpl, ItemMod, ItemStruct, ImplItemConst, ImplItemMethod, Visibility, Item, ItemFn, Attribute, Generics, Type, Path, ImplItem};
use crate::docs::Docs;
use std::convert::TryFrom;

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

pub struct SearchItemImpl {
    pub docs: Docs,
    pub attrs: Vec<Attribute>,
    pub defaultness: bool,
    pub unsafety: bool,
    pub generics: Generics,
    pub trait_: Option<Path>,
    pub self_ty: Box<Type>,
    pub items: Vec<SearchImplItem>,
}

pub enum SearchImplItem {
    Const(Docs, ImplItemConst),
    Method(Docs, ImplItemMethod),
    // Type(Docs, ImplItemType),
    // Macro(Docs, ImplItemMacro),
    // Verbatim(Docs, TokenStream),
    // some variants omitted
}

impl TryFrom<&ImplItem> for SearchImplItem {
    type Error = ();

    fn try_from(item: &ImplItem) -> Result<Self, Self::Error> {
        match item {
            ImplItem::Const(item) => {
                Ok(SearchImplItem::Const(Docs::from(&item.attrs), item.clone()))
            }
            ImplItem::Method(item) => {
                Ok(SearchImplItem::Method(Docs::from(&item.attrs), item.clone()))
            }
            _ => Err(())
        }
    }
}


impl From<&ItemImpl> for SearchItemImpl {
    fn from(impl_item: &ItemImpl) -> Self {
        let docs = Docs::from(&impl_item.attrs);
        SearchItemImpl {
            docs,
            attrs: impl_item.attrs.clone(),
            defaultness: impl_item.defaultness.is_some(),
            unsafety: impl_item.unsafety.is_some(),
            generics: impl_item.generics.clone(),
            trait_: impl_item.trait_.as_ref().map(|(_, path, _)| path.clone()),
            self_ty: impl_item.self_ty.clone(),
            items: impl_item.items.iter().map(SearchImplItem::try_from).filter_map(Result::ok).collect()
        }
    }
}

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
            semi: mod_item.semi.is_some()
        }
    }
}

pub struct SearchTree {
    pub docs: Docs,
    pub attrs: Vec<Attribute>,
    pub items: Vec<SearchItem>
}

impl SearchTree {
    pub fn new(file: &File) -> Self {
        SearchTree {
            docs: Docs::from(&file.attrs),
            attrs: file.attrs.clone(),
            items: file.items.iter().map(SearchItem::try_from).filter_map(Result::ok).collect()
        }
    }

    pub fn search<Q: Fn(&SearchItem) -> bool>(&self, query: Q) -> Vec<&SearchItem> {
        self.items.iter().filter(|x| query(*x)).collect()
    }
}
