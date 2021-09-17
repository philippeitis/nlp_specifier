use syn::{Abi, AngleBracketedGenericArguments, WherePredicate, WhereClause, Visibility, VisRestricted, VisPublic, VisCrate, Variant, Variadic, UseTree, UseRename, UsePath, UseName, UseGroup, UseGlob, UnOp, TypeTuple, TypeTraitObject, TypeSlice, TypeReference, TypePtr, TypePath, TypeParen, TypeParamBound, TypeParam, TypeNever, TypeMacro, TypeInfer, TypeImplTrait, TypeGroup, AttrStyle, Attribute, BareFnArg, BinOp, Binding, Block, BoundLifetimes, ConstParam, Constraint, DataEnum, DataStruct, DataUnion, DeriveInput, Arm, Expr, Data, Local, Macro, Member, Meta, MetaList, TypeArray, TypeBareFn, Type, TraitItemType, TraitItemMethod, TraitItemMacro, TraitItemConst, TraitItem, TraitBound, TraitBoundModifier, PatType, PatWild, Path, PathArguments, PathSegment, PredicateEq, PredicateLifetime, PredicateType, RangeLimits, Receiver, ReturnType, Signature, Stmt, QSelf, PatTupleStruct, PatStruct, PatTuple, PatSlice, PatRest, PatReference, PatRange, PatPath, PatOr, PatMacro, PatLit, PatIdent, PatBox, Pat, MacroDelimiter, MetaNameValue, MethodTurbofish, NestedMeta, ParenthesizedGenericArguments, LifetimeDef, Label, ItemUse, ItemUnion, ItemType, ItemTraitAlias, ItemTrait, ItemStruct, ItemStatic, ItemMod, ItemMacro2, ItemMacro, ItemImpl, ExprArray, ExprAssignOp, ExprAssign, ExprAsync, ExprAwait, ExprBinary, ExprBlock, ExprBox, ExprBreak, ExprCall, ExprCast, ExprClosure, ExprContinue, ExprField, ExprForLoop, ExprGroup, ExprIf, ExprIndex, ExprLet, ExprLit, ExprLoop, ExprMacro, ExprMatch, ExprMethodCall, ExprParen, ExprPath, ExprRange, ExprReference, ExprRepeat, ExprReturn, ExprStruct, ItemForeignMod, ItemExternCrate, ItemFn, ItemConst, ItemEnum, Item, Index, ImplItemType, ImplItemMethod, ImplItemMacro, ImplItemConst, ImplItem, Generics, GenericParam, GenericArgument, GenericMethodArgument, ForeignItemType, ForeignItemStatic, ForeignItemMacro, ForeignItemFn, ForeignItem, FnArg, File, ExprTry, ExprTryBlock, ExprTuple, ExprType, ExprUnary, ExprUnsafe, ExprWhile, ExprYield, Field, FieldPat, FieldValue, Fields, FieldsNamed, FieldsUnnamed};

enum VisitItem {
    Abi(Abi),
    AngleBracketedGenericArguments(AngleBracketedGenericArguments),
    Arm(Arm),
    AttrStyle(AttrStyle),
    Attribute(Attribute),
    BareFnArg(BareFnArg),
    BinOp(BinOp),
    Binding(Binding),
    Block(Block),
    BoundLifetimes(BoundLifetimes),
    ConstParam(ConstParam),
    Constraint(Constraint),
    Data(Data),
    DataEnum(DataEnum),
    DataStruct(DataStruct),
    DataUnion(DataUnion),
    DeriveInput(DeriveInput),
    Expr(Expr),
    ExprArray(ExprArray),
    ExprAssign(ExprAssign),
    ExprAssignOp(ExprAssignOp),
    ExprAsync(ExprAsync),
    ExprAwait(ExprAwait),
    ExprBinary(ExprBinary),
    ExprBlock(ExprBlock),
    ExprBox(ExprBox),
    ExprBreak(ExprBreak),
    ExprCall(ExprCall),
    ExprCast(ExprCast),
    ExprClosure(ExprClosure),
    ExprContinue(ExprContinue),
    ExprField(ExprField),
    ExprForLoop(ExprForLoop),
    ExprGroup(ExprGroup),
    ExprIf(ExprIf),
    ExprIndex(ExprIndex),
    ExprLet(ExprLet),
    ExprLit(ExprLit),
    ExprLoop(ExprLoop),
    ExprMacro(ExprMacro),
    ExprMatch(ExprMatch),
    ExprMethodCall(ExprMethodCall),
    ExprParen(ExprParen),
    ExprPath(ExprPath),
    ExprRange(ExprRange),
    ExprReference(ExprReference),
    ExprRepeat(ExprRepeat),
    ExprReturn(ExprReturn),
    ExprStruct(ExprStruct),
    ExprTry(ExprTry),
    ExprTryBlock(ExprTryBlock),
    ExprTuple(ExprTuple),
    ExprType(ExprType),
    ExprUnary(ExprUnary),
    ExprUnsafe(ExprUnsafe),
    ExprWhile(ExprWhile),
    ExprYield(ExprYield),
    Field(Field),
    FieldPat(FieldPat),
    FieldValue(FieldValue),
    Fields(Fields),
    FieldsNamed(FieldsNamed),
    FieldsUnnamed(FieldsUnnamed),
    File(File),
    FnArg(FnArg),
    ForeignItem(ForeignItem),
    ForeignItemFn(ForeignItemFn),
    ForeignItemMacro(ForeignItemMacro),
    ForeignItemStatic(ForeignItemStatic),
    ForeignItemType(ForeignItemType),
    GenericArgument(GenericArgument),
    GenericMethodArgument(GenericMethodArgument),
    GenericParam(GenericParam),
    Generics(Generics),
    ImplItem(ImplItem),
    ImplItemConst(ImplItemConst),
    ImplItemMacro(ImplItemMacro),
    ImplItemMethod(ImplItemMethod),
    ImplItemType(ImplItemType),
    Index(Index),
    Item(Item),
    ItemConst(ItemConst),
    ItemEnum(ItemEnum),
    ItemExternCrate(ItemExternCrate),
    ItemFn(ItemFn),
    ItemForeignMod(ItemForeignMod),
    ItemImpl(ItemImpl),
    ItemMacro(ItemMacro),
    ItemMacro2(ItemMacro2),
    ItemMod(ItemMod),
    ItemStatic(ItemStatic),
    ItemStruct(ItemStruct),
    ItemTrait(ItemTrait),
    ItemTraitAlias(ItemTraitAlias),
    ItemType(ItemType),
    ItemUnion(ItemUnion),
    ItemUse(ItemUse),
    Label(Label),
    Lifetime(Lifetime),
    LifetimeDef(LifetimeDef),
    Lit(Lit),
    LitBool(LitBool),
    LitByte(LitByte),
    LitByteStr(LitByteStr),
    LitChar(LitChar),
    LitFloat(LitFloat),
    LitInt(LitInt),
    LitStr(LitStr),
    Local(Local),
    Macro(Macro),
    MacroDelimiter(MacroDelimiter),
    Member(Member),
    Meta(Meta),
    MetaList(MetaList),
    MetaNameValue(MetaNameValue),
    MethodTurbofish(MethodTurbofish),
    NestedMeta(NestedMeta),
    ParenthesizedGenericArguments(ParenthesizedGenericArguments),
    Pat(Pat),
    PatBox(PatBox),
    PatIdent(PatIdent),
    PatLit(PatLit),
    PatMacro(PatMacro),
    PatOr(PatOr),
    PatPath(PatPath),
    PatRange(PatRange),
    PatReference(PatReference),
    PatRest(PatRest),
    PatSlice(PatSlice),
    PatStruct(PatStruct),
    PatTuple(PatTuple),
    PatTupleStruct(PatTupleStruct),
    PatType(PatType),
    PatWild(PatWild),
    Path(Path),
    PathArguments(PathArguments),
    PathSegment(PathSegment),
    PredicateEq(PredicateEq),
    PredicateLifetime(PredicateLifetime),
    PredicateType(PredicateType),
    QSelf(QSelf),
    RangeLimits(RangeLimits),
    Receiver(Receiver),
    ReturnType(ReturnType),
    Signature(Signature),
    Stmt(Stmt),
    TraitBound(TraitBound),
    TraitBoundModifier(TraitBoundModifier),
    TraitItem(TraitItem),
    TraitItemConst(TraitItemConst),
    TraitItemMacro(TraitItemMacro),
    TraitItemMethod(TraitItemMethod),
    TraitItemType(TraitItemType),
    Type(Type),
    TypeArray(TypeArray),
    TypeBareFn(TypeBareFn),
    TypeGroup(TypeGroup),
    TypeImplTrait(TypeImplTrait),
    TypeInfer(TypeInfer),
    TypeMacro(TypeMacro),
    TypeNever(TypeNever),
    TypeParam(TypeParam),
    TypeParamBound(TypeParamBound),
    TypeParen(TypeParen),
    TypePath(TypePath),
    TypePtr(TypePtr),
    TypeReference(TypeReference),
    TypeSlice(TypeSlice),
    TypeTraitObject(TypeTraitObject),
    TypeTuple(TypeTuple),
    UnOp(UnOp),
    UseGlob(UseGlob),
    UseGroup(UseGroup),
    UseName(UseName),
    UsePath(UsePath),
    UseRename(UseRename),
    UseTree(UseTree),
    Variadic(Variadic),
    Variant(Variant),
    VisCrate(VisCrate),
    VisPublic(VisPublic),
    VisRestricted(VisRestricted),
    Visibility(Visibility),
    WhereClause(WhereClause),
    WherePredicate(WherePredicate),
}

pub trait Visit<'ast> {
    fn visit_abi(&mut self, i: &'ast Abi) {
        visit_abi(self, i);
    }
    fn visit_angle_bracketed_generic_arguments(&mut self, i: &'ast AngleBracketedGenericArguments) {
        visit_angle_bracketed_generic_arguments(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_arm(&mut self, i: &'ast Arm) {
        visit_arm(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_attr_style(&mut self, i: &'ast AttrStyle) {
        visit_attr_style(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_attribute(&mut self, i: &'ast Attribute) {
        visit_attribute(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_bare_fn_arg(&mut self, i: &'ast BareFnArg) {
        visit_bare_fn_arg(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_bin_op(&mut self, i: &'ast BinOp) {
        visit_bin_op(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_binding(&mut self, i: &'ast Binding) {
        visit_binding(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_block(&mut self, i: &'ast Block) {
        visit_block(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_bound_lifetimes(&mut self, i: &'ast BoundLifetimes) {
        visit_bound_lifetimes(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_const_param(&mut self, i: &'ast ConstParam) {
        visit_const_param(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_constraint(&mut self, i: &'ast Constraint) {
        visit_constraint(self, i);
    }
    #[cfg(feature = "derive")]
    fn visit_data(&mut self, i: &'ast Data) {
        visit_data(self, i);
    }
    #[cfg(feature = "derive")]
    fn visit_data_enum(&mut self, i: &'ast DataEnum) {
        visit_data_enum(self, i);
    }
    #[cfg(feature = "derive")]
    fn visit_data_struct(&mut self, i: &'ast DataStruct) {
        visit_data_struct(self, i);
    }
    #[cfg(feature = "derive")]
    fn visit_data_union(&mut self, i: &'ast DataUnion) {
        visit_data_union(self, i);
    }
    #[cfg(feature = "derive")]
    fn visit_derive_input(&mut self, i: &'ast DeriveInput) {
        visit_derive_input(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_expr(&mut self, i: &'ast Expr) {
        visit_expr(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_array(&mut self, i: &'ast ExprArray) {
        visit_expr_array(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_assign(&mut self, i: &'ast ExprAssign) {
        visit_expr_assign(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_assign_op(&mut self, i: &'ast ExprAssignOp) {
        visit_expr_assign_op(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_async(&mut self, i: &'ast ExprAsync) {
        visit_expr_async(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_await(&mut self, i: &'ast ExprAwait) {
        visit_expr_await(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_expr_binary(&mut self, i: &'ast ExprBinary) {
        visit_expr_binary(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_block(&mut self, i: &'ast ExprBlock) {
        visit_expr_block(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_box(&mut self, i: &'ast ExprBox) {
        visit_expr_box(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_break(&mut self, i: &'ast ExprBreak) {
        visit_expr_break(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_expr_call(&mut self, i: &'ast ExprCall) {
        visit_expr_call(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_expr_cast(&mut self, i: &'ast ExprCast) {
        visit_expr_cast(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_closure(&mut self, i: &'ast ExprClosure) {
        visit_expr_closure(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_continue(&mut self, i: &'ast ExprContinue) {
        visit_expr_continue(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_expr_field(&mut self, i: &'ast ExprField) {
        visit_expr_field(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_for_loop(&mut self, i: &'ast ExprForLoop) {
        visit_expr_for_loop(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_group(&mut self, i: &'ast ExprGroup) {
        visit_expr_group(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_if(&mut self, i: &'ast ExprIf) {
        visit_expr_if(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_expr_index(&mut self, i: &'ast ExprIndex) {
        visit_expr_index(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_let(&mut self, i: &'ast ExprLet) {
        visit_expr_let(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_expr_lit(&mut self, i: &'ast ExprLit) {
        visit_expr_lit(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_loop(&mut self, i: &'ast ExprLoop) {
        visit_expr_loop(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_macro(&mut self, i: &'ast ExprMacro) {
        visit_expr_macro(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_match(&mut self, i: &'ast ExprMatch) {
        visit_expr_match(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_method_call(&mut self, i: &'ast ExprMethodCall) {
        visit_expr_method_call(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_expr_paren(&mut self, i: &'ast ExprParen) {
        visit_expr_paren(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_expr_path(&mut self, i: &'ast ExprPath) {
        visit_expr_path(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_range(&mut self, i: &'ast ExprRange) {
        visit_expr_range(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_reference(&mut self, i: &'ast ExprReference) {
        visit_expr_reference(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_repeat(&mut self, i: &'ast ExprRepeat) {
        visit_expr_repeat(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_return(&mut self, i: &'ast ExprReturn) {
        visit_expr_return(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_struct(&mut self, i: &'ast ExprStruct) {
        visit_expr_struct(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_try(&mut self, i: &'ast ExprTry) {
        visit_expr_try(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_try_block(&mut self, i: &'ast ExprTryBlock) {
        visit_expr_try_block(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_tuple(&mut self, i: &'ast ExprTuple) {
        visit_expr_tuple(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_type(&mut self, i: &'ast ExprType) {
        visit_expr_type(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_expr_unary(&mut self, i: &'ast ExprUnary) {
        visit_expr_unary(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_unsafe(&mut self, i: &'ast ExprUnsafe) {
        visit_expr_unsafe(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_while(&mut self, i: &'ast ExprWhile) {
        visit_expr_while(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_expr_yield(&mut self, i: &'ast ExprYield) {
        visit_expr_yield(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_field(&mut self, i: &'ast Field) {
        visit_field(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_field_pat(&mut self, i: &'ast FieldPat) {
        visit_field_pat(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_field_value(&mut self, i: &'ast FieldValue) {
        visit_field_value(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_fields(&mut self, i: &'ast Fields) {
        visit_fields(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_fields_named(&mut self, i: &'ast FieldsNamed) {
        visit_fields_named(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_fields_unnamed(&mut self, i: &'ast FieldsUnnamed) {
        visit_fields_unnamed(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_file(&mut self, i: &'ast File) {
        visit_file(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_fn_arg(&mut self, i: &'ast FnArg) {
        visit_fn_arg(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_foreign_item(&mut self, i: &'ast ForeignItem) {
        visit_foreign_item(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_foreign_item_fn(&mut self, i: &'ast ForeignItemFn) {
        visit_foreign_item_fn(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_foreign_item_macro(&mut self, i: &'ast ForeignItemMacro) {
        visit_foreign_item_macro(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_foreign_item_static(&mut self, i: &'ast ForeignItemStatic) {
        visit_foreign_item_static(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_foreign_item_type(&mut self, i: &'ast ForeignItemType) {
        visit_foreign_item_type(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_generic_argument(&mut self, i: &'ast GenericArgument) {
        visit_generic_argument(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_generic_method_argument(&mut self, i: &'ast GenericMethodArgument) {
        visit_generic_method_argument(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_generic_param(&mut self, i: &'ast GenericParam) {
        visit_generic_param(self, i);
    }
    fn visit_generics(&mut self, i: &'ast Generics) {
        visit_generics(self, i);
    }
    fn visit_ident(&mut self, i: &'ast Ident) {
        visit_ident(self, i);
    }
    fn visit_impl_item(&mut self, i: &'ast ImplItem) {
        visit_impl_item(self, i);
    }
    fn visit_impl_item_const(&mut self, i: &'ast ImplItemConst) {
        visit_impl_item_const(self, i);
    }
    fn visit_impl_item_macro(&mut self, i: &'ast ImplItemMacro) {
        visit_impl_item_macro(self, i);
    }
    fn visit_impl_item_method(&mut self, i: &'ast ImplItemMethod) {
        visit_impl_item_method(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_impl_item_type(&mut self, i: &'ast ImplItemType) {
        visit_impl_item_type(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_index(&mut self, i: &'ast Index) {
        visit_index(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item(&mut self, i: &'ast Item) {
        visit_item(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_const(&mut self, i: &'ast ItemConst) {
        visit_item_const(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_enum(&mut self, i: &'ast ItemEnum) {
        visit_item_enum(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_extern_crate(&mut self, i: &'ast ItemExternCrate) {
        visit_item_extern_crate(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_fn(&mut self, i: &'ast ItemFn) {
        visit_item_fn(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_foreign_mod(&mut self, i: &'ast ItemForeignMod) {
        visit_item_foreign_mod(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_impl(&mut self, i: &'ast ItemImpl) {
        visit_item_impl(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_macro(&mut self, i: &'ast ItemMacro) {
        visit_item_macro(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_macro2(&mut self, i: &'ast ItemMacro2) {
        visit_item_macro2(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_mod(&mut self, i: &'ast ItemMod) {
        visit_item_mod(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_static(&mut self, i: &'ast ItemStatic) {
        visit_item_static(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_struct(&mut self, i: &'ast ItemStruct) {
        visit_item_struct(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_trait(&mut self, i: &'ast ItemTrait) {
        visit_item_trait(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_trait_alias(&mut self, i: &'ast ItemTraitAlias) {
        visit_item_trait_alias(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_type(&mut self, i: &'ast ItemType) {
        visit_item_type(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_union(&mut self, i: &'ast ItemUnion) {
        visit_item_union(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_item_use(&mut self, i: &'ast ItemUse) {
        visit_item_use(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_label(&mut self, i: &'ast Label) {
        visit_label(self, i);
    }
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        visit_lifetime(self, i);
    }
    fn visit_lifetime_def(&mut self, i: &'ast LifetimeDef) {
        visit_lifetime_def(self, i);
    }
    fn visit_lit(&mut self, i: &'ast Lit) {
        visit_lit(self, i);
    }
    fn visit_lit_bool(&mut self, i: &'ast LitBool) {
        visit_lit_bool(self, i);
    }
    fn visit_lit_byte(&mut self, i: &'ast LitByte) {
        visit_lit_byte(self, i);
    }
    fn visit_lit_byte_str(&mut self, i: &'ast LitByteStr) {
        visit_lit_byte_str(self, i);
    }
    fn visit_lit_char(&mut self, i: &'ast LitChar) {
        visit_lit_char(self, i);
    }
    fn visit_lit_float(&mut self, i: &'ast LitFloat) {
        visit_lit_float(self, i);
    }
    fn visit_lit_int(&mut self, i: &'ast LitInt) {
        visit_lit_int(self, i);
    }
    fn visit_lit_str(&mut self, i: &'ast LitStr) {
        visit_lit_str(self, i);
    }
    fn visit_local(&mut self, i: &'ast Local) {
        visit_local(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_macro(&mut self, i: &'ast Macro) {
        visit_macro(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_macro_delimiter(&mut self, i: &'ast MacroDelimiter) {
        visit_macro_delimiter(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_member(&mut self, i: &'ast Member) {
        visit_member(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_meta(&mut self, i: &'ast Meta) {
        visit_meta(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_meta_list(&mut self, i: &'ast MetaList) {
        visit_meta_list(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_meta_name_value(&mut self, i: &'ast MetaNameValue) {
        visit_meta_name_value(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_method_turbofish(&mut self, i: &'ast MethodTurbofish) {
        visit_method_turbofish(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_nested_meta(&mut self, i: &'ast NestedMeta) {
        visit_nested_meta(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_parenthesized_generic_arguments(&mut self, i: &'ast ParenthesizedGenericArguments) {
        visit_parenthesized_generic_arguments(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat(&mut self, i: &'ast Pat) {
        visit_pat(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_box(&mut self, i: &'ast PatBox) {
        visit_pat_box(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_ident(&mut self, i: &'ast PatIdent) {
        visit_pat_ident(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_lit(&mut self, i: &'ast PatLit) {
        visit_pat_lit(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_macro(&mut self, i: &'ast PatMacro) {
        visit_pat_macro(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_or(&mut self, i: &'ast PatOr) {
        visit_pat_or(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_path(&mut self, i: &'ast PatPath) {
        visit_pat_path(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_range(&mut self, i: &'ast PatRange) {
        visit_pat_range(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_reference(&mut self, i: &'ast PatReference) {
        visit_pat_reference(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_rest(&mut self, i: &'ast PatRest) {
        visit_pat_rest(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_slice(&mut self, i: &'ast PatSlice) {
        visit_pat_slice(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_struct(&mut self, i: &'ast PatStruct) {
        visit_pat_struct(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_tuple(&mut self, i: &'ast PatTuple) {
        visit_pat_tuple(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_tuple_struct(&mut self, i: &'ast PatTupleStruct) {
        visit_pat_tuple_struct(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_type(&mut self, i: &'ast PatType) {
        visit_pat_type(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_pat_wild(&mut self, i: &'ast PatWild) {
        visit_pat_wild(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_path(&mut self, i: &'ast Path) {
        visit_path(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_path_arguments(&mut self, i: &'ast PathArguments) {
        visit_path_arguments(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_path_segment(&mut self, i: &'ast PathSegment) {
        visit_path_segment(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_predicate_eq(&mut self, i: &'ast PredicateEq) {
        visit_predicate_eq(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_predicate_lifetime(&mut self, i: &'ast PredicateLifetime) {
        visit_predicate_lifetime(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_predicate_type(&mut self, i: &'ast PredicateType) {
        visit_predicate_type(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_qself(&mut self, i: &'ast QSelf) {
        visit_qself(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_range_limits(&mut self, i: &'ast RangeLimits) {
        visit_range_limits(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_receiver(&mut self, i: &'ast Receiver) {
        visit_receiver(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_return_type(&mut self, i: &'ast ReturnType) {
        visit_return_type(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_signature(&mut self, i: &'ast Signature) {
        visit_signature(self, i);
    }
    fn visit_stmt(&mut self, i: &'ast Stmt) {
        visit_stmt(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
        visit_trait_bound(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_trait_bound_modifier(&mut self, i: &'ast TraitBoundModifier) {
        visit_trait_bound_modifier(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_trait_item(&mut self, i: &'ast TraitItem) {
        visit_trait_item(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_trait_item_const(&mut self, i: &'ast TraitItemConst) {
        visit_trait_item_const(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_trait_item_macro(&mut self, i: &'ast TraitItemMacro) {
        visit_trait_item_macro(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_trait_item_method(&mut self, i: &'ast TraitItemMethod) {
        visit_trait_item_method(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_trait_item_type(&mut self, i: &'ast TraitItemType) {
        visit_trait_item_type(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type(&mut self, i: &'ast Type) {
        visit_type(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_array(&mut self, i: &'ast TypeArray) {
        visit_type_array(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_bare_fn(&mut self, i: &'ast TypeBareFn) {
        visit_type_bare_fn(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_group(&mut self, i: &'ast TypeGroup) {
        visit_type_group(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_impl_trait(&mut self, i: &'ast TypeImplTrait) {
        visit_type_impl_trait(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_infer(&mut self, i: &'ast TypeInfer) {
        visit_type_infer(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_macro(&mut self, i: &'ast TypeMacro) {
        visit_type_macro(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_never(&mut self, i: &'ast TypeNever) {
        visit_type_never(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_param(&mut self, i: &'ast TypeParam) {
        visit_type_param(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_param_bound(&mut self, i: &'ast TypeParamBound) {
        visit_type_param_bound(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_paren(&mut self, i: &'ast TypeParen) {
        visit_type_paren(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_path(&mut self, i: &'ast TypePath) {
        visit_type_path(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_ptr(&mut self, i: &'ast TypePtr) {
        visit_type_ptr(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        visit_type_reference(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_slice(&mut self, i: &'ast TypeSlice) {
        visit_type_slice(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_trait_object(&mut self, i: &'ast TypeTraitObject) {
        visit_type_trait_object(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_type_tuple(&mut self, i: &'ast TypeTuple) {
        visit_type_tuple(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_un_op(&mut self, i: &'ast UnOp) {
        visit_un_op(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_use_glob(&mut self, i: &'ast UseGlob) {
        visit_use_glob(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_use_group(&mut self, i: &'ast UseGroup) {
        visit_use_group(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_use_name(&mut self, i: &'ast UseName) {
        visit_use_name(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_use_path(&mut self, i: &'ast UsePath) {
        visit_use_path(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_use_rename(&mut self, i: &'ast UseRename) {
        visit_use_rename(self, i);
    }
    #[cfg(feature = "full")]
    fn visit_use_tree(&mut self, i: &'ast UseTree) {
        visit_use_tree(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_variadic(&mut self, i: &'ast Variadic) {
        visit_variadic(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_variant(&mut self, i: &'ast Variant) {
        visit_variant(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_vis_crate(&mut self, i: &'ast VisCrate) {
        visit_vis_crate(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_vis_public(&mut self, i: &'ast VisPublic) {
        visit_vis_public(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_vis_restricted(&mut self, i: &'ast VisRestricted) {
        visit_vis_restricted(self, i);
    }
    #[cfg(any(feature = "derive", feature = "full"))]
    fn visit_visibility(&mut self, i: &'ast Visibility) {
        visit_visibility(self, i);
    }
    fn visit_where_clause(&mut self, i: &'ast WhereClause) {
        visit_where_clause(self, i);
    }
    fn visit_where_predicate(&mut self, i: &'ast WherePredicate) {
        visit_where_predicate(self, i);
    }
}