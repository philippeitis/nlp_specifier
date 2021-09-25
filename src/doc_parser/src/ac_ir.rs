/// Abstract Code Intermediate Representation
/// Represents code where we do not have concrete types or functions - instead, we have
/// some textual description which must be resolved to a concrete function or type
use crate::nlp_ir::Specification;

/// Unresolved expressions
enum UExpr {
}

/// Resolved Expressions - all type information is available
enum Expr {
    //
}

/// Candidate type:
///     - `1` can be an integer, usize, u32
enum TypeCandidate {
    OneOf(Vec<Type>),
    Final(Type),
}

/// A quantifier, which can be universal or not.
struct ExprQuantifierU {
    is_universal: bool,
    idents: Vec<String>,
    body: Expr,
}

/// An IFF statement
struct ExprIff {
    rhs: Expr,
    lhs: Expr
}

trait TypeHint {
    /// If the expression type can be discovered without any hints, it is returned here:
    ///     - `a == b`
    ///     - `a && b`
    ///     - "hello world"
    ///     - 32_usize
    fn naive_type(&self) -> Option<Type> {

    }
}

/// Represents previously seen items and their types. Contains a set of specifications
/// which provide hints.
struct Context {
    ident_type: Vec<(String, String)>,
    specifications: Vec<Specification>
}

impl Context {
    /// Trees can include:
    ///     - statements (eg. Let `i` be equal to `1`)
    ///     - isolated specifications (Returns `true` if some condition is met)
    ///     - connected specifications (Returns `true` if some condition. Otherwise, returns `False`
    fn attach_tree(&mut self, spec: Tree) {

    }
}

/// Can resolve types for:
/// Literal: Idents in function signature (very easy)
/// Literal: Literals (eg. "hello world", 3.0f32, ambiguous such as 1: int -> more specific)
/// Code: Method calls on resolved items or idents (eg. hello is int, hello.add(1), 32.0_f32.div(1))