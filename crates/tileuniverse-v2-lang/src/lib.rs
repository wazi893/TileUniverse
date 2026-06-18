//! V2 source-language AST — the C-like front-end types (`Expr`, `Stmt`, `Func`,
//! `Program`, `ArithOp`, `CmpOp`, `Cond`, constructors). Extracted from
//! `tile_cpu::v2_compiler` into a leaf crate so `synth` (HLS) can build/consume
//! the AST without a dependency edge into the tile-CPU codegen.

/// Binary arithmetic / bitwise operators (value-producing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
}

/// Comparison operators (used only in `if`/`while` conditions in D.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A value-producing expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// Integer literal.
    Int(i64),
    /// Local variable / parameter reference.
    Var(String),
    /// Binary arithmetic/bitwise op.
    Bin(ArithOp, Box<Expr>, Box<Expr>),
    /// Function call `name(args...)` — result in R0.
    Call(String, Vec<Expr>),
    /// Global-array element read `name[index]` (Sprint 382, Gate F). The array base is a
    /// fixed RAM address; the access lowers to `LD R0, [Rindex + base]`.
    Index(String, Box<Expr>),
}

/// A boolean condition (branch-only in D.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cond {
    Cmp(CmpOp, Expr, Expr),
}

/// A statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    /// Declare + initialize a local: `let name = expr`.
    Let(String, Expr),
    /// Assign to an existing local/param: `name = expr`.
    Assign(String, Expr),
    /// `if cond { then } else { els }`.
    If(Cond, Vec<Stmt>, Vec<Stmt>),
    /// `while cond { body }`.
    While(Cond, Vec<Stmt>),
    /// `return expr`.
    Return(Expr),
    /// Expression evaluated for effect (e.g. a call), result discarded.
    Expr(Expr),
    /// Global-array element store `name[index] = value` (Sprint 382, Gate F).
    StoreIndex(String, Expr, Expr),
}

/// A function definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Func {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
}

/// A global array (Sprint 382, Gate F): a fixed-size block of RAM cells with an optional
/// initializer. Bases are assigned at compile time from address 0 upward; the entry stub
/// zero/literal-initializes every cell so ISS and physical agree regardless of RAM reset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalArray {
    pub name: String,
    /// Number of cells (the declared length; `init` may be shorter — trailing cells 0).
    pub len: usize,
    /// Initial cell values (each 0..=65535); cells beyond `init.len()` start at 0.
    pub init: Vec<i64>,
}

/// A whole program: globals, a set of functions, an entry point, and the initial stack top.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub globals: Vec<GlobalArray>,
    pub funcs: Vec<Func>,
    /// Sprint 388 (HW/SW co-design, Phase 2): functions marked `accel` in the source.
    /// They are NOT compiled to instructions — they are synthesized by the HLS middle
    /// layer into a tile datapath behind the MMIO accelerator device, and call sites
    /// lower to the indexed MMIO protocol (ARG_SELECT / ARG_DATA / RESULT). At most
    /// one accel function per program this sprint (one device window).
    pub accel_funcs: Vec<Func>,
    pub entry: String,
    /// Initial SP (RAM address, empty-descending). Must stay within RAM (0..127).
    pub stack_top: u8,
}

impl Program {
    /// Build a program with the default stack top (120, leaving headroom in RAM).
    pub fn new(entry: &str, funcs: Vec<Func>) -> Self {
        Program {
            globals: Vec::new(),
            funcs,
            accel_funcs: Vec::new(),
            entry: entry.to_string(),
            stack_top: 120,
        }
    }

    /// Build a program with global arrays (Gate F).
    pub fn with_globals(entry: &str, globals: Vec<GlobalArray>, funcs: Vec<Func>) -> Self {
        Program {
            globals,
            funcs,
            accel_funcs: Vec::new(),
            entry: entry.to_string(),
            stack_top: 120,
        }
    }
}

/// A compilation failure with a human-readable message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError(pub String);

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "compile error: {}", self.0)
    }
}

// ===================================================================
// Ergonomic AST builders (keep test programs readable)
// ===================================================================

pub fn int(n: i64) -> Expr {
    Expr::Int(n)
}
pub fn var(name: &str) -> Expr {
    Expr::Var(name.to_string())
}
pub fn add(a: Expr, b: Expr) -> Expr {
    Expr::Bin(ArithOp::Add, Box::new(a), Box::new(b))
}
pub fn sub(a: Expr, b: Expr) -> Expr {
    Expr::Bin(ArithOp::Sub, Box::new(a), Box::new(b))
}
pub fn mul(a: Expr, b: Expr) -> Expr {
    Expr::Bin(ArithOp::Mul, Box::new(a), Box::new(b))
}
pub fn band(a: Expr, b: Expr) -> Expr {
    Expr::Bin(ArithOp::And, Box::new(a), Box::new(b))
}
pub fn bor(a: Expr, b: Expr) -> Expr {
    Expr::Bin(ArithOp::Or, Box::new(a), Box::new(b))
}
pub fn bxor(a: Expr, b: Expr) -> Expr {
    Expr::Bin(ArithOp::Xor, Box::new(a), Box::new(b))
}
pub fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Call(name.to_string(), args)
}
pub fn lt(a: Expr, b: Expr) -> Cond {
    Cond::Cmp(CmpOp::Lt, a, b)
}
pub fn le(a: Expr, b: Expr) -> Cond {
    Cond::Cmp(CmpOp::Le, a, b)
}
pub fn gt(a: Expr, b: Expr) -> Cond {
    Cond::Cmp(CmpOp::Gt, a, b)
}
pub fn ge(a: Expr, b: Expr) -> Cond {
    Cond::Cmp(CmpOp::Ge, a, b)
}
pub fn eq(a: Expr, b: Expr) -> Cond {
    Cond::Cmp(CmpOp::Eq, a, b)
}
pub fn ne(a: Expr, b: Expr) -> Cond {
    Cond::Cmp(CmpOp::Ne, a, b)
}
