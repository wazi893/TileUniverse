//! Sprint 373 (Gate D.2): text front end — a C-like source language parsed into the
//! Gate D.1 [`Program`] AST.
//!
//! This puts a real lexer + recursive-descent parser on top of the D.1 codegen, so a
//! `.c`-style source file compiles end-to-end:
//! `source text → parse → AST → compile_program → asm → assemble_v2 → Vec<u32>`.
//! The backend (compiler, assembler, ISS, physical executor) is untouched.
//!
//! # Surface language
//! ```c
//! int fact(int n) {
//!     if (n < 2) { return 1; }
//!     return n * fact(n - 1);
//! }
//! int main() { return fact(5); }
//! ```
//! - Only type `int` (the u64 register width); functions with ≤4 params; recursion.
//! - Value expressions with C precedence: `|` < `^` < `&` < `+ -` < `*`; primaries
//!   are integer literals (decimal / `0x` hex), identifiers, calls `f(a,b)`, and
//!   `( expr )`.
//! - Conditions (in `if`/`while` only): `expr CMP expr` for `== != < <= > >=`; a bare
//!   `if (e)` / `while (e)` desugars to `e != 0` (C-like truthiness).
//! - Statements: `return e;`, `int x = e;`, `x = e;`, `f(args);`, `if/else`, `while`.
//! - `//` line comments; whitespace insignificant.

use crate::tile_cpu::v2_compiler::{
    ArithOp, CmpOp, CompileError, Cond, Expr, Func, GlobalArray, Program, Stmt, compile_to_words,
};

/// A lex/parse failure with a 1-based source line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at line {}: {}", self.line, self.message)
    }
}

fn err<T>(line: usize, message: impl Into<String>) -> Result<T, ParseError> {
    Err(ParseError {
        line,
        message: message.into(),
    })
}

// ===================================================================
// Lexer
// ===================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Int(i64),
    Ident(String),
    KwInt,
    KwIf,
    KwElse,
    KwWhile,
    KwReturn,
    Plus,
    Minus,
    Star,
    Amp,
    Pipe,
    Caret,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Assign,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Eof,
}

#[derive(Debug, Clone)]
struct Lexed {
    tok: Tok,
    line: usize,
}

fn tokenize(src: &str) -> Result<Vec<Lexed>, ParseError> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut line = 1usize;
    let mut out = Vec::new();
    let n = chars.len();

    let push = |out: &mut Vec<Lexed>, tok: Tok, line: usize| out.push(Lexed { tok, line });

    while i < n {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\r' => i += 1,
            '\n' => {
                line += 1;
                i += 1;
            }
            // `//` line comment.
            '/' if i + 1 < n && chars[i + 1] == '/' => {
                while i < n && chars[i] != '\n' {
                    i += 1;
                }
            }
            '(' => {
                push(&mut out, Tok::LParen, line);
                i += 1;
            }
            ')' => {
                push(&mut out, Tok::RParen, line);
                i += 1;
            }
            '{' => {
                push(&mut out, Tok::LBrace, line);
                i += 1;
            }
            '}' => {
                push(&mut out, Tok::RBrace, line);
                i += 1;
            }
            '[' => {
                push(&mut out, Tok::LBracket, line);
                i += 1;
            }
            ']' => {
                push(&mut out, Tok::RBracket, line);
                i += 1;
            }
            ',' => {
                push(&mut out, Tok::Comma, line);
                i += 1;
            }
            ';' => {
                push(&mut out, Tok::Semi, line);
                i += 1;
            }
            '+' => {
                push(&mut out, Tok::Plus, line);
                i += 1;
            }
            '-' => {
                push(&mut out, Tok::Minus, line);
                i += 1;
            }
            '*' => {
                push(&mut out, Tok::Star, line);
                i += 1;
            }
            '&' => {
                push(&mut out, Tok::Amp, line);
                i += 1;
            }
            '|' => {
                push(&mut out, Tok::Pipe, line);
                i += 1;
            }
            '^' => {
                push(&mut out, Tok::Caret, line);
                i += 1;
            }
            '=' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    push(&mut out, Tok::EqEq, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Assign, line);
                    i += 1;
                }
            }
            '!' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    push(&mut out, Tok::Ne, line);
                    i += 2;
                } else {
                    return err(line, "unexpected '!' (did you mean '!='?)");
                }
            }
            '<' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    push(&mut out, Tok::Le, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Lt, line);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < n && chars[i + 1] == '=' {
                    push(&mut out, Tok::Ge, line);
                    i += 2;
                } else {
                    push(&mut out, Tok::Gt, line);
                    i += 1;
                }
            }
            c if c.is_ascii_digit() => {
                let start = i;
                let (radix, skip) =
                    if c == '0' && i + 1 < n && (chars[i + 1] == 'x' || chars[i + 1] == 'X') {
                        (16u32, 2)
                    } else {
                        (10u32, 0)
                    };
                i += skip;
                let dig_start = i;
                while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let text: String = chars[dig_start..i].iter().collect();
                let value = i64::from_str_radix(&text, radix).map_err(|_| ParseError {
                    line,
                    message: format!(
                        "invalid integer literal '{}'",
                        chars[start..i].iter().collect::<String>()
                    ),
                })?;
                push(&mut out, Tok::Int(value), line);
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                let tok = match word.as_str() {
                    "int" => Tok::KwInt,
                    "if" => Tok::KwIf,
                    "else" => Tok::KwElse,
                    "while" => Tok::KwWhile,
                    "return" => Tok::KwReturn,
                    _ => Tok::Ident(word),
                };
                push(&mut out, tok, line);
            }
            other => return err(line, format!("unexpected character '{other}'")),
        }
    }
    out.push(Lexed {
        tok: Tok::Eof,
        line,
    });
    Ok(out)
}

// ===================================================================
// Parser (recursive descent)
// ===================================================================

struct Parser {
    toks: Vec<Lexed>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn line(&self) -> usize {
        self.toks[self.pos].line
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].tok.clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    /// Consume a token that must equal `want`, else a parse error naming `what`.
    fn expect(&mut self, want: &Tok, what: &str) -> Result<(), ParseError> {
        if self.peek() == want {
            self.bump();
            Ok(())
        } else {
            err(
                self.line(),
                format!("expected {what}, found {:?}", self.peek()),
            )
        }
    }

    fn expect_ident(&mut self, what: &str) -> Result<String, ParseError> {
        match self.peek().clone() {
            Tok::Ident(name) => {
                self.bump();
                Ok(name)
            }
            other => err(self.line(), format!("expected {what}, found {other:?}")),
        }
    }

    // ---- expressions (C precedence; no comparisons as values) ----

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_bin(0)
    }

    /// Precedence-climbing over the binary arithmetic/bitwise levels.
    /// Level 0 = `|`, 1 = `^`, 2 = `&`, 3 = `+ -`, 4 = `*`.
    fn parse_bin(&mut self, level: usize) -> Result<Expr, ParseError> {
        if level == 5 {
            return self.parse_primary();
        }
        let mut left = self.parse_bin(level + 1)?;
        loop {
            let op = match (level, self.peek()) {
                (0, Tok::Pipe) => ArithOp::Or,
                (1, Tok::Caret) => ArithOp::Xor,
                (2, Tok::Amp) => ArithOp::And,
                (3, Tok::Plus) => ArithOp::Add,
                (3, Tok::Minus) => ArithOp::Sub,
                (4, Tok::Star) => ArithOp::Mul,
                _ => break,
            };
            self.bump();
            let right = self.parse_bin(level + 1)?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            Tok::Int(n) => {
                self.bump();
                Ok(Expr::Int(n))
            }
            Tok::Ident(name) => {
                self.bump();
                if self.peek() == &Tok::LParen {
                    let args = self.parse_call_args()?;
                    Ok(Expr::Call(name, args))
                } else if self.peek() == &Tok::LBracket {
                    // Array read `name[index]` (Sprint 382, Gate F).
                    self.bump();
                    let idx = self.parse_expr()?;
                    self.expect(&Tok::RBracket, "']'")?;
                    Ok(Expr::Index(name, Box::new(idx)))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Tok::LParen => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect(&Tok::RParen, "')'")?;
                Ok(e)
            }
            // A comparison operator here means someone used a comparison as a value.
            Tok::EqEq | Tok::Ne | Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge => err(
                self.line(),
                "comparisons are only allowed in if/while conditions (no boolean values yet)",
            ),
            other => err(
                self.line(),
                format!("expected an expression, found {other:?}"),
            ),
        }
    }

    /// Parse `( a, b, ... )` argument list (the `(` is the current token).
    fn parse_call_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        self.expect(&Tok::LParen, "'('")?;
        let mut args = Vec::new();
        if self.peek() != &Tok::RParen {
            loop {
                args.push(self.parse_expr()?);
                if self.peek() == &Tok::Comma {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen, "')'")?;
        Ok(args)
    }

    // ---- conditions (if/while only) ----

    fn parse_cond(&mut self) -> Result<Cond, ParseError> {
        let a = self.parse_expr()?;
        let op = match self.peek() {
            Tok::EqEq => Some(CmpOp::Eq),
            Tok::Ne => Some(CmpOp::Ne),
            Tok::Lt => Some(CmpOp::Lt),
            Tok::Le => Some(CmpOp::Le),
            Tok::Gt => Some(CmpOp::Gt),
            Tok::Ge => Some(CmpOp::Ge),
            _ => None,
        };
        match op {
            Some(cmp) => {
                self.bump();
                let b = self.parse_expr()?;
                Ok(Cond::Cmp(cmp, a, b))
            }
            // Bare condition: truthiness `a != 0`.
            None => Ok(Cond::Cmp(CmpOp::Ne, a, Expr::Int(0))),
        }
    }

    // ---- statements ----

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.expect(&Tok::LBrace, "'{'")?;
        let mut stmts = Vec::new();
        while self.peek() != &Tok::RBrace {
            if self.peek() == &Tok::Eof {
                return err(self.line(), "unexpected end of input (missing '}')");
            }
            stmts.push(self.parse_stmt()?);
        }
        self.expect(&Tok::RBrace, "'}'")?;
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek().clone() {
            Tok::KwReturn => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect(&Tok::Semi, "';'")?;
                Ok(Stmt::Return(e))
            }
            Tok::KwInt => {
                self.bump();
                let name = self.expect_ident("variable name")?;
                self.expect(&Tok::Assign, "'='")?;
                let e = self.parse_expr()?;
                self.expect(&Tok::Semi, "';'")?;
                Ok(Stmt::Let(name, e))
            }
            Tok::KwIf => {
                self.bump();
                self.expect(&Tok::LParen, "'('")?;
                let cond = self.parse_cond()?;
                self.expect(&Tok::RParen, "')'")?;
                let then_body = self.parse_block()?;
                let else_body = if self.peek() == &Tok::KwElse {
                    self.bump();
                    self.parse_block()?
                } else {
                    Vec::new()
                };
                Ok(Stmt::If(cond, then_body, else_body))
            }
            Tok::KwWhile => {
                self.bump();
                self.expect(&Tok::LParen, "'('")?;
                let cond = self.parse_cond()?;
                self.expect(&Tok::RParen, "')'")?;
                let body = self.parse_block()?;
                Ok(Stmt::While(cond, body))
            }
            Tok::Ident(name) => {
                self.bump();
                match self.peek() {
                    Tok::Assign => {
                        self.bump();
                        let e = self.parse_expr()?;
                        self.expect(&Tok::Semi, "';'")?;
                        Ok(Stmt::Assign(name, e))
                    }
                    Tok::LParen => {
                        let args = self.parse_call_args()?;
                        self.expect(&Tok::Semi, "';'")?;
                        Ok(Stmt::Expr(Expr::Call(name, args)))
                    }
                    // Array store `name[index] = value;` (Sprint 382, Gate F).
                    Tok::LBracket => {
                        self.bump();
                        let idx = self.parse_expr()?;
                        self.expect(&Tok::RBracket, "']'")?;
                        self.expect(&Tok::Assign, "'=' (array element store)")?;
                        let val = self.parse_expr()?;
                        self.expect(&Tok::Semi, "';'")?;
                        Ok(Stmt::StoreIndex(name, idx, val))
                    }
                    _ => err(
                        self.line(),
                        format!(
                            "expected '=', '(' or '[' after '{name}', found {:?}",
                            self.peek()
                        ),
                    ),
                }
            }
            other => err(
                self.line(),
                format!("expected a statement, found {other:?}"),
            ),
        }
    }

    // ---- functions / program ----

    /// Parse a function whose `int name` prefix has already been consumed (starts at `(`).
    fn parse_func_rest(&mut self, name: String) -> Result<Func, ParseError> {
        self.expect(&Tok::LParen, "'('")?;
        let mut params = Vec::new();
        if self.peek() != &Tok::RParen {
            loop {
                self.expect(&Tok::KwInt, "'int' (parameter type)")?;
                params.push(self.expect_ident("parameter name")?);
                if self.peek() == &Tok::Comma {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen, "')'")?;
        let body = self.parse_block()?;
        Ok(Func { name, params, body })
    }

    /// Parse a global array whose `int name` prefix has been consumed (starts at `[`).
    /// Forms: `int name[SIZE];` (zero-init) or `int name[] = { v0, v1, ... };` (size
    /// inferred from the initializer). Sprint 382, Gate F.
    fn parse_global_rest(&mut self, name: String) -> Result<GlobalArray, ParseError> {
        self.expect(&Tok::LBracket, "'['")?;
        // Optional explicit size.
        let explicit_len = match self.peek().clone() {
            Tok::Int(n) => {
                self.bump();
                if n <= 0 {
                    return err(self.line(), "array size must be positive");
                }
                Some(n as usize)
            }
            _ => None,
        };
        self.expect(&Tok::RBracket, "']'")?;
        let mut init = Vec::new();
        if self.peek() == &Tok::Assign {
            self.bump();
            self.expect(&Tok::LBrace, "'{' (array initializer)")?;
            if self.peek() != &Tok::RBrace {
                loop {
                    match self.peek().clone() {
                        Tok::Int(n) => {
                            self.bump();
                            init.push(n);
                        }
                        other => {
                            return err(
                                self.line(),
                                format!("expected an integer initializer, found {other:?}"),
                            );
                        }
                    }
                    if self.peek() == &Tok::Comma {
                        self.bump();
                        // Allow a trailing comma before `}`.
                        if self.peek() == &Tok::RBrace {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Tok::RBrace, "'}'")?;
        }
        self.expect(&Tok::Semi, "';'")?;
        let len = match (explicit_len, init.is_empty()) {
            (Some(n), _) => n,
            (None, false) => init.len(),
            (None, true) => {
                return err(
                    self.line(),
                    "array needs an explicit size or an initializer (e.g. `int a[8];`)",
                );
            }
        };
        if init.len() > len {
            return err(
                self.line(),
                format!(
                    "initializer has {} elements but array length is {len}",
                    init.len()
                ),
            );
        }
        Ok(GlobalArray { name, len, init })
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut funcs = Vec::new();
        let mut globals = Vec::new();
        let mut accel_funcs = Vec::new();
        while self.peek() != &Tok::Eof {
            // Sprint 388 (HW/SW co-design): `accel int f(...) { ... }` marks a function
            // for HLS synthesis to a tile datapath. `accel` is a *contextual* keyword —
            // only an identifier in this top-level position; everywhere else it remains
            // a normal identifier.
            let is_accel = matches!(self.peek(), Tok::Ident(w) if w == "accel");
            if is_accel {
                self.bump();
            }
            // Every top-level item begins `int name`; the next token decides global vs fn.
            self.expect(&Tok::KwInt, "'int' (top-level declaration)")?;
            let name = self.expect_ident("function or global name")?;
            match self.peek() {
                Tok::LBracket if !is_accel => globals.push(self.parse_global_rest(name)?),
                Tok::LParen => {
                    let func = self.parse_func_rest(name)?;
                    if is_accel {
                        accel_funcs.push(func);
                    } else {
                        funcs.push(func);
                    }
                }
                other => {
                    return err(
                        self.line(),
                        if is_accel {
                            format!(
                                "expected '(' after accel function name '{name}', found {other:?}"
                            )
                        } else {
                            format!(
                                "expected '(' (function) or '[' (global) after '{name}', found {other:?}"
                            )
                        },
                    );
                }
            }
        }
        if funcs.is_empty() {
            return err(self.line(), "empty program (no functions)");
        }
        let mut program = Program::with_globals("main", globals, funcs);
        program.accel_funcs = accel_funcs;
        Ok(program)
    }
}

/// Parse C-like source text into a [`Program`] AST (entry `main`, default stack top).
pub fn parse_program(src: &str) -> Result<Program, ParseError> {
    let toks = tokenize(src)?;
    let mut parser = Parser { toks, pos: 0 };
    parser.parse_program()
}

/// Sprint 376 (Gate E.2) / Sprint 379 (Perf P3): the standard library prelude, written
/// in the language itself (the ISA has no divide/modulo). `div10(n)` = `n / 10` via
/// **chunked** subtraction (subtract 10000/1000/100/10 in tiers) — O(digits) instead of
/// O(n/10), so printing large numbers is feasible. `print_int` computes the remainder
/// digit as `n - q*10` (one MUL+SUB), avoiding a second division pass. Auto-included
/// when `print_int` is referenced but not user-defined. Supports n in 0..=65535.
const PRELUDE_SRC: &str = r#"
    int div10(int n) {
        int q = 0;
        while (n >= 10000) { n = n - 10000; q = q + 1000; }
        while (n >= 1000)  { n = n - 1000;  q = q + 100;  }
        while (n >= 100)   { n = n - 100;   q = q + 10;   }
        while (n >= 10)    { n = n - 10;    q = q + 1;    }
        return q;
    }
    int print_int(int n) {
        if (n >= 10) {
            int q = div10(n);
            print_int(q);
            n = n - q * 10;
        }
        putchar(48 + n);
        return 0;
    }
"#;

/// Collect every function name that appears in a `Call(...)` anywhere in the program.
fn collect_called_names(funcs: &[Func]) -> std::collections::HashSet<String> {
    fn walk_expr(e: &Expr, out: &mut std::collections::HashSet<String>) {
        match e {
            Expr::Call(name, args) => {
                out.insert(name.clone());
                for a in args {
                    walk_expr(a, out);
                }
            }
            Expr::Bin(_, a, b) => {
                walk_expr(a, out);
                walk_expr(b, out);
            }
            Expr::Index(_, idx) => walk_expr(idx, out),
            Expr::Int(_) | Expr::Var(_) => {}
        }
    }
    fn walk_cond(c: &Cond, out: &mut std::collections::HashSet<String>) {
        let Cond::Cmp(_, a, b) = c;
        walk_expr(a, out);
        walk_expr(b, out);
    }
    fn walk_stmts(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
        for s in stmts {
            match s {
                Stmt::Let(_, e) | Stmt::Assign(_, e) | Stmt::Return(e) | Stmt::Expr(e) => {
                    walk_expr(e, out)
                }
                Stmt::StoreIndex(_, idx, val) => {
                    walk_expr(idx, out);
                    walk_expr(val, out);
                }
                Stmt::If(c, t, e) => {
                    walk_cond(c, out);
                    walk_stmts(t, out);
                    walk_stmts(e, out);
                }
                Stmt::While(c, b) => {
                    walk_cond(c, out);
                    walk_stmts(b, out);
                }
            }
        }
    }
    let mut out = std::collections::HashSet::new();
    for f in funcs {
        walk_stmts(&f.body, &mut out);
    }
    out
}

/// Auto-include prelude functions (`print_int` + its `div10`/`mod10` helpers) when the
/// program calls them but doesn't define them. A user-supplied definition wins.
/// `pub(crate)` since Sprint 388 so `v2_hls_accel::compile_source_with_accel` shares it.
pub(crate) fn inject_prelude(program: &mut Program) -> Result<(), ParseError> {
    let called = collect_called_names(&program.funcs);
    let defined = |p: &Program, name: &str| p.funcs.iter().any(|f| f.name == name);
    if called.contains("print_int") && !defined(program, "print_int") {
        let prelude = parse_program(PRELUDE_SRC)?;
        for f in prelude.funcs {
            if !defined(program, &f.name) {
                program.funcs.push(f);
            }
        }
    }
    Ok(())
}

/// Parse + compile source text straight to machine words (auto-including the prelude).
///
/// Rejects `accel`-marked source: the accelerator device the lowered code talks to
/// must be synthesized and registered, which this `Vec<u32>`-only signature cannot
/// return — use [`compile_source_with_accel`]
/// (crate::tile_cpu::v2_hls_accel::compile_source_with_accel) instead.
pub fn compile_source(src: &str) -> Result<Vec<u32>, CompileError> {
    let mut program = parse_program(src).map_err(|e| CompileError(e.to_string()))?;
    inject_prelude(&mut program).map_err(|e| CompileError(e.to_string()))?;
    if !program.accel_funcs.is_empty() {
        return Err(CompileError(
            "source declares an accel function; use compile_source_with_accel, which \
             returns the synthesized accelerator device alongside the program words"
                .to_string(),
        ));
    }
    compile_to_words(&program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_cpu::v2_compiler::{call, int, lt, mul, sub, var};
    use crate::tile_cpu::v2_iss::V2Iss;

    const FACT_SRC: &str = r#"
        // recursive factorial
        int fact(int n) {
            if (n < 2) { return 1; }
            return n * fact(n - 1);
        }
        int main() {
            return fact(5);
        }
    "#;

    fn run_source(src: &str) -> u64 {
        let words = compile_source(src).expect("compile_source failed");
        let mut iss = V2Iss::from_program(&words);
        iss.enable_wide_pc();
        for _ in 0..2_000_000 {
            if iss.state().halted {
                return iss.state().regs[0];
            }
            iss.step();
        }
        panic!("source program did not halt within the step budget");
    }

    /// Run a source program on the ISS with MMIO interception and return the captured
    /// console output (Gate E).
    fn run_source_console(src: &str) -> String {
        let words = compile_source(src).expect("compile_source failed");
        let mut iss = V2Iss::from_program_with_mmio(&words);
        iss.enable_wide_pc();
        for _ in 0..5_000_000 {
            if iss.state().halted {
                return iss.console_string();
            }
            iss.step();
        }
        panic!("source program did not halt within the step budget");
    }

    /// Sprint 376: the `div10`/`mod10` helpers (repeated subtraction) compute n/10 and
    /// n%10 — validated via return value (no MMIO needed).
    #[test]
    fn test_v2_parser_div10_mod10_iss() {
        let div10 = r#"
            int div10(int n) { int q = 0; while (n >= 10) { n = n - 10; q = q + 1; } return q; }
            int main() { return div10(255); }
        "#;
        assert_eq!(run_source(div10), 25);
        let mod10 = r#"
            int mod10(int n) { while (n >= 10) { n = n - 10; } return n; }
            int main() { return mod10(255); }
        "#;
        assert_eq!(run_source(mod10), 5);
    }

    /// Sprint 376: `print_int` (auto-included prelude) prints decimal numbers. Covers
    /// 0, a single digit, a 3-digit, and a 5-digit value (ISS).
    #[test]
    fn test_v2_parser_print_int_iss() {
        let out = run_source_console(
            r#"
            int main() {
                print_int(0);     putchar(32);
                print_int(7);     putchar(32);
                print_int(255);   putchar(32);
                print_int(12345);
                return 0;
            }
        "#,
        );
        assert_eq!(out, "0 7 255 12345");
    }

    /// The parsed source produces exactly the hand-built D.1 AST.
    #[test]
    fn test_v2_parser_factorial_ast_roundtrip() {
        let parsed = parse_program(FACT_SRC).expect("parse failed");
        // The source uses early-return: an `if` with an empty else, followed by a
        // sibling `return` — not an if/else.
        let fact = Func {
            name: "fact".to_string(),
            params: vec!["n".to_string()],
            body: vec![
                Stmt::If(lt(var("n"), int(2)), vec![Stmt::Return(int(1))], vec![]),
                Stmt::Return(mul(var("n"), call("fact", vec![sub(var("n"), int(1))]))),
            ],
        };
        let main = Func {
            name: "main".to_string(),
            params: vec![],
            body: vec![Stmt::Return(call("fact", vec![int(5)]))],
        };
        assert_eq!(parsed, Program::new("main", vec![fact, main]));
    }

    #[test]
    fn test_v2_parser_factorial_iss() {
        assert_eq!(run_source(FACT_SRC), 120);
    }

    #[test]
    fn test_v2_parser_fib_iss() {
        let src = r#"
            int fib(int n) {
                if (n < 2) { return n; }
                return fib(n - 1) + fib(n - 2);
            }
            int main() { return fib(10); }
        "#;
        assert_eq!(run_source(src), 55);
    }

    #[test]
    fn test_v2_parser_gcd_iss() {
        // Subtraction-based gcd; exercises while, if/else, !=, >, and bare-truthiness.
        let src = r#"
            int gcd(int a, int b) {
                while (a != b) {
                    if (a > b) { a = a - b; }
                    else { b = b - a; }
                }
                return a;
            }
            int main() { return gcd(48, 36); }
        "#;
        assert_eq!(run_source(src), 12);
    }

    #[test]
    fn test_v2_parser_while_sum_and_hex_iss() {
        // `i <= 10`, assignment, and a 0x hex literal; bare `while (count)` truthiness.
        let src = r#"
            int main() {
                int s = 0;
                int i = 1;
                while (i <= 0x0A) {
                    s = s + i;
                    i = i + 1;
                }
                return s;
            }
        "#;
        assert_eq!(run_source(src), 55);
    }

    /// Sprint 382 (Gate F): source-level global array — fill `a[i]=i*i` then sum = 30.
    #[test]
    fn test_v2_parser_array_fill_sum_iss() {
        let src = r#"
            int a[5];
            int main() {
                int i = 0;
                while (i < 5) { a[i] = i * i; i = i + 1; }
                int s = 0;
                int k = 0;
                while (k < 5) { s = s + a[k]; k = k + 1; }
                return s;
            }
        "#;
        assert_eq!(run_source(src), 30);
    }

    /// Sprint 382 (Gate F): array initializer `{11,22,33}`, read back `data[0]+data[2]`.
    #[test]
    fn test_v2_parser_array_initializer_iss() {
        let src = r#"
            int data[] = { 11, 22, 33 };
            int main() { return data[0] + data[2]; }
        "#;
        assert_eq!(run_source(src), 44);
    }

    /// Sprint 382 (Gate F): a tiny stack machine — `prog` holds push/add opcodes; the
    /// interpreter runs them on an array-backed stack. Proves array-driven dispatch
    /// (a self-hosting-class shape). Program: push 7, push 5, add → 12.
    #[test]
    fn test_v2_parser_array_stack_machine_iss() {
        // opcode 1 = push next cell as value; opcode 2 = add top two. Encoded inline.
        let src = r#"
            int code[] = { 1, 7, 1, 5, 2 };
            int stk[8];
            int run(int n) {
                int pc = 0;
                int sp = 0;
                while (pc < n) {
                    int op = code[pc];
                    if (op == 1) { stk[sp] = code[pc + 1]; sp = sp + 1; pc = pc + 2; }
                    if (op == 2) {
                        int b = stk[sp - 1];
                        int a = stk[sp - 2];
                        stk[sp - 2] = a + b;
                        sp = sp - 1;
                        pc = pc + 1;
                    }
                }
                return stk[0];
            }
            int main() { return run(5); }
        "#;
        assert_eq!(run_source(src), 12);
    }

    #[test]
    fn test_v2_parser_precedence() {
        // 2 + 3 * 4 = 14 (not 20): `*` binds tighter than `+`.
        let src = "int main() { return 2 + 3 * 4; }";
        assert_eq!(run_source(src), 14);
        // Parentheses override: (2 + 3) * 4 = 20.
        let src2 = "int main() { return (2 + 3) * 4; }";
        assert_eq!(run_source(src2), 20);
    }

    /// Sprint 388: the `accel` contextual keyword routes a function into
    /// `Program::accel_funcs` (HLS synthesis target) instead of `funcs`.
    #[test]
    fn test_v2_parser_accel_marker() {
        let src = r#"
            accel int madd(int a, int b, int c) { return a * b + c; }
            int main() { return madd(2, 3, 4); }
        "#;
        let prog = parse_program(src).expect("parse");
        assert_eq!(prog.accel_funcs.len(), 1);
        assert_eq!(prog.accel_funcs[0].name, "madd");
        assert_eq!(prog.accel_funcs[0].params, vec!["a", "b", "c"]);
        assert_eq!(prog.funcs.len(), 1, "main stays a regular function");

        // `accel` stays an ordinary identifier everywhere else.
        let src2 = "int main() { int accel = 7; return accel + 1; }";
        assert_eq!(run_source(src2), 8);

        // Plain compile_source must reject accel-marked source with a pointer to
        // the device-returning entry point.
        let err = compile_source(src).unwrap_err();
        assert!(
            err.0.contains("compile_source_with_accel"),
            "error should point at the SoC entry point: {err}"
        );
    }

    #[test]
    fn test_v2_parser_errors() {
        // Missing semicolon.
        assert!(parse_program("int main() { return 1 }").is_err());
        // Unbalanced paren.
        assert!(parse_program("int main() { return (1; }").is_err());
        // Comparison used as a value.
        assert!(parse_program("int main() { return 1 < 2; }").is_err());
        // Bad character.
        let e = parse_program("int main() { return 1 @ 2; }").unwrap_err();
        assert!(e.message.contains("unexpected character"), "{e:?}");
        // Line number is tracked (the '@' is on line 1 here).
        assert_eq!(e.line, 1);
    }
}
