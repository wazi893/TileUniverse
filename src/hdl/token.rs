/// Span tracks source location for error reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

/// Token types produced by the HDL lexer.
///
/// Note: Tile type names (And, Or, Wire, etc.) are lexed as `Ident` tokens.
/// They are resolved to actual TileType values during semantic analysis,
/// using `parse_tile_type_token` from blueprint.rs.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Module,
    Input,
    Output,
    Tile,
    Wire,
    Const,
    Inst,
    Asm,

    // Symbols
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Colon,
    Eq,    // =
    Arrow, // ->
    At,    // @
    Dot,   // .

    // Literals
    Ident(String),
    IntLiteral(u64),
    StringLiteral(String), // for asm block contents

    // Meta
    Eof,
}
