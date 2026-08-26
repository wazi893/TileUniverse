use super::HdlError;
use super::token::{Span, Token};

/// Tokenize HDL source text into a vector of (Token, Span) pairs.
///
/// Handles:
/// - `//` line comments
/// - `/* */` block comments
/// - Keywords: module, input, output, tile, wire, const, inst, asm
/// - Symbols: ( ) { } , ; : = -> @ .
/// - Integer literals: decimal and 0x hex
/// - Identifiers: [a-zA-Z_][a-zA-Z0-9_]*
/// - Asm block bodies: after `asm IDENT = {`, everything until `}` is a StringLiteral
pub fn tokenize(source: &str) -> Result<Vec<(Token, Span)>, Vec<HdlError>> {
    let mut tokens = Vec::new();
    let mut errors = Vec::new();
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut pos = 0;
    let mut line: usize = 1;
    let mut col: usize = 1;

    // Track whether we're inside an asm block body.
    let mut asm_brace_depth: Option<usize> = None;

    while pos < len {
        // If inside an asm block body, consume until matching `}`
        if let Some(depth) = asm_brace_depth {
            let start_line = line;
            let start_col = col;
            let mut body = String::new();
            let mut cur_depth = depth;

            while pos < len {
                let ch = bytes[pos] as char;
                if ch == '{' {
                    cur_depth += 1;
                    body.push(ch);
                    advance(&mut pos, &mut col);
                } else if ch == '}' {
                    if cur_depth == 1 {
                        // End of asm block body
                        tokens.push((
                            Token::StringLiteral(body.trim().to_string()),
                            Span {
                                line: start_line,
                                col: start_col,
                            },
                        ));
                        tokens.push((Token::RBrace, Span { line, col }));
                        advance(&mut pos, &mut col);
                        asm_brace_depth = None;
                        break;
                    } else {
                        cur_depth -= 1;
                        body.push(ch);
                        advance(&mut pos, &mut col);
                    }
                } else if ch == '\n' {
                    body.push(ch);
                    pos += 1;
                    line += 1;
                    col = 1;
                } else {
                    body.push(ch);
                    advance(&mut pos, &mut col);
                }
            }
            if asm_brace_depth.is_some() {
                errors.push(HdlError {
                    message: "unterminated asm block".to_string(),
                    line: start_line,
                    col: start_col,
                });
                asm_brace_depth = None;
            }
            continue;
        }

        let ch = bytes[pos] as char;

        // Skip whitespace
        if ch.is_ascii_whitespace() {
            if ch == '\n' {
                line += 1;
                col = 1;
                pos += 1;
            } else {
                advance(&mut pos, &mut col);
            }
            continue;
        }

        // Skip // line comments
        if ch == '/' && pos + 1 < len && bytes[pos + 1] == b'/' {
            advance(&mut pos, &mut col);
            advance(&mut pos, &mut col);
            while pos < len && bytes[pos] != b'\n' {
                advance(&mut pos, &mut col);
            }
            continue;
        }

        // Skip /* */ block comments
        if ch == '/' && pos + 1 < len && bytes[pos + 1] == b'*' {
            let start_line = line;
            let start_col = col;
            advance(&mut pos, &mut col);
            advance(&mut pos, &mut col);
            let mut closed = false;
            while pos + 1 < len {
                if bytes[pos] == b'*' && bytes[pos + 1] == b'/' {
                    advance(&mut pos, &mut col);
                    advance(&mut pos, &mut col);
                    closed = true;
                    break;
                }
                if bytes[pos] == b'\n' {
                    line += 1;
                    col = 1;
                    pos += 1;
                } else {
                    advance(&mut pos, &mut col);
                }
            }
            if !closed {
                errors.push(HdlError {
                    message: "unterminated block comment".to_string(),
                    line: start_line,
                    col: start_col,
                });
            }
            continue;
        }

        let span = Span { line, col };

        // Symbols
        match ch {
            '(' => {
                tokens.push((Token::LParen, span));
                advance(&mut pos, &mut col);
                continue;
            }
            ')' => {
                tokens.push((Token::RParen, span));
                advance(&mut pos, &mut col);
                continue;
            }
            '{' => {
                // Check if this opens an asm block: pattern is `Asm Ident Eq {`
                let is_asm_open = is_asm_eq_pattern(&tokens);
                tokens.push((Token::LBrace, span));
                advance(&mut pos, &mut col);
                if is_asm_open {
                    asm_brace_depth = Some(1);
                }
                continue;
            }
            '}' => {
                tokens.push((Token::RBrace, span));
                advance(&mut pos, &mut col);
                continue;
            }
            ',' => {
                tokens.push((Token::Comma, span));
                advance(&mut pos, &mut col);
                continue;
            }
            ';' => {
                tokens.push((Token::Semicolon, span));
                advance(&mut pos, &mut col);
                continue;
            }
            ':' => {
                tokens.push((Token::Colon, span));
                advance(&mut pos, &mut col);
                continue;
            }
            '@' => {
                tokens.push((Token::At, span));
                advance(&mut pos, &mut col);
                continue;
            }
            '.' => {
                tokens.push((Token::Dot, span));
                advance(&mut pos, &mut col);
                continue;
            }
            '-' if pos + 1 < len && bytes[pos + 1] == b'>' => {
                tokens.push((Token::Arrow, span));
                advance(&mut pos, &mut col);
                advance(&mut pos, &mut col);
                continue;
            }
            '=' => {
                tokens.push((Token::Eq, span));
                advance(&mut pos, &mut col);
                continue;
            }
            _ => {}
        }

        // Integer literals
        if ch.is_ascii_digit() {
            let start = pos;
            if ch == '0' && pos + 1 < len && (bytes[pos + 1] == b'x' || bytes[pos + 1] == b'X') {
                // Hex literal
                advance(&mut pos, &mut col);
                advance(&mut pos, &mut col);
                let hex_start = pos;
                while pos < len && (bytes[pos] as char).is_ascii_hexdigit() {
                    advance(&mut pos, &mut col);
                }
                if pos == hex_start {
                    errors.push(HdlError {
                        message: "expected hex digits after 0x".to_string(),
                        line: span.line,
                        col: span.col,
                    });
                    continue;
                }
                let hex_str = &source[hex_start..pos];
                match u64::from_str_radix(hex_str, 16) {
                    Ok(v) => tokens.push((Token::IntLiteral(v), span)),
                    Err(_) => errors.push(HdlError {
                        message: format!("invalid hex literal: 0x{}", hex_str),
                        line: span.line,
                        col: span.col,
                    }),
                }
            } else {
                // Decimal literal
                while pos < len && (bytes[pos] as char).is_ascii_digit() {
                    advance(&mut pos, &mut col);
                }
                let dec_str = &source[start..pos];
                match dec_str.parse::<u64>() {
                    Ok(v) => tokens.push((Token::IntLiteral(v), span)),
                    Err(_) => errors.push(HdlError {
                        message: format!("invalid integer literal: {}", dec_str),
                        line: span.line,
                        col: span.col,
                    }),
                }
            }
            continue;
        }

        // Identifiers and keywords
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start = pos;
            while pos < len && ((bytes[pos] as char).is_ascii_alphanumeric() || bytes[pos] == b'_')
            {
                advance(&mut pos, &mut col);
            }
            let word = &source[start..pos];
            let tok = match word {
                "module" => Token::Module,
                "input" => Token::Input,
                "output" => Token::Output,
                "tile" => Token::Tile,
                "wire" => Token::Wire,
                "const" => Token::Const,
                "inst" => Token::Inst,
                "asm" => Token::Asm,
                _ => Token::Ident(word.to_string()),
            };
            tokens.push((tok, span));
            continue;
        }

        // Unrecognized character
        errors.push(HdlError {
            message: format!("unexpected character: '{}'", ch),
            line: span.line,
            col: span.col,
        });
        advance(&mut pos, &mut col);
    }

    // Append EOF
    tokens.push((Token::Eof, Span { line, col }));

    if errors.is_empty() {
        Ok(tokens)
    } else {
        Err(errors)
    }
}

/// Check if the last 3 tokens form the pattern `Asm Ident Eq`,
/// meaning the next `{` opens an asm block body.
fn is_asm_eq_pattern(tokens: &[(Token, Span)]) -> bool {
    let n = tokens.len();
    if n < 3 {
        return false;
    }
    matches!(
        (&tokens[n - 3].0, &tokens[n - 2].0, &tokens[n - 1].0),
        (Token::Asm, Token::Ident(_), Token::Eq)
    )
}

#[inline]
fn advance(pos: &mut usize, col: &mut usize) {
    *pos += 1;
    *col += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokenize and drop spans, returning just the token kinds (trailing `Eof` included).
    fn kinds(src: &str) -> Vec<Token> {
        tokenize(src)
            .expect("expected successful tokenize")
            .into_iter()
            .map(|(t, _)| t)
            .collect()
    }

    #[test]
    fn empty_source_is_just_eof() {
        assert_eq!(kinds(""), vec![Token::Eof]);
        // Whitespace-only input is likewise empty.
        assert_eq!(kinds("  \n\t "), vec![Token::Eof]);
    }

    #[test]
    fn keywords_lex_to_keyword_tokens() {
        assert_eq!(
            kinds("module input output tile wire const inst asm"),
            vec![
                Token::Module,
                Token::Input,
                Token::Output,
                Token::Tile,
                Token::Wire,
                Token::Const,
                Token::Inst,
                Token::Asm,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn tile_type_names_stay_idents() {
        // Tile type names (And/Or/...) are NOT keywords — they resolve later in semantic analysis.
        assert_eq!(
            kinds("And Or Xor foo_bar _x1"),
            vec![
                Token::Ident("And".to_string()),
                Token::Ident("Or".to_string()),
                Token::Ident("Xor".to_string()),
                Token::Ident("foo_bar".to_string()),
                Token::Ident("_x1".to_string()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn all_symbols_lex() {
        assert_eq!(
            kinds("( ) { } , ; : = -> @ ."),
            vec![
                Token::LParen,
                Token::RParen,
                Token::LBrace,
                Token::RBrace,
                Token::Comma,
                Token::Semicolon,
                Token::Colon,
                Token::Eq,
                Token::Arrow,
                Token::At,
                Token::Dot,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn decimal_and_hex_int_literals() {
        assert_eq!(
            kinds("0 42 255 0xff 0xFF 0X10 0x0"),
            vec![
                Token::IntLiteral(0),
                Token::IntLiteral(42),
                Token::IntLiteral(255),
                Token::IntLiteral(0xff),
                Token::IntLiteral(0xff),
                Token::IntLiteral(0x10),
                Token::IntLiteral(0),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn line_and_block_comments_are_skipped() {
        assert_eq!(
            kinds("wire // trailing comment\n tile /* inline */ input"),
            vec![Token::Wire, Token::Tile, Token::Input, Token::Eof]
        );
        // A block comment spanning newlines is fully consumed.
        assert_eq!(kinds("/* a\n b\n c */ wire"), vec![Token::Wire, Token::Eof]);
    }

    #[test]
    fn asm_block_body_is_a_single_string_literal() {
        // `asm IDENT = { ... }` captures the (trimmed) body verbatim as one StringLiteral.
        assert_eq!(
            kinds("asm blk = {  LDI R0, 5  }"),
            vec![
                Token::Asm,
                Token::Ident("blk".to_string()),
                Token::Eq,
                Token::LBrace,
                Token::StringLiteral("LDI R0, 5".to_string()),
                Token::RBrace,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn asm_block_body_keeps_nested_braces() {
        // Nested braces inside the asm body do not terminate it early.
        assert_eq!(
            kinds("asm b = { a { x } c }"),
            vec![
                Token::Asm,
                Token::Ident("b".to_string()),
                Token::Eq,
                Token::LBrace,
                Token::StringLiteral("a { x } c".to_string()),
                Token::RBrace,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn braces_outside_asm_stay_structural() {
        // A `{` not preceded by `asm IDENT =` is a plain LBrace, and its contents are lexed normally.
        assert_eq!(
            kinds("module m { wire }"),
            vec![
                Token::Module,
                Token::Ident("m".to_string()),
                Token::LBrace,
                Token::Wire,
                Token::RBrace,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn spans_track_line_and_column() {
        let toks = tokenize("module\n  tile").expect("ok");
        // module @ 1:1, tile @ 2:3 (two leading spaces), Eof @ 2:7.
        assert_eq!(toks[0], (Token::Module, Span { line: 1, col: 1 }));
        assert_eq!(toks[1], (Token::Tile, Span { line: 2, col: 3 }));
        assert_eq!(toks[2].0, Token::Eof);
        assert_eq!(toks[2].1, Span { line: 2, col: 7 });
    }

    #[test]
    fn unexpected_char_is_reported_with_location() {
        let errs = tokenize("wire $").expect_err("expected lex error");
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("unexpected character"));
        assert_eq!((errs[0].line, errs[0].col), (1, 6));
    }

    #[test]
    fn unterminated_block_comment_is_an_error() {
        let errs = tokenize("wire /* never closes").expect_err("expected lex error");
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("unterminated block comment"));
        assert_eq!((errs[0].line, errs[0].col), (1, 6));
    }

    #[test]
    fn unterminated_asm_block_is_an_error() {
        let errs = tokenize("asm b = { body without close").expect_err("expected lex error");
        assert!(
            errs.iter()
                .any(|e| e.message.contains("unterminated asm block"))
        );
    }

    #[test]
    fn bare_hex_prefix_is_an_error() {
        let errs = tokenize("0x").expect_err("expected lex error");
        assert!(errs[0].message.contains("expected hex digits after 0x"));
    }
}
