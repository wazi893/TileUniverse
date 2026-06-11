//! OpenQASM 3.0 Lexer
//!
//! Tokenizes QASM3 source text into a stream of tokens.

use std::iter::Peekable;
use std::str::Chars;

/// Token types in QASM3
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // Keywords
    OpenQASM,
    Include,
    Qubit,
    Bit,
    Measure,
    Reset,
    Barrier,
    If,
    Gate,
    Def,
    Return,

    // Literals
    Identifier(String),
    Integer(i64),
    Float(f64),
    String(String),

    // Operators and punctuation
    Semicolon,    // ;
    Comma,        // ,
    Colon,        // :
    Arrow,        // ->
    Equals,       // =
    LeftParen,    // (
    RightParen,   // )
    LeftBracket,  // [
    RightBracket, // ]
    LeftBrace,    // {
    RightBrace,   // }
    Plus,         // +
    Minus,        // -
    Star,         // *
    Slash,        // /
    Caret,        // ^

    // Special
    Pi,      // pi constant
    Tau,     // tau constant (2π)
    Euler,   // e constant
    Comment, // // comment (skipped)
    Newline, // \n (usually skipped)
    Eof,     // End of file
}

/// A token with position information
#[derive(Clone, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(kind: TokenKind, line: usize, column: usize) -> Self {
        Self { kind, line, column }
    }
}

/// QASM3 Lexer
pub struct Lexer<'a> {
    input: Peekable<Chars<'a>>,
    line: usize,
    column: usize,
    current_char: Option<char>,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given input
    pub fn new(input: &'a str) -> Self {
        let mut lexer = Self {
            input: input.chars().peekable(),
            line: 1,
            column: 0,
            current_char: None,
        };
        lexer.advance();
        lexer
    }

    /// Advance to the next character
    fn advance(&mut self) -> Option<char> {
        let prev = self.current_char;
        self.current_char = self.input.next();
        if let Some(c) = self.current_char {
            if c == '\n' {
                self.line += 1;
                self.column = 0;
            } else {
                self.column += 1;
            }
        }
        prev
    }

    /// Peek at the current character without consuming
    fn peek(&self) -> Option<char> {
        self.current_char
    }

    /// Peek at the next character
    fn peek_next(&mut self) -> Option<char> {
        self.input.peek().copied()
    }

    /// Skip whitespace (except newlines if needed)
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Skip a line comment
    fn skip_line_comment(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.advance();
        }
    }

    /// Read an identifier or keyword
    fn read_identifier(&mut self) -> String {
        let mut result = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                result.push(c);
                self.advance();
            } else {
                break;
            }
        }
        result
    }

    /// Read a number (integer or float)
    fn read_number(&mut self) -> TokenKind {
        let mut result = String::new();
        let mut is_float = false;

        // Handle negative numbers
        if self.peek() == Some('-') {
            result.push('-');
            self.advance();
        }

        // Integer part
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                result.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Decimal part
        if self.peek() == Some('.') {
            if let Some(next) = self.peek_next() {
                if next.is_ascii_digit() {
                    is_float = true;
                    result.push('.');
                    self.advance();

                    while let Some(c) = self.peek() {
                        if c.is_ascii_digit() {
                            result.push(c);
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        // Exponent part
        if let Some(c) = self.peek() {
            if c == 'e' || c == 'E' {
                is_float = true;
                result.push(c);
                self.advance();

                if let Some(sign) = self.peek() {
                    if sign == '+' || sign == '-' {
                        result.push(sign);
                        self.advance();
                    }
                }

                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        result.push(c);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        if is_float {
            TokenKind::Float(result.parse().unwrap_or(0.0))
        } else {
            TokenKind::Integer(result.parse().unwrap_or(0))
        }
    }

    /// Read a string literal
    fn read_string(&mut self) -> String {
        let quote = self.peek().unwrap();
        self.advance(); // consume opening quote

        let mut result = String::new();
        while let Some(c) = self.peek() {
            if c == quote {
                self.advance(); // consume closing quote
                break;
            } else if c == '\\' {
                self.advance();
                if let Some(escaped) = self.peek() {
                    match escaped {
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        '\\' => result.push('\\'),
                        '"' => result.push('"'),
                        '\'' => result.push('\''),
                        _ => result.push(escaped),
                    }
                    self.advance();
                }
            } else {
                result.push(c);
                self.advance();
            }
        }
        result
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let line = self.line;
        let column = self.column;

        let kind = match self.peek() {
            None => TokenKind::Eof,

            Some(c) => match c {
                // Comments
                '/' => {
                    self.advance();
                    if self.peek() == Some('/') {
                        self.skip_line_comment();
                        return self.next_token(); // Skip comment, get next
                    } else {
                        TokenKind::Slash
                    }
                }

                // Punctuation
                ';' => {
                    self.advance();
                    TokenKind::Semicolon
                }
                ',' => {
                    self.advance();
                    TokenKind::Comma
                }
                ':' => {
                    self.advance();
                    TokenKind::Colon
                }
                '(' => {
                    self.advance();
                    TokenKind::LeftParen
                }
                ')' => {
                    self.advance();
                    TokenKind::RightParen
                }
                '[' => {
                    self.advance();
                    TokenKind::LeftBracket
                }
                ']' => {
                    self.advance();
                    TokenKind::RightBracket
                }
                '{' => {
                    self.advance();
                    TokenKind::LeftBrace
                }
                '}' => {
                    self.advance();
                    TokenKind::RightBrace
                }
                '+' => {
                    self.advance();
                    TokenKind::Plus
                }
                '*' => {
                    self.advance();
                    TokenKind::Star
                }
                '^' => {
                    self.advance();
                    TokenKind::Caret
                }

                // Arrow or minus
                '-' => {
                    self.advance();
                    if self.peek() == Some('>') {
                        self.advance();
                        TokenKind::Arrow
                    } else if self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        // Negative number
                        let mut s = String::from("-");
                        while let Some(c) = self.peek() {
                            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' {
                                s.push(c);
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        if s.contains('.') || s.contains('e') || s.contains('E') {
                            TokenKind::Float(s.parse().unwrap_or(0.0))
                        } else {
                            TokenKind::Integer(s.parse().unwrap_or(0))
                        }
                    } else {
                        TokenKind::Minus
                    }
                }

                // Equals
                '=' => {
                    self.advance();
                    TokenKind::Equals
                }

                // Strings
                '"' | '\'' => {
                    let s = self.read_string();
                    TokenKind::String(s)
                }

                // Numbers
                '0'..='9' => self.read_number(),

                // Identifiers and keywords
                'a'..='z' | 'A'..='Z' | '_' => {
                    let ident = self.read_identifier();
                    match ident.as_str() {
                        "OPENQASM" => TokenKind::OpenQASM,
                        "include" => TokenKind::Include,
                        "qubit" => TokenKind::Qubit,
                        "bit" => TokenKind::Bit,
                        "measure" => TokenKind::Measure,
                        "reset" => TokenKind::Reset,
                        "barrier" => TokenKind::Barrier,
                        "if" => TokenKind::If,
                        "gate" => TokenKind::Gate,
                        "def" => TokenKind::Def,
                        "return" => TokenKind::Return,
                        "pi" => TokenKind::Pi,
                        "tau" => TokenKind::Tau,
                        "euler" | "e" => TokenKind::Euler,
                        _ => TokenKind::Identifier(ident),
                    }
                }

                // Unknown character - skip
                _ => {
                    self.advance();
                    return self.next_token();
                }
            },
        };

        Token::new(kind, line, column)
    }

    /// Tokenize the entire input into a vector
    pub fn tokenize(mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            let is_eof = matches!(token.kind, TokenKind::Eof);
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokens() {
        let input = "h q[0];";
        let tokens = Lexer::new(input).tokenize();

        assert!(matches!(tokens[0].kind, TokenKind::Identifier(ref s) if s == "h"));
        assert!(matches!(tokens[1].kind, TokenKind::Identifier(ref s) if s == "q"));
        assert!(matches!(tokens[2].kind, TokenKind::LeftBracket));
        assert!(matches!(tokens[3].kind, TokenKind::Integer(0)));
        assert!(matches!(tokens[4].kind, TokenKind::RightBracket));
        assert!(matches!(tokens[5].kind, TokenKind::Semicolon));
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is literal lexer test data, not π
    fn test_numbers() {
        let input = "42 3.14 -1 1e-5";
        let tokens = Lexer::new(input).tokenize();

        assert!(matches!(tokens[0].kind, TokenKind::Integer(42)));
        assert!(matches!(tokens[1].kind, TokenKind::Float(f) if (f - 3.14).abs() < 0.001));
        assert!(matches!(tokens[2].kind, TokenKind::Integer(-1)));
        assert!(matches!(tokens[3].kind, TokenKind::Float(f) if (f - 1e-5).abs() < 1e-10));
    }

    #[test]
    fn test_keywords() {
        let input = "OPENQASM include qubit bit measure";
        let tokens = Lexer::new(input).tokenize();

        assert!(matches!(tokens[0].kind, TokenKind::OpenQASM));
        assert!(matches!(tokens[1].kind, TokenKind::Include));
        assert!(matches!(tokens[2].kind, TokenKind::Qubit));
        assert!(matches!(tokens[3].kind, TokenKind::Bit));
        assert!(matches!(tokens[4].kind, TokenKind::Measure));
    }

    #[test]
    fn test_string() {
        let input = r#"include "stdgates.inc";"#;
        let tokens = Lexer::new(input).tokenize();

        assert!(matches!(tokens[0].kind, TokenKind::Include));
        assert!(matches!(tokens[1].kind, TokenKind::String(ref s) if s == "stdgates.inc"));
        assert!(matches!(tokens[2].kind, TokenKind::Semicolon));
    }

    #[test]
    fn test_comments() {
        let input = "h q[0]; // this is a comment\nx q[1];";
        let tokens = Lexer::new(input).tokenize();

        // Comment should be skipped
        let identifiers: Vec<_> = tokens
            .iter()
            .filter_map(|t| {
                if let TokenKind::Identifier(s) = &t.kind {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();

        assert!(identifiers.contains(&"h"));
        assert!(identifiers.contains(&"x"));
    }

    #[test]
    fn test_arrow() {
        let input = "measure q[0] -> c[0];";
        let tokens = Lexer::new(input).tokenize();

        assert!(matches!(tokens[0].kind, TokenKind::Measure));
        assert!(tokens.iter().any(|t| matches!(t.kind, TokenKind::Arrow)));
    }
}
