use super::HdlError;
use super::ast::*;
use super::token::{Span, Token};

/// Parse a token stream into a list of module declarations.
///
/// This is a recursive descent parser. On error it attempts to skip to the
/// next `;` or `}` and continue, collecting multiple errors.
pub fn parse(tokens: &[(Token, Span)]) -> Result<Vec<ModuleDecl>, Vec<HdlError>> {
    let mut parser = Parser {
        tokens,
        pos: 0,
        errors: Vec::new(),
    };
    let modules = parser.parse_program();
    if parser.errors.is_empty() {
        Ok(modules)
    } else {
        Err(parser.errors)
    }
}

struct Parser<'a> {
    tokens: &'a [(Token, Span)],
    pos: usize,
    errors: Vec<HdlError>,
}

impl<'a> Parser<'a> {
    // ---- helpers ----

    fn peek(&self) -> &Token {
        &self.tokens[self.pos].0
    }

    fn span(&self) -> &Span {
        &self.tokens[self.pos].1
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos].0;
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<(), HdlError> {
        if self.peek() == expected {
            self.advance();
            Ok(())
        } else {
            Err(self.error(format!("expected {:?}, found {:?}", expected, self.peek())))
        }
    }

    fn expect_ident(&mut self) -> Result<String, HdlError> {
        match self.peek().clone() {
            Token::Ident(name) => {
                self.advance();
                Ok(name)
            }
            other => Err(self.error(format!("expected identifier, found {:?}", other))),
        }
    }

    fn expect_int(&mut self) -> Result<u64, HdlError> {
        match self.peek().clone() {
            Token::IntLiteral(v) => {
                self.advance();
                Ok(v)
            }
            other => Err(self.error(format!("expected integer literal, found {:?}", other))),
        }
    }

    fn error(&self, message: String) -> HdlError {
        let span = self.span();
        HdlError {
            message,
            line: span.line,
            col: span.col,
        }
    }

    /// Skip tokens until we find a `;` or `}` (error recovery).
    fn skip_to_sync(&mut self) {
        loop {
            match self.peek() {
                Token::Semicolon => {
                    self.advance();
                    return;
                }
                Token::RBrace | Token::Eof => return,
                _ => {
                    self.advance();
                }
            }
        }
    }

    // ---- grammar rules ----

    fn parse_program(&mut self) -> Vec<ModuleDecl> {
        let mut modules = Vec::new();
        while *self.peek() != Token::Eof {
            match self.parse_module() {
                Ok(m) => modules.push(m),
                Err(e) => {
                    self.errors.push(e);
                    self.skip_to_sync();
                }
            }
        }
        modules
    }

    fn parse_module(&mut self) -> Result<ModuleDecl, HdlError> {
        self.expect(&Token::Module)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LParen)?;

        let ports = if *self.peek() != Token::RParen {
            self.parse_port_list()?
        } else {
            Vec::new()
        };

        self.expect(&Token::RParen)?;
        self.expect(&Token::LBrace)?;

        let mut body = Vec::new();
        while *self.peek() != Token::RBrace && *self.peek() != Token::Eof {
            match self.parse_statement() {
                Ok(stmt) => body.push(stmt),
                Err(e) => {
                    self.errors.push(e);
                    self.skip_to_sync();
                }
            }
        }

        self.expect(&Token::RBrace)?;

        Ok(ModuleDecl { name, ports, body })
    }

    fn parse_port_list(&mut self) -> Result<Vec<PortDecl>, HdlError> {
        let mut ports = Vec::new();
        ports.push(self.parse_port_decl()?);
        while *self.peek() == Token::Comma {
            self.advance(); // consume ','
            ports.push(self.parse_port_decl()?);
        }
        Ok(ports)
    }

    fn parse_port_decl(&mut self) -> Result<PortDecl, HdlError> {
        let direction = match self.peek() {
            Token::Input => {
                self.advance();
                PortDir::Input
            }
            Token::Output => {
                self.advance();
                PortDir::Output
            }
            other => {
                return Err(self.error(format!("expected 'input' or 'output', found {:?}", other)));
            }
        };
        let name = self.expect_ident()?;
        Ok(PortDecl { name, direction })
    }

    fn parse_statement(&mut self) -> Result<Statement, HdlError> {
        match self.peek() {
            Token::Tile => self.parse_tile_stmt(),
            Token::Wire => self.parse_wire_stmt(),
            Token::Const => self.parse_const_stmt(),
            Token::Inst => self.parse_inst_stmt(),
            Token::Asm => self.parse_asm_stmt(),
            other => Err(self.error(format!(
                "expected statement (tile/wire/const/inst/asm), found {:?}",
                other
            ))),
        }
    }

    /// tile IDENT = IDENT ("@" "(" INT "," INT ")")? ";"
    fn parse_tile_stmt(&mut self) -> Result<Statement, HdlError> {
        self.expect(&Token::Tile)?;
        let name = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        let tile_type = self.expect_ident()?;

        let position = self.parse_optional_position()?;

        self.expect(&Token::Semicolon)?;

        Ok(Statement::TileDecl {
            name,
            tile_type,
            position,
            initial_value: None,
        })
    }

    /// wire IDENT ":" endpoint "->" endpoint ";"
    fn parse_wire_stmt(&mut self) -> Result<Statement, HdlError> {
        self.expect(&Token::Wire)?;
        let name = self.expect_ident()?;
        self.expect(&Token::Colon)?;
        let from = self.parse_wire_endpoint()?;
        self.expect(&Token::Arrow)?;
        let to = self.parse_wire_endpoint()?;
        self.expect(&Token::Semicolon)?;

        Ok(Statement::WireDecl { name, from, to })
    }

    /// const IDENT "=" INT ";"
    fn parse_const_stmt(&mut self) -> Result<Statement, HdlError> {
        self.expect(&Token::Const)?;
        let name = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        let value = self.expect_int()?;
        self.expect(&Token::Semicolon)?;

        Ok(Statement::ConstDecl { name, value })
    }

    /// inst IDENT "=" IDENT "(" arg_list? ")" ("@" "(" INT "," INT ")")? ";"
    fn parse_inst_stmt(&mut self) -> Result<Statement, HdlError> {
        self.expect(&Token::Inst)?;
        let name = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        let module_name = self.expect_ident()?;
        self.expect(&Token::LParen)?;

        let mut args = Vec::new();
        if *self.peek() != Token::RParen {
            args.push(self.expect_ident()?);
            while *self.peek() == Token::Comma {
                self.advance();
                args.push(self.expect_ident()?);
            }
        }

        self.expect(&Token::RParen)?;
        let position = self.parse_optional_position()?;
        self.expect(&Token::Semicolon)?;

        Ok(Statement::InstDecl {
            name,
            module_name,
            args,
            position,
        })
    }

    /// asm IDENT "=" "{" StringLiteral "}" ";"
    fn parse_asm_stmt(&mut self) -> Result<Statement, HdlError> {
        self.expect(&Token::Asm)?;
        let name = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        self.expect(&Token::LBrace)?;

        // The lexer should have produced a StringLiteral for the asm body
        let source = match self.peek().clone() {
            Token::StringLiteral(s) => {
                self.advance();
                s
            }
            other => {
                return Err(self.error(format!("expected asm body, found {:?}", other)));
            }
        };

        self.expect(&Token::RBrace)?;
        self.expect(&Token::Semicolon)?;

        Ok(Statement::AsmDecl { name, source })
    }

    /// Parse an optional position: "@" "(" INT "," INT ")"
    fn parse_optional_position(&mut self) -> Result<Option<Position>, HdlError> {
        if *self.peek() == Token::At {
            self.advance(); // consume '@'
            self.expect(&Token::LParen)?;
            let x = self.expect_int()? as u32;
            self.expect(&Token::Comma)?;
            let y = self.expect_int()? as u32;
            self.expect(&Token::RParen)?;
            Ok(Some(Position { x, y }))
        } else {
            Ok(None)
        }
    }

    /// Parse a wire endpoint: IDENT ("." IDENT)?
    fn parse_wire_endpoint(&mut self) -> Result<WireEndpoint, HdlError> {
        let first = self.expect_ident()?;
        if *self.peek() == Token::Dot {
            self.advance(); // consume '.'
            let second = self.expect_ident()?;
            Ok(WireEndpoint::QualifiedPort(first, second))
        } else {
            // Will be resolved to either Port or Tile during semantic analysis.
            // For now, store as Port (the semantic pass will distinguish).
            Ok(WireEndpoint::Port(first))
        }
    }
}
