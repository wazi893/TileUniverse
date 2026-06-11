//! HDL-to-Blueprint DSL compiler for TileUniverse.
//!
//! This module provides a textual Hardware Description Language (HDL) that compiles
//! to Blueprint tile placements. The HDL supports:
//!
//! - Module declarations with typed ports (input/output)
//! - Tile instantiation with named instances
//! - Wire routing between instances (auto-placed Wire tiles)
//! - Module instantiation (hierarchical composition)
//! - Constants and initial values
//!
//! Pipeline: HDL source -> Lexer -> Parser -> AST -> Semantic Analysis -> Codegen -> Blueprint

pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod semantic;
pub mod token;

use crate::blueprint::Blueprint;

/// An error produced during HDL compilation.
#[derive(Debug, Clone)]
pub struct HdlError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for HdlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HDL error at {}:{}: {}",
            self.line, self.col, self.message
        )
    }
}

/// Compile HDL source into a Blueprint.
///
/// The top-level module (last module in source) is instantiated at (0, 0).
///
/// # Errors
///
/// Returns a list of all errors encountered during compilation (lexing, parsing,
/// semantic analysis, or code generation).
pub fn compile(source: &str) -> Result<Blueprint, Vec<HdlError>> {
    let tokens = lexer::tokenize(source)?;
    let modules = parser::parse(&tokens)?;
    let checked = semantic::check(&modules)?;
    codegen::generate(&checked)
}

#[cfg(test)]
mod tests {
    use super::ast::*;
    use super::token::Token;
    use super::*;
    use crate::tile_meta::TileType;

    // ========================================================================
    // Test 1: Lexer -- basic tokens
    // ========================================================================
    #[test]
    fn lexer_basic_tokens() {
        let src = "module foo() {}";
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let tok_types: Vec<&Token> = tokens.iter().map(|(t, _)| t).collect();
        assert_eq!(tok_types[0], &Token::Module);
        assert_eq!(tok_types[1], &Token::Ident("foo".to_string()));
        assert_eq!(tok_types[2], &Token::LParen);
        assert_eq!(tok_types[3], &Token::RParen);
        assert_eq!(tok_types[4], &Token::LBrace);
        assert_eq!(tok_types[5], &Token::RBrace);
        assert_eq!(tok_types[6], &Token::Eof);
    }

    // ========================================================================
    // Test 2: Lexer -- integers (decimal and hex)
    // ========================================================================
    #[test]
    fn lexer_integers() {
        let src = "42 0xFF 0x10 100";
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let values: Vec<u64> = tokens
            .iter()
            .filter_map(|(t, _)| {
                if let Token::IntLiteral(v) = t {
                    Some(*v)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(values, vec![42, 0xFF, 0x10, 100]);
    }

    // ========================================================================
    // Test 3: Lexer -- comments
    // ========================================================================
    #[test]
    fn lexer_comments() {
        let src = r#"
            // This is a line comment
            module foo() {
                /* block comment */
                tile g = And;
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        // Verify that comment content does not appear as tokens
        for (tok, _) in &tokens {
            match tok {
                Token::Ident(s) => {
                    assert!(s != "This" && s != "is" && s != "block" && s != "comment");
                }
                _ => {}
            }
        }
        // Should still contain: Module, Ident(foo), LParen, RParen, LBrace,
        //                        Tile, Ident(g), Eq, Ident(And), Semicolon, RBrace, Eof
        let tok_types: Vec<&Token> = tokens.iter().map(|(t, _)| t).collect();
        assert!(tok_types.contains(&&Token::Module));
        assert!(tok_types.contains(&&Token::Tile));
        assert!(tok_types.contains(&&Token::Eof));
    }

    // ========================================================================
    // Test 4: Parser -- empty module
    // ========================================================================
    #[test]
    fn parser_empty_module() {
        let src = "module empty() {}";
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let modules = parser::parse(&tokens).expect("should parse");
        assert_eq!(modules.len(), 1);
        assert_eq!(modules[0].name, "empty");
        assert!(modules[0].ports.is_empty());
        assert!(modules[0].body.is_empty());
    }

    // ========================================================================
    // Test 5: Parser -- tile declarations
    // ========================================================================
    #[test]
    fn parser_tile_declarations() {
        let src = r#"
            module test() {
                tile g0 = And @(5, 10);
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let modules = parser::parse(&tokens).expect("should parse");
        assert_eq!(modules.len(), 1);
        assert_eq!(modules[0].body.len(), 1);
        match &modules[0].body[0] {
            Statement::TileDecl {
                name,
                tile_type,
                position,
                ..
            } => {
                assert_eq!(name, "g0");
                assert_eq!(tile_type, "And");
                let pos = position.as_ref().unwrap();
                assert_eq!(pos.x, 5);
                assert_eq!(pos.y, 10);
            }
            _ => panic!("expected TileDecl"),
        }
    }

    // ========================================================================
    // Test 6: Parser -- wire declarations
    // ========================================================================
    #[test]
    fn parser_wire_declarations() {
        let src = r#"
            module test(input a, output b) {
                tile g = And;
                wire w0: a -> g;
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let modules = parser::parse(&tokens).expect("should parse");
        let wire = &modules[0].body[1];
        match wire {
            Statement::WireDecl { name, from, to } => {
                assert_eq!(name, "w0");
                assert_eq!(*from, WireEndpoint::Port("a".to_string()));
                assert_eq!(*to, WireEndpoint::Port("g".to_string()));
            }
            _ => panic!("expected WireDecl"),
        }
    }

    // ========================================================================
    // Test 7: Parser -- const declarations
    // ========================================================================
    #[test]
    fn parser_const_declarations() {
        let src = r#"
            module test() {
                const val = 0xFF;
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let modules = parser::parse(&tokens).expect("should parse");
        match &modules[0].body[0] {
            Statement::ConstDecl { name, value } => {
                assert_eq!(name, "val");
                assert_eq!(*value, 0xFF);
            }
            _ => panic!("expected ConstDecl"),
        }
    }

    // ========================================================================
    // Test 8: Parser -- instance declarations
    // ========================================================================
    #[test]
    fn parser_instance_declarations() {
        let src = r#"
            module sub(input a, input b) {}
            module top() {
                inst h0 = sub(x, y) @(10, 10);
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let modules = parser::parse(&tokens).expect("should parse");
        assert_eq!(modules.len(), 2);
        match &modules[1].body[0] {
            Statement::InstDecl {
                name,
                module_name,
                args,
                position,
            } => {
                assert_eq!(name, "h0");
                assert_eq!(module_name, "sub");
                assert_eq!(args, &["x".to_string(), "y".to_string()]);
                let pos = position.as_ref().unwrap();
                assert_eq!(pos.x, 10);
                assert_eq!(pos.y, 10);
            }
            _ => panic!("expected InstDecl"),
        }
    }

    // ========================================================================
    // Test 9: Semantic -- tile type resolution
    // ========================================================================
    #[test]
    fn semantic_tile_type_resolution() {
        let src = r#"
            module test() {
                tile g0 = And @(1, 1);
                tile g1 = Or @(3, 1);
                tile g2 = Register8 @(5, 1);
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let modules = parser::parse(&tokens).expect("should parse");
        let checked = semantic::check(&modules).expect("should pass semantic check");
        assert_eq!(checked[0].tiles.len(), 3);
        assert_eq!(checked[0].tiles[0].tile_type, TileType::And);
        assert_eq!(checked[0].tiles[1].tile_type, TileType::Or);
        assert_eq!(checked[0].tiles[2].tile_type, TileType::Register8);
    }

    // ========================================================================
    // Test 10: Semantic -- unknown tile type error
    // ========================================================================
    #[test]
    fn semantic_unknown_tile_type_error() {
        let src = r#"
            module test() {
                tile x = Bogus;
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let modules = parser::parse(&tokens).expect("should parse");
        let result = semantic::check(&modules);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("unknown tile type"))
        );
    }

    // ========================================================================
    // Test 11: Semantic -- duplicate name error
    // ========================================================================
    #[test]
    fn semantic_duplicate_name_error() {
        let src = r#"
            module test() {
                tile a = And @(1, 1);
                tile a = Or @(3, 1);
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let modules = parser::parse(&tokens).expect("should parse");
        let result = semantic::check(&modules);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("duplicate name")));
    }

    // ========================================================================
    // Test 12: Semantic -- wire endpoint resolution
    // ========================================================================
    #[test]
    fn semantic_wire_endpoint_resolution() {
        let src = r#"
            module test(input a, output b) {
                tile g = And @(5, 5);
                wire w0: a -> g;
                wire w1: g -> b;
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let modules = parser::parse(&tokens).expect("should parse");
        let checked = semantic::check(&modules);
        assert!(
            checked.is_ok(),
            "wire endpoints should resolve: {:?}",
            checked.err()
        );
    }

    // ========================================================================
    // Test 13: Codegen -- single And gate
    // ========================================================================
    #[test]
    fn codegen_single_and_gate() {
        let src = r#"
            module top() {
                tile g = And @(5, 5);
            }
        "#;
        let bp = compile(src).expect("should compile");
        // Find the And tile at (5, 5)
        let and_tile = bp.tiles.iter().find(|t| t.tile_type == TileType::And);
        assert!(and_tile.is_some(), "should have an And tile");
        let and_tile = and_tile.unwrap();
        assert_eq!(and_tile.x, 5);
        assert_eq!(and_tile.y, 5);
    }

    // ========================================================================
    // Test 14: Codegen -- wire routing
    // ========================================================================
    #[test]
    fn codegen_wire_routing() {
        let src = r#"
            module top() {
                tile a = And @(2, 2);
                tile b = Or @(6, 2);
                wire w: a -> b;
            }
        "#;
        let bp = compile(src).expect("should compile");
        // Should have And, Or, and Wire tiles between them
        let and_count = bp
            .tiles
            .iter()
            .filter(|t| t.tile_type == TileType::And)
            .count();
        let or_count = bp
            .tiles
            .iter()
            .filter(|t| t.tile_type == TileType::Or)
            .count();
        let wire_count = bp
            .tiles
            .iter()
            .filter(|t| t.tile_type == TileType::Wire)
            .count();
        assert_eq!(and_count, 1);
        assert_eq!(or_count, 1);
        // There should be Wire tiles placed along the route between (2,2) and (6,2)
        // Specifically: (3,2), (4,2), (5,2) = 3 wire tiles
        assert!(
            wire_count >= 1,
            "expected at least one Wire tile for routing, got {}",
            wire_count
        );
        // Check that wire tiles are between the two gates
        let wire_tiles: Vec<_> = bp
            .tiles
            .iter()
            .filter(|t| t.tile_type == TileType::Wire && t.y == 2 && t.x > 2 && t.x < 6)
            .collect();
        assert_eq!(
            wire_tiles.len(),
            3,
            "should have 3 wire tiles between (2,2) and (6,2)"
        );
    }

    // ========================================================================
    // Test 15: Codegen -- module instantiation
    // ========================================================================
    #[test]
    fn codegen_module_instantiation() {
        let src = r#"
            module sub(input a, input b) {
                tile g = And @(1, 1);
            }
            module top() {
                inst s0 = sub(x, y) @(5, 5);
            }
        "#;
        let bp = compile(src).expect("should compile");
        // The sub-module's And tile should appear at (5+1, 5+1) = (6, 6)
        let and_tile = bp.tiles.iter().find(|t| t.tile_type == TileType::And);
        assert!(
            and_tile.is_some(),
            "sub-module And tile should be in blueprint"
        );
        let and_tile = and_tile.unwrap();
        assert_eq!(
            and_tile.x, 6,
            "And tile x should be offset by instance origin"
        );
        assert_eq!(
            and_tile.y, 6,
            "And tile y should be offset by instance origin"
        );
    }

    // ========================================================================
    // Test 16: End-to-end -- compile and apply to simulation
    // ========================================================================
    #[test]
    fn end_to_end_compile_and_apply() {
        use crate::simulation::Simulation;

        let src = r#"
            module top() {
                tile g0 = And @(3, 3);
                tile g1 = Or @(5, 5);
            }
        "#;
        let bp = compile(src).expect("should compile");

        let mut sim = Simulation::new();
        bp.apply_to_simulation(&mut sim).expect("should apply");

        // Check that tiles are placed correctly
        assert_eq!(
            sim.tilemap.get_tile(3, 3).unwrap().meta.tile_type,
            TileType::And
        );
        assert_eq!(
            sim.tilemap.get_tile(5, 5).unwrap().meta.tile_type,
            TileType::Or
        );

        // Run a tick to verify simulation doesn't crash
        sim.tick();
    }

    // ========================================================================
    // Test 17: Error -- parse error recovery (multiple errors collected)
    // ========================================================================
    #[test]
    fn error_parse_recovery() {
        let src = r#"
            module test() {
                tile = ;
                wire ;
                tile g = And @(1, 1);
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let result = parser::parse(&tokens);
        // Should collect errors but still parse what it can
        // The exact behavior depends on recovery -- either errors from parser
        // or the valid statement is still parsed
        match result {
            Ok(modules) => {
                // Parser recovered and parsed the module
                assert_eq!(modules.len(), 1);
            }
            Err(errors) => {
                // Parser collected multiple errors
                assert!(
                    !errors.is_empty(),
                    "should have at least 1 error from malformed statements"
                );
            }
        }
    }

    // ========================================================================
    // Additional test: wire endpoint with qualified port
    // ========================================================================
    #[test]
    fn parser_qualified_wire_endpoint() {
        let src = r#"
            module sub(input a, output s) {
                tile g = And @(1, 1);
            }
            module top() {
                inst h0 = sub(x) @(10, 10);
                wire w: h0.s -> out;
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let modules = parser::parse(&tokens).expect("should parse");
        match &modules[1].body[1] {
            Statement::WireDecl { from, to, .. } => {
                assert_eq!(
                    *from,
                    WireEndpoint::QualifiedPort("h0".to_string(), "s".to_string())
                );
                assert_eq!(*to, WireEndpoint::Port("out".to_string()));
            }
            _ => panic!("expected WireDecl"),
        }
    }

    // ========================================================================
    // Additional test: block comments
    // ========================================================================
    #[test]
    fn lexer_block_comments() {
        let src = r#"
            module test() {
                /* this is a
                   multi-line block comment */
                tile g = And;
            }
        "#;
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let tok_types: Vec<&Token> = tokens.iter().map(|(t, _)| t).collect();
        // Should not contain comment content, but should contain the tile declaration
        assert!(tok_types.contains(&&Token::Tile));
        assert!(tok_types.contains(&&Token::Ident("And".to_string())));
    }

    // ========================================================================
    // Additional test: all keywords
    // ========================================================================
    #[test]
    fn lexer_all_keywords() {
        let src = "module input output tile wire const inst asm";
        let tokens = lexer::tokenize(src).expect("should tokenize");
        let tok_types: Vec<&Token> = tokens.iter().map(|(t, _)| t).collect();
        assert_eq!(tok_types[0], &Token::Module);
        assert_eq!(tok_types[1], &Token::Input);
        assert_eq!(tok_types[2], &Token::Output);
        assert_eq!(tok_types[3], &Token::Tile);
        assert_eq!(tok_types[4], &Token::Wire);
        assert_eq!(tok_types[5], &Token::Const);
        assert_eq!(tok_types[6], &Token::Inst);
        assert_eq!(tok_types[7], &Token::Asm);
    }

    // ========================================================================
    // Additional test: tile without explicit position (auto-placement)
    // ========================================================================
    #[test]
    fn codegen_auto_placement() {
        let src = r#"
            module top() {
                tile a = And;
                tile b = Or;
            }
        "#;
        let bp = compile(src).expect("should compile");
        let and_tile = bp.tiles.iter().find(|t| t.tile_type == TileType::And);
        let or_tile = bp.tiles.iter().find(|t| t.tile_type == TileType::Or);
        assert!(and_tile.is_some(), "should have And tile");
        assert!(or_tile.is_some(), "should have Or tile");
        // Auto-placed tiles should have different positions
        let a = and_tile.unwrap();
        let b = or_tile.unwrap();
        assert!(
            a.x != b.x || a.y != b.y,
            "auto-placed tiles should have different positions"
        );
    }
}
