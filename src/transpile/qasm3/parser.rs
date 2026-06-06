//! OpenQASM 3.0 Parser
//!
//! Parses tokenized QASM3 into QGate circuits.

use super::ast::{QasmGate, QasmProgram, QasmStatement};
use super::lexer::{Lexer, Token, TokenKind};
use logic_fabric_core::quantum::QGate;
use std::f64::consts::PI;

/// Parse error with location
#[derive(Clone, Debug)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl ParseError {
    pub fn new(message: impl Into<String>, line: usize, column: usize) -> Self {
        Self {
            message: message.into(),
            line,
            column,
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Parse error at line {}, column {}: {}",
            self.line, self.column, self.message
        )
    }
}

impl std::error::Error for ParseError {}

/// Parse result type
pub type ParseResult<T> = Result<T, ParseError>;

/// QASM3 Parser
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    /// Create a new parser from source text
    pub fn new(source: &str) -> Self {
        let tokens = Lexer::new(source).tokenize();
        Self { tokens, pos: 0 }
    }

    /// Get current token
    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token {
            kind: TokenKind::Eof,
            line: 0,
            column: 0,
        })
    }

    /// Advance to next token
    fn advance(&mut self) -> &Token {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        self.current()
    }

    /// Check if current token matches kind
    fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.current().kind) == std::mem::discriminant(kind)
    }

    /// Expect a specific token, error if not found
    fn expect(&mut self, kind: TokenKind) -> ParseResult<Token> {
        if self.check(&kind) {
            let tok = self.current().clone();
            self.advance();
            Ok(tok)
        } else {
            let curr = self.current();
            Err(ParseError::new(
                format!("Expected {:?}, found {:?}", kind, curr.kind),
                curr.line,
                curr.column,
            ))
        }
    }

    /// Parse QASM3 program
    pub fn parse(&mut self) -> ParseResult<QasmProgram> {
        let mut program = QasmProgram::default();

        // Optional: OPENQASM version
        if self.check(&TokenKind::OpenQASM) {
            self.advance();
            // Version number (e.g., 3.0)
            if let TokenKind::Integer(v) = &self.current().kind {
                program.version = format!("{}.0", v);
                self.advance();
            } else if let TokenKind::Float(v) = &self.current().kind {
                program.version = format!("{}", v);
                self.advance();
            }
            self.expect(TokenKind::Semicolon)?;
        }

        // Parse includes and declarations
        while !self.check(&TokenKind::Eof) {
            match &self.current().kind {
                TokenKind::Include => {
                    self.advance();
                    if let TokenKind::String(s) = &self.current().kind {
                        program.includes.push(s.clone());
                        self.advance();
                    }
                    self.expect(TokenKind::Semicolon)?;
                }
                TokenKind::Qubit => {
                    self.advance();
                    // qubit[N] name;
                    if self.check(&TokenKind::LeftBracket) {
                        self.advance();
                        if let TokenKind::Integer(n) = &self.current().kind {
                            program.num_qubits = *n as usize;
                            self.advance();
                        }
                        self.expect(TokenKind::RightBracket)?;
                    }
                    if let TokenKind::Identifier(name) = &self.current().kind {
                        program.qubit_register = name.clone();
                        self.advance();
                    }
                    self.expect(TokenKind::Semicolon)?;
                }
                TokenKind::Bit => {
                    self.advance();
                    // bit[N] name;
                    if self.check(&TokenKind::LeftBracket) {
                        self.advance();
                        if let TokenKind::Integer(n) = &self.current().kind {
                            program.num_clbits = *n as usize;
                            self.advance();
                        }
                        self.expect(TokenKind::RightBracket)?;
                    }
                    if let TokenKind::Identifier(name) = &self.current().kind {
                        program.clbit_register = name.clone();
                        self.advance();
                    }
                    self.expect(TokenKind::Semicolon)?;
                }
                TokenKind::Measure => {
                    let stmt = self.parse_measure()?;
                    program.statements.push(stmt);
                }
                TokenKind::Reset => {
                    self.advance();
                    let qubit = self.parse_qubit_ref()?;
                    self.expect(TokenKind::Semicolon)?;
                    program.statements.push(QasmStatement::Reset(qubit));
                }
                TokenKind::Barrier => {
                    self.advance();
                    let mut qubits = vec![self.parse_qubit_ref()?];
                    while self.check(&TokenKind::Comma) {
                        self.advance();
                        qubits.push(self.parse_qubit_ref()?);
                    }
                    self.expect(TokenKind::Semicolon)?;
                    program.statements.push(QasmStatement::Barrier(qubits));
                }
                TokenKind::Identifier(_) => {
                    // Gate application
                    let gate = self.parse_gate()?;
                    program.statements.push(QasmStatement::Gate(gate));
                }
                _ => {
                    // Skip unknown tokens
                    self.advance();
                }
            }
        }

        Ok(program)
    }

    /// Parse a qubit reference like q[0]
    fn parse_qubit_ref(&mut self) -> ParseResult<usize> {
        // Skip register name
        if let TokenKind::Identifier(_) = &self.current().kind {
            self.advance();
        }
        self.expect(TokenKind::LeftBracket)?;
        let idx = if let TokenKind::Integer(n) = &self.current().kind {
            let v = *n as usize;
            self.advance();
            v
        } else {
            return Err(ParseError::new(
                "Expected qubit index",
                self.current().line,
                self.current().column,
            ));
        };
        self.expect(TokenKind::RightBracket)?;
        Ok(idx)
    }

    /// Parse a classical bit reference like c[0]
    fn parse_clbit_ref(&mut self) -> ParseResult<usize> {
        // Skip register name
        if let TokenKind::Identifier(_) = &self.current().kind {
            self.advance();
        }
        self.expect(TokenKind::LeftBracket)?;
        let idx = if let TokenKind::Integer(n) = &self.current().kind {
            let v = *n as usize;
            self.advance();
            v
        } else {
            return Err(ParseError::new(
                "Expected classical bit index",
                self.current().line,
                self.current().column,
            ));
        };
        self.expect(TokenKind::RightBracket)?;
        Ok(idx)
    }

    /// Parse measure statement
    fn parse_measure(&mut self) -> ParseResult<QasmStatement> {
        self.expect(TokenKind::Measure)?;
        let qubit = self.parse_qubit_ref()?;

        // Optional arrow -> c[n]
        let clbit = if self.check(&TokenKind::Arrow) {
            self.advance();
            self.parse_clbit_ref()?
        } else {
            qubit // Default to same index
        };

        self.expect(TokenKind::Semicolon)?;
        Ok(QasmStatement::Measure { qubit, clbit })
    }

    /// Parse a gate expression value (number or pi expression)
    fn parse_expr(&mut self) -> ParseResult<f64> {
        let mut value = self.parse_term()?;

        // Handle + and -
        while self.check(&TokenKind::Plus) || self.check(&TokenKind::Minus) {
            let op = self.current().kind.clone();
            self.advance();
            let right = self.parse_term()?;
            match op {
                TokenKind::Plus => value += right,
                TokenKind::Minus => value -= right,
                _ => {}
            }
        }

        Ok(value)
    }

    /// Parse a term (handles * and /)
    fn parse_term(&mut self) -> ParseResult<f64> {
        let mut value = self.parse_factor()?;

        while self.check(&TokenKind::Star) || self.check(&TokenKind::Slash) {
            let op = self.current().kind.clone();
            self.advance();
            let right = self.parse_factor()?;
            match op {
                TokenKind::Star => value *= right,
                TokenKind::Slash => value /= right,
                _ => {}
            }
        }

        Ok(value)
    }

    /// Parse a factor (number, pi, or parenthesized expression)
    fn parse_factor(&mut self) -> ParseResult<f64> {
        // Handle unary minus
        if self.check(&TokenKind::Minus) {
            self.advance();
            return Ok(-self.parse_factor()?);
        }

        match &self.current().kind {
            TokenKind::Integer(n) => {
                let v = *n as f64;
                self.advance();
                Ok(v)
            }
            TokenKind::Float(f) => {
                let v = *f;
                self.advance();
                Ok(v)
            }
            TokenKind::Pi => {
                self.advance();
                Ok(PI)
            }
            TokenKind::Tau => {
                self.advance();
                Ok(2.0 * PI)
            }
            TokenKind::Euler => {
                self.advance();
                Ok(std::f64::consts::E)
            }
            TokenKind::LeftParen => {
                self.advance();
                let v = self.parse_expr()?;
                self.expect(TokenKind::RightParen)?;
                Ok(v)
            }
            _ => Err(ParseError::new(
                format!("Expected number or pi, found {:?}", self.current().kind),
                self.current().line,
                self.current().column,
            )),
        }
    }

    /// Parse gate parameters like (theta) or (theta, phi, lambda)
    fn parse_params(&mut self) -> ParseResult<Vec<f64>> {
        let mut params = Vec::new();
        if self.check(&TokenKind::LeftParen) {
            self.advance();
            if !self.check(&TokenKind::RightParen) {
                params.push(self.parse_expr()?);
                while self.check(&TokenKind::Comma) {
                    self.advance();
                    params.push(self.parse_expr()?);
                }
            }
            self.expect(TokenKind::RightParen)?;
        }
        Ok(params)
    }

    /// Parse gate application
    fn parse_gate(&mut self) -> ParseResult<QasmGate> {
        let name = if let TokenKind::Identifier(s) = &self.current().kind {
            s.clone()
        } else {
            return Err(ParseError::new(
                "Expected gate name",
                self.current().line,
                self.current().column,
            ));
        };
        self.advance();

        // Parse parameters
        let params = self.parse_params()?;

        // Parse qubit operands
        let q0 = self.parse_qubit_ref()?;

        let mut operands = vec![q0];
        while self.check(&TokenKind::Comma) {
            self.advance();
            operands.push(self.parse_qubit_ref()?);
        }

        self.expect(TokenKind::Semicolon)?;

        // Create gate based on name and operands
        let gate = match name.to_lowercase().as_str() {
            // Single-qubit gates (no params)
            "h" => QasmGate::H(operands[0]),
            "x" => QasmGate::X(operands[0]),
            "y" => QasmGate::Y(operands[0]),
            "z" => QasmGate::Z(operands[0]),
            "t" => QasmGate::T(operands[0]),
            "tdg" => QasmGate::Tdg(operands[0]),
            "s" => QasmGate::S(operands[0]),
            "sdg" => QasmGate::Sdg(operands[0]),
            "sx" => QasmGate::SX(operands[0]),
            "sxdg" => QasmGate::SXdg(operands[0]),
            "id" | "i" => QasmGate::Id(operands[0]),

            // Single-qubit rotation gates
            "rx" => QasmGate::Rx(operands[0], params.get(0).copied().unwrap_or(0.0)),
            "ry" => QasmGate::Ry(operands[0], params.get(0).copied().unwrap_or(0.0)),
            "rz" => QasmGate::Rz(operands[0], params.get(0).copied().unwrap_or(0.0)),
            "p" | "phase" => QasmGate::P(operands[0], params.get(0).copied().unwrap_or(0.0)),
            "u1" => QasmGate::U1(operands[0], params.get(0).copied().unwrap_or(0.0)),
            "u2" => QasmGate::U2(
                operands[0],
                params.get(0).copied().unwrap_or(0.0),
                params.get(1).copied().unwrap_or(0.0),
            ),
            "u3" | "u" => QasmGate::U3(
                operands[0],
                params.get(0).copied().unwrap_or(0.0),
                params.get(1).copied().unwrap_or(0.0),
                params.get(2).copied().unwrap_or(0.0),
            ),

            // Two-qubit gates
            "cx" | "cnot" => QasmGate::CX(operands[0], operands.get(1).copied().unwrap_or(1)),
            "cz" => QasmGate::CZ(operands[0], operands.get(1).copied().unwrap_or(1)),
            "swap" => QasmGate::Swap(operands[0], operands.get(1).copied().unwrap_or(1)),
            "iswap" => QasmGate::ISwap(operands[0], operands.get(1).copied().unwrap_or(1)),
            "crx" => QasmGate::CRx(
                operands[0],
                operands.get(1).copied().unwrap_or(1),
                params.get(0).copied().unwrap_or(0.0),
            ),
            "cry" => QasmGate::CRy(
                operands[0],
                operands.get(1).copied().unwrap_or(1),
                params.get(0).copied().unwrap_or(0.0),
            ),
            "crz" => QasmGate::CRz(
                operands[0],
                operands.get(1).copied().unwrap_or(1),
                params.get(0).copied().unwrap_or(0.0),
            ),
            "cp" | "cphase" => QasmGate::CP(
                operands[0],
                operands.get(1).copied().unwrap_or(1),
                params.get(0).copied().unwrap_or(0.0),
            ),
            "rzz" => QasmGate::RZZ(
                operands[0],
                operands.get(1).copied().unwrap_or(1),
                params.get(0).copied().unwrap_or(0.0),
            ),
            "rxx" => QasmGate::RXX(
                operands[0],
                operands.get(1).copied().unwrap_or(1),
                params.get(0).copied().unwrap_or(0.0),
            ),
            "ryy" => QasmGate::RYY(
                operands[0],
                operands.get(1).copied().unwrap_or(1),
                params.get(0).copied().unwrap_or(0.0),
            ),

            // Three-qubit gates
            "ccx" | "toffoli" => QasmGate::CCX(
                operands[0],
                operands.get(1).copied().unwrap_or(1),
                operands.get(2).copied().unwrap_or(2),
            ),
            "ccz" => QasmGate::CCZ(
                operands[0],
                operands.get(1).copied().unwrap_or(1),
                operands.get(2).copied().unwrap_or(2),
            ),
            "cswap" | "fredkin" => QasmGate::CSwap(
                operands[0],
                operands.get(1).copied().unwrap_or(1),
                operands.get(2).copied().unwrap_or(2),
            ),

            _ => {
                return Err(ParseError::new(
                    format!("Unknown gate: {}", name),
                    self.current().line,
                    self.current().column,
                ));
            }
        };

        Ok(gate)
    }
}

/// Parse QASM3 source to QGate circuit
///
/// This is the main entry point for parsing QASM3 strings.
///
/// # Example
///
/// ```ignore
/// use engine::transpile::qasm3::parse_qasm3;
///
/// let qasm = r#"
///     OPENQASM 3.0;
///     qubit[2] q;
///     h q[0];
///     cx q[0], q[1];
/// "#;
///
/// let circuit = parse_qasm3(qasm).unwrap();
/// assert_eq!(circuit.len(), 2);
/// ```
pub fn parse_qasm3(source: &str) -> ParseResult<Vec<QGate>> {
    let mut parser = Parser::new(source);
    let program = parser.parse()?;
    Ok(qasm_to_qgate(&program))
}

/// Parse QASM3 with strict validation
pub fn parse_qasm3_strict(source: &str) -> ParseResult<Vec<QGate>> {
    let mut parser = Parser::new(source);
    let program = parser.parse()?;

    // Validate version
    if !program.version.starts_with("3") {
        return Err(ParseError::new(
            format!("Expected QASM 3.x, found version {}", program.version),
            1,
            1,
        ));
    }

    Ok(qasm_to_qgate(&program))
}

/// Convert QasmProgram to Vec<QGate>
pub fn qasm_to_qgate(program: &QasmProgram) -> Vec<QGate> {
    let mut gates = Vec::new();

    for stmt in &program.statements {
        match stmt {
            QasmStatement::Gate(g) => {
                if let Some(qgate) = qasm_gate_to_qgate(g) {
                    gates.push(qgate);
                }
            }
            QasmStatement::Measure { qubit, .. } => {
                gates.push(QGate::Measure(*qubit as u8));
            }
            QasmStatement::Reset(_) | QasmStatement::Barrier(_) | QasmStatement::Comment(_) => {
                // These don't map to QGate directly
            }
        }
    }

    gates
}

/// Convert a single QasmGate to QGate
fn qasm_gate_to_qgate(gate: &QasmGate) -> Option<QGate> {
    let result = match gate {
        // Single-qubit gates
        QasmGate::H(q) => QGate::H(*q as u8),
        QasmGate::X(q) => QGate::X(*q as u8),
        QasmGate::Y(q) => QGate::Y(*q as u8),
        QasmGate::Z(q) => QGate::Z(*q as u8),
        QasmGate::T(q) => QGate::T(*q as u8),
        QasmGate::Tdg(q) => QGate::Tdg(*q as u8),
        QasmGate::S(q) => QGate::Phase(*q as u8, std::f32::consts::FRAC_PI_2),
        QasmGate::Sdg(q) => QGate::Phase(*q as u8, -std::f32::consts::FRAC_PI_2),
        QasmGate::SX(q) => QGate::Rx(*q as u8, std::f32::consts::FRAC_PI_2),
        QasmGate::SXdg(q) => QGate::Rx(*q as u8, -std::f32::consts::FRAC_PI_2),
        QasmGate::Id(_) => return None, // Identity is a no-op

        // Rotation gates
        QasmGate::Rx(q, theta) => QGate::Rx(*q as u8, *theta as f32),
        QasmGate::Ry(q, theta) => QGate::Ry(*q as u8, *theta as f32),
        QasmGate::Rz(q, theta) => QGate::Rz(*q as u8, *theta as f32),
        QasmGate::P(q, theta) => QGate::Phase(*q as u8, *theta as f32),
        QasmGate::U1(q, lambda) => QGate::Phase(*q as u8, *lambda as f32),
        QasmGate::U2(q, phi, lambda) => {
            // U2(φ,λ) = U3(π/2, φ, λ)
            QGate::U3(
                *q as u8,
                std::f32::consts::FRAC_PI_2,
                *phi as f32,
                *lambda as f32,
            )
        }
        QasmGate::U3(q, theta, phi, lambda) => {
            QGate::U3(*q as u8, *theta as f32, *phi as f32, *lambda as f32)
        }
        QasmGate::U(q, theta, phi, lambda) => {
            QGate::U3(*q as u8, *theta as f32, *phi as f32, *lambda as f32)
        }

        // Two-qubit gates
        QasmGate::CX(c, t) => QGate::CNot(*c as u8, *t as u8),
        QasmGate::CZ(a, b) => QGate::CZ(*a as u8, *b as u8),
        QasmGate::Swap(a, b) => QGate::Swap(*a as u8, *b as u8),
        QasmGate::ISwap(_, _) => return None, // No direct QGate for iSWAP
        QasmGate::CRx(_, _, _) => return None, // No direct QGate for CRx
        QasmGate::CRy(_, _, _) => return None, // No direct QGate for CRy
        QasmGate::CRz(c, t, theta) => QGate::CRz(*c as u8, *t as u8, *theta as f32),
        QasmGate::CP(c, t, theta) => QGate::CPhase(*c as u8, *t as u8, *theta as f32),
        QasmGate::RZZ(_, _, _) => return None, // No direct QGate for RZZ
        QasmGate::RXX(_, _, _) => return None, // No direct QGate for RXX
        QasmGate::RYY(_, _, _) => return None, // No direct QGate for RYY

        // Three-qubit gates
        QasmGate::CCX(a, b, c) => QGate::Toffoli(*a as u8, *b as u8, *c as u8),
        QasmGate::CCZ(a, b, c) => QGate::CCZ(*a as u8, *b as u8, *c as u8),
        QasmGate::CSwap(_, _, _) => return None, // No direct QGate for CSWAP
    };

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_circuit() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            h q[0];
            cx q[0], q[1];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.len(), 2);
        assert!(matches!(circuit[0], QGate::H(0)));
        assert!(matches!(circuit[1], QGate::CNot(0, 1)));
    }

    #[test]
    fn test_parse_rotations() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            rx(1.5708) q[0];
            ry(pi/2) q[0];
            rz(pi) q[0];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.len(), 3);
        assert!(matches!(circuit[0], QGate::Rx(0, _)));
        assert!(matches!(circuit[1], QGate::Ry(0, _)));
        assert!(matches!(circuit[2], QGate::Rz(0, _)));
    }

    #[test]
    fn test_parse_t_gates() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            t q[0];
            tdg q[1];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.len(), 2);
        assert!(matches!(circuit[0], QGate::T(0)));
        assert!(matches!(circuit[1], QGate::Tdg(1)));
    }

    #[test]
    fn test_parse_three_qubit() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            ccx q[0], q[1], q[2];
            ccz q[0], q[1], q[2];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.len(), 2);
        assert!(matches!(circuit[0], QGate::Toffoli(0, 1, 2)));
        assert!(matches!(circuit[1], QGate::CCZ(0, 1, 2)));
    }

    #[test]
    fn test_parse_measure() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            bit[1] c;
            h q[0];
            measure q[0] -> c[0];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.len(), 2);
        assert!(matches!(circuit[0], QGate::H(0)));
        assert!(matches!(circuit[1], QGate::Measure(0)));
    }

    #[test]
    fn test_parse_pi_expressions() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            rz(pi/4) q[0];
            rz(-pi/2) q[0];
            rz(2*pi) q[0];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.len(), 3);

        // Check angles
        if let QGate::Rz(_, angle) = circuit[0] {
            assert!((angle - (PI / 4.0) as f32).abs() < 1e-5);
        }
        if let QGate::Rz(_, angle) = circuit[1] {
            assert!((angle - (-PI / 2.0) as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_parse_u3() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            u3(pi, 0, pi) q[0];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.len(), 1);
        assert!(matches!(circuit[0], QGate::U3(0, _, _, _)));
    }

    #[test]
    fn test_parse_s_gates() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            s q[0];
            sdg q[0];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.len(), 2);

        // S maps to Phase(π/2)
        if let QGate::Phase(q, angle) = circuit[0] {
            assert_eq!(q, 0);
            assert!((angle - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        }
    }

    #[test]
    fn test_parse_without_header() {
        // Should work without OPENQASM header
        let qasm = r#"
            qubit[2] q;
            h q[0];
            cx q[0], q[1];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.len(), 2);
    }

    #[test]
    fn test_strict_rejects_qasm2() {
        let qasm = r#"
            OPENQASM 2.0;
            qubit[1] q;
            h q[0];
        "#;

        let result = parse_qasm3_strict(qasm);
        assert!(result.is_err());
    }
}
