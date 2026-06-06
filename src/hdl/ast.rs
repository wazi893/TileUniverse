/// Port direction: input or output.
#[derive(Debug, Clone, PartialEq)]
pub enum PortDir {
    Input,
    Output,
}

/// A port declaration within a module signature.
#[derive(Debug, Clone, PartialEq)]
pub struct PortDecl {
    pub name: String,
    pub direction: PortDir,
}

/// An explicit grid position `@(x, y)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    pub x: u32,
    pub y: u32,
}

/// A wire endpoint: either a port name, a tile instance, or a qualified port (inst.port).
#[derive(Debug, Clone, PartialEq)]
pub enum WireEndpoint {
    /// Module port name (e.g. `a`, `sum`)
    Port(String),
    /// Tile instance name (e.g. `xor0`)
    Tile(String),
    /// Qualified port: instance_name.port_name (e.g. `ha0.sum`)
    QualifiedPort(String, String),
}

/// A statement within a module body.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    TileDecl {
        name: String,
        tile_type: String,
        position: Option<Position>,
        initial_value: Option<u64>,
    },
    WireDecl {
        name: String,
        from: WireEndpoint,
        to: WireEndpoint,
    },
    ConstDecl {
        name: String,
        value: u64,
    },
    InstDecl {
        name: String,
        module_name: String,
        args: Vec<String>,
        position: Option<Position>,
    },
    AsmDecl {
        name: String,
        source: String,
    },
}

/// A module declaration: `module name(ports) { body }`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDecl {
    pub name: String,
    pub ports: Vec<PortDecl>,
    pub body: Vec<Statement>,
}
