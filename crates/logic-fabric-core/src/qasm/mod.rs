use crate::quantum::QGate;
use std::collections::HashMap;
use std::f32::consts::PI;

pub fn parse_qasm(source: &str) -> Result<(Vec<QGate>, usize), String> {
    let mut gates = Vec::new();
    let mut qubit_map: HashMap<String, usize> = HashMap::new();
    let mut next_qubit_id = 0;
    let mut total_qubits = 0;
    let mut in_gate_def = false;

    for (line_num, line) in source.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let clean_line = line.split("//").next().unwrap_or("").trim();
        if clean_line.is_empty() {
            continue;
        }
        if clean_line.starts_with("gate ") {
            in_gate_def = true;
            continue;
        }
        if in_gate_def {
            if clean_line.contains('}') {
                in_gate_def = false;
            }
            continue;
        }
        if clean_line.starts_with("if") {
            continue;
        }

        let parts: Vec<&str> = clean_line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let cmd = parts[0];

        match cmd {
            "OPENQASM" | "include" => {}
            "qreg" => {
                let decl = parts[1].trim_end_matches(';');
                if let Some(pos) = decl.find('[') {
                    let name = &decl[..pos];
                    let size_str = &decl[pos + 1..decl.len() - 1];
                    let size: usize = size_str
                        .parse()
                        .map_err(|_| format!("Invalid qreg size on line {}", line_num + 1))?;
                    for i in 0..size {
                        qubit_map.insert(format!("{}[{}]", name, i), next_qubit_id);
                        next_qubit_id += 1;
                    }
                    total_qubits += size;
                }
            }
            "barrier" | "measure" => {}
            _ => {
                let mut valid_cmd = cmd;
                let mut params = "";
                if let Some(pidx) = cmd.find('(') {
                    valid_cmd = &cmd[..pidx];
                    params = &cmd[pidx + 1..cmd.len() - 1];
                }
                let args_str = clean_line[parts[0].len()..].trim().trim_end_matches(';');
                let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();

                match valid_cmd {
                    "h" => {
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::H(q as u8));
                    }
                    "x" => {
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::X(q as u8));
                    }
                    "y" => {
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Y(q as u8));
                    }
                    "z" => {
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Z(q as u8));
                    }
                    "cx" => {
                        if args.len() != 2 {
                            return Err(format!("cx requires 2 args on line {}", line_num + 1));
                        }
                        let q1 = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        let q2 = lookup_qubit(args[1], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::CNot(q1 as u8, q2 as u8));
                    }
                    "cz" => {
                        if args.len() != 2 {
                            return Err(format!("cz requires 2 args on line {}", line_num + 1));
                        }
                        let q1 = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        let q2 = lookup_qubit(args[1], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::CZ(q1 as u8, q2 as u8));
                    }
                    "swap" => {
                        if args.len() != 2 {
                            return Err(format!("swap requires 2 args on line {}", line_num + 1));
                        }
                        let q1 = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        let q2 = lookup_qubit(args[1], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Swap(q1 as u8, q2 as u8));
                    }
                    "rx" => {
                        let theta = parse_float(params)?;
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Rx(q as u8, theta));
                    }
                    "ry" => {
                        let theta = parse_float(params)?;
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Ry(q as u8, theta));
                    }
                    "rz" => {
                        let theta = parse_float(params)?;
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Rz(q as u8, theta));
                    }
                    "p" | "phase" => {
                        let theta = parse_float(params)?;
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Phase(q as u8, theta));
                    }
                    "u3" | "u" => {
                        // Keep U3 as a single gate to preserve structure during optimization.
                        // U3(θ, φ, λ) matrix: [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
                        let param_parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();
                        if param_parts.len() != 3 {
                            return Err(format!("u3 requires 3 params on line {}", line_num + 1));
                        }
                        let theta = parse_float(param_parts[0])?;
                        let phi = parse_float(param_parts[1])?;
                        let lambda = parse_float(param_parts[2])?;
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id) as u8;
                        gates.push(QGate::U3(q, theta, phi, lambda));
                    }
                    "u2" => {
                        // U2(φ, λ) = U3(π/2, φ, λ)
                        let param_parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();
                        if param_parts.len() != 2 {
                            return Err(format!("u2 requires 2 params on line {}", line_num + 1));
                        }
                        let phi = parse_float(param_parts[0])?;
                        let lambda = parse_float(param_parts[1])?;
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id) as u8;
                        gates.push(QGate::U3(q, PI / 2.0, phi, lambda));
                    }
                    "u1" => {
                        let lambda = parse_float(params)?;
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id) as u8;
                        if lambda.abs() > 1e-10 {
                            gates.push(QGate::Rz(q, lambda));
                        }
                    }
                    "s" => {
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Phase(q as u8, PI / 2.0));
                    }
                    "sdg" => {
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Phase(q as u8, -PI / 2.0));
                    }
                    "t" => {
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Phase(q as u8, PI / 4.0));
                    }
                    "tdg" => {
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Phase(q as u8, -PI / 4.0));
                    }
                    "id" | "i" => {}
                    "sx" => {
                        // √X gate = U3(π/2, -π/2, π/2) up to global phase e^{iπ/4}.
                        // Global phase is irrelevant for gate-count optimization and measurement.
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id) as u8;
                        gates.push(QGate::U3(q, PI / 2.0, -PI / 2.0, PI / 2.0));
                    }
                    "sxdg" => {
                        // (√X)† = U3(π/2, π/2, -π/2) up to global phase.
                        let q = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id) as u8;
                        gates.push(QGate::U3(q, PI / 2.0, PI / 2.0, -PI / 2.0));
                    }
                    "ccx" => {
                        if args.len() != 3 {
                            return Err(format!("ccx requires 3 args on line {}", line_num + 1));
                        }
                        let c1 = lookup_qubit(args[0], &mut qubit_map, &mut next_qubit_id);
                        let c2 = lookup_qubit(args[1], &mut qubit_map, &mut next_qubit_id);
                        let t = lookup_qubit(args[2], &mut qubit_map, &mut next_qubit_id);
                        gates.push(QGate::Toffoli(c1 as u8, c2 as u8, t as u8));
                    }
                    "creg" => {}
                    _ => {
                        return Err(format!(
                            "Unknown gate '{}' on line {} - custom gates not supported",
                            valid_cmd,
                            line_num + 1
                        ));
                    }
                }
            }
        }
    }
    Ok((gates, total_qubits.max(next_qubit_id)))
}

fn lookup_qubit(name: &str, map: &mut HashMap<String, usize>, next_id: &mut usize) -> usize {
    if let Some(&id) = map.get(name) {
        return id;
    }
    let id = *next_id;
    map.insert(name.to_string(), id);
    *next_id += 1;
    id
}

fn parse_float(s: &str) -> Result<f32, String> {
    if s.is_empty() {
        return Ok(0.0);
    }
    let s = s.trim();
    let (negative, expr) = if let Some(stripped) = s.strip_prefix('-') {
        (true, stripped.trim())
    } else {
        (false, s)
    };
    let val_str = expr.replace("pi", &PI.to_string());
    let result = if val_str.contains('/') {
        let parts: Vec<&str> = val_str.splitn(2, '/').collect();
        if parts.len() == 2 {
            let numerator = parse_product(parts[0])?;
            let denominator = parse_product(parts[1])?;
            if denominator.abs() < 1e-10 {
                return Err(format!("Division by zero: {}", s));
            }
            numerator / denominator
        } else {
            parse_product(&val_str)?
        }
    } else {
        parse_product(&val_str)?
    };
    Ok(if negative { -result } else { result })
}

fn parse_product(s: &str) -> Result<f32, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(1.0);
    }
    if s.contains('*') {
        let parts: Vec<&str> = s.split('*').collect();
        let mut product = 1.0f32;
        for p in parts {
            let p = p.trim();
            if p.is_empty() {
                continue;
            }
            product *= p
                .parse::<f32>()
                .map_err(|_| format!("Invalid number: {}", p))?;
        }
        Ok(product)
    } else {
        s.parse::<f32>()
            .map_err(|_| format!("Invalid number: {}", s))
    }
}

pub fn to_qasm(gates: &[QGate], num_qubits: usize) -> String {
    let mut out = String::new();
    out.push_str("OPENQASM 2.0;\n");
    out.push_str("include \"qelib1.inc\";\n");
    out.push_str(&format!("qreg q[{}];\n\n", num_qubits));
    for gate in gates {
        match gate {
            QGate::H(q) => out.push_str(&format!("h q[{}];\n", q)),
            QGate::X(q) => out.push_str(&format!("x q[{}];\n", q)),
            QGate::Y(q) => out.push_str(&format!("y q[{}];\n", q)),
            QGate::Z(q) => out.push_str(&format!("z q[{}];\n", q)),
            QGate::CNot(c, t) => out.push_str(&format!("cx q[{}], q[{}];\n", c, t)),
            QGate::CZ(c, t) => out.push_str(&format!("cz q[{}], q[{}];\n", c, t)),
            QGate::Swap(a, b) => out.push_str(&format!("swap q[{}], q[{}];\n", a, b)),
            QGate::Rx(q, t) => out.push_str(&format!("rx({}) q[{}];\n", t, q)),
            QGate::Ry(q, t) => out.push_str(&format!("ry({}) q[{}];\n", t, q)),
            QGate::Rz(q, t) => out.push_str(&format!("rz({}) q[{}];\n", t, q)),
            QGate::Phase(q, t) => out.push_str(&format!("p({}) q[{}];\n", t, q)),
            QGate::Toffoli(c1, c2, t) => {
                out.push_str(&format!("ccx q[{}], q[{}], q[{}];\n", c1, c2, t))
            }
            QGate::U3(q, theta, phi, lambda) => {
                out.push_str(&format!("u3({},{},{}) q[{}];\n", theta, phi, lambda, q))
            }
            _ => out.push_str(&format!("// Unknown gate {:?}\n", gate)),
        }
    }
    out
}
