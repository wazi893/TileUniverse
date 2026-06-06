use crate::diagnostics::EngineDiagnostics;
use crate::net::NetId;
use crate::simulation::StructuralIssueKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LintSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum LintKind {
    StructuralIssue(StructuralIssueKind),
    HighFanout {
        net_id: NetId,
        fanout: u32,
        limit: u32,
    },
    FloatingNet {
        net_id: NetId,
    },
    MixedRegionNet {
        net_id: NetId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintIssue {
    pub kind: LintKind,
    pub severity: LintSeverity,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintProfile {
    pub max_fanout_warning: u32,
    pub max_fanout_error: u32,
    pub allow_floating_nets: bool,
    pub allow_mixed_region_nets: bool,
    pub treat_structural_issues_as_errors: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintSummary {
    pub issue_count: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub is_ok: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintResult {
    pub summary: LintSummary,
    pub issues: Vec<LintIssue>,
}

impl LintProfile {
    pub fn default_relaxed() -> Self {
        Self {
            max_fanout_warning: 16,
            max_fanout_error: 64,
            allow_floating_nets: true,
            allow_mixed_region_nets: true,
            treat_structural_issues_as_errors: false,
        }
    }

    pub fn strict() -> Self {
        Self {
            max_fanout_warning: 16,
            max_fanout_error: 32,
            allow_floating_nets: false,
            allow_mixed_region_nets: false,
            treat_structural_issues_as_errors: true,
        }
    }
}

pub fn run_lint(diag: &EngineDiagnostics, profile: &LintProfile) -> LintResult {
    let mut issues: Vec<LintIssue> = Vec::new();

    // Structural issues
    for iss in &diag.structural_issues {
        let severity = if profile.treat_structural_issues_as_errors {
            LintSeverity::Error
        } else {
            LintSeverity::Warning
        };
        let message = match iss.kind {
            StructuralIssueKind::FanoutExceeded { fanout, max } => {
                format!(
                    "Structural fanout exceeded at ({},{}) fanout={} max={}",
                    iss.x, iss.y, fanout, max
                )
            }
            StructuralIssueKind::UnclockedRegister => {
                format!("Structural unclocked register at ({},{})", iss.x, iss.y)
            }
            StructuralIssueKind::OrphanLogic => {
                format!("Structural orphan logic at ({},{})", iss.x, iss.y)
            }
        };
        issues.push(LintIssue {
            kind: LintKind::StructuralIssue(iss.kind.clone()),
            severity,
            message,
        });
    }

    // Net-based checks
    for net in &diag.net_report.nets {
        // Fanout
        if net.fanout_max > profile.max_fanout_warning {
            let severity = if net.fanout_max > profile.max_fanout_error {
                LintSeverity::Error
            } else {
                LintSeverity::Warning
            };
            issues.push(LintIssue {
                kind: LintKind::HighFanout {
                    net_id: net.id,
                    fanout: net.fanout_max,
                    limit: if severity == LintSeverity::Error {
                        profile.max_fanout_error
                    } else {
                        profile.max_fanout_warning
                    },
                },
                severity,
                message: format!(
                    "High fanout on net {} (fanout={}, limit={})",
                    net.id,
                    net.fanout_max,
                    if severity == LintSeverity::Error {
                        profile.max_fanout_error
                    } else {
                        profile.max_fanout_warning
                    }
                ),
            });
        }

        // Floating
        if !profile.allow_floating_nets && net.is_floating {
            issues.push(LintIssue {
                kind: LintKind::FloatingNet { net_id: net.id },
                severity: LintSeverity::Warning,
                message: format!("Floating net {}", net.id),
            });
        }

        // Mixed region
        if !profile.allow_mixed_region_nets && net.is_mixed_region {
            issues.push(LintIssue {
                kind: LintKind::MixedRegionNet { net_id: net.id },
                severity: LintSeverity::Warning,
                message: format!("Mixed-region net {}", net.id),
            });
        }
    }

    // Deterministic ordering: severity desc, then kind, then net_id if present
    issues.sort_by(|a, b| {
        b.severity
            .cmp(&a.severity)
            .then(a.kind.cmp(&b.kind))
            .then_with(|| {
                let a_id = match a.kind {
                    LintKind::HighFanout { net_id, .. }
                    | LintKind::FloatingNet { net_id }
                    | LintKind::MixedRegionNet { net_id } => Some(net_id),
                    _ => None,
                };
                let b_id = match b.kind {
                    LintKind::HighFanout { net_id, .. }
                    | LintKind::FloatingNet { net_id }
                    | LintKind::MixedRegionNet { net_id } => Some(net_id),
                    _ => None,
                };
                a_id.cmp(&b_id)
            })
    });

    let error_count = issues
        .iter()
        .filter(|i| i.severity == LintSeverity::Error)
        .count();
    let warning_count = issues
        .iter()
        .filter(|i| i.severity == LintSeverity::Warning)
        .count();
    let summary = LintSummary {
        issue_count: issues.len(),
        error_count,
        warning_count,
        is_ok: error_count == 0,
    };

    LintResult { summary, issues }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::EngineDiagnostics;
    use crate::net::net_summary::GlobalSummary;
    use crate::net::{NetKind, NetReport, NetStats};
    use crate::simulation::{StructuralIssue, StructuralIssueKind};

    fn make_diag(structural: Vec<StructuralIssue>, nets: Vec<NetStats>) -> EngineDiagnostics {
        let report = NetReport { nets: nets.clone() };
        let global_summary = GlobalSummary {
            total_nets: nets.len() as u32,
            clock_nets: nets
                .iter()
                .filter(|n| n.kind == NetKind::Clock || n.has_clock)
                .count() as u32,
            floating_nets: nets.iter().filter(|n| n.is_floating).count() as u32,
            mixed_region_nets: nets.iter().filter(|n| n.is_mixed_region).count() as u32,
            max_fanout: nets.iter().map(|n| n.fanout_max).max().unwrap_or(0),
            fanout_histogram: vec![0; 17],
        };
        EngineDiagnostics {
            structural_issues: structural,
            net_report: report,
            global_summary,
            region_summaries: Vec::new(),
        }
    }

    fn net(id: NetId, fanout: u32, is_floating: bool, is_mixed: bool) -> NetStats {
        NetStats {
            id,
            kind: NetKind::Data,
            node_count: 1,
            driver_count: if is_floating { 0 } else { 1 },
            sink_count: 1,
            fanout_max: fanout,
            region: Some(0),
            x0: 0,
            y0: 0,
            x1: 0,
            y1: 0,
            has_clock: false,
            is_floating,
            is_mixed_region: is_mixed,
        }
    }

    #[test]
    fn lint_empty_relaxed_ok() {
        let diag = make_diag(Vec::new(), Vec::new());
        let res = run_lint(&diag, &LintProfile::default_relaxed());
        assert!(res.summary.is_ok);
        assert!(res.issues.is_empty());
    }

    #[test]
    fn lint_structural_errors_flagged() {
        let structural = vec![StructuralIssue {
            x: 1,
            y: 2,
            kind: StructuralIssueKind::UnclockedRegister,
        }];
        let diag = make_diag(structural, Vec::new());
        let res = run_lint(&diag, &LintProfile::strict());
        assert!(!res.summary.is_ok);
        assert!(res.summary.error_count >= 1);
        assert!(
            res.issues
                .iter()
                .any(|i| matches!(i.kind, LintKind::StructuralIssue(_)))
        );
    }

    #[test]
    fn lint_fanout_warning_vs_error() {
        let nets = vec![net(1, 20, false, false), net(2, 70, false, false)];
        let diag = make_diag(Vec::new(), nets);
        let res = run_lint(&diag, &LintProfile::default_relaxed());
        assert!(
            res.issues
                .iter()
                .any(|i| matches!(i.kind, LintKind::HighFanout { net_id: 1, .. })
                    && i.severity == LintSeverity::Warning)
        );
        assert!(
            res.issues
                .iter()
                .any(|i| matches!(i.kind, LintKind::HighFanout { net_id: 2, .. })
                    && i.severity == LintSeverity::Error)
        );
    }

    #[test]
    fn lint_floating_and_mixed_flagged() {
        let nets = vec![net(1, 0, true, false), net(2, 0, false, true)];
        let diag = make_diag(Vec::new(), nets);
        let res = run_lint(&diag, &LintProfile::strict());
        assert!(
            res.issues
                .iter()
                .any(|i| matches!(i.kind, LintKind::FloatingNet { net_id: 1 }))
        );
        assert!(
            res.issues
                .iter()
                .any(|i| matches!(i.kind, LintKind::MixedRegionNet { net_id: 2 }))
        );
    }

    #[test]
    fn lint_deterministic() {
        let nets = vec![net(2, 70, false, false), net(1, 20, false, false)];
        let diag = make_diag(Vec::new(), nets);
        let res1 = run_lint(&diag, &LintProfile::default_relaxed());
        let res2 = run_lint(&diag, &LintProfile::default_relaxed());
        assert_eq!(res1, res2);
    }
}
