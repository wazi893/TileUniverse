use crate::net::NetReport;
use crate::net::net_summary::{GlobalSummary, RegionSummary};
use crate::simulation::StructuralIssue;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineDiagnostics {
    pub structural_issues: Vec<StructuralIssue>,
    pub net_report: NetReport,
    pub global_summary: GlobalSummary,
    pub region_summaries: Vec<RegionSummary>,
}

impl EngineDiagnostics {
    pub(crate) fn new(
        structural_issues: Vec<StructuralIssue>,
        net_report: NetReport,
        global_summary: GlobalSummary,
        region_summaries: Vec<RegionSummary>,
    ) -> Self {
        Self {
            structural_issues,
            net_report,
            global_summary,
            region_summaries,
        }
    }
}
