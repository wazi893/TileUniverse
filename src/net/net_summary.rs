use crate::field::FieldGrid;
use crate::net::{NetReport, NetStats};

pub const FANOUT_BUCKET_MAX: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalSummary {
    pub total_nets: u32,
    pub clock_nets: u32,
    pub floating_nets: u32,
    pub mixed_region_nets: u32,
    pub max_fanout: u32,
    pub fanout_histogram: Vec<u32>, // len = FANOUT_BUCKET_MAX + 1 (last bucket is overflow)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionSummary {
    pub region_id: u32,
    pub nets: u32,
    pub drivers: u32,
    pub sinks: u32,
    pub clock_nets: u32,
    pub floating_nets: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryReport {
    pub global: GlobalSummary,
    pub regions: Vec<RegionSummary>,
}

pub fn summarize_nets(report: &NetReport) -> GlobalSummary {
    let mut total_nets = 0u32;
    let mut clock_nets = 0u32;
    let mut floating_nets = 0u32;
    let mut mixed_region_nets = 0u32;
    let mut max_fanout = 0u32;
    let mut fanout_histogram = vec![0u32; FANOUT_BUCKET_MAX + 1];

    for net in &report.nets {
        total_nets += 1;
        if net.kind == crate::net::NetKind::Clock || net.has_clock {
            clock_nets += 1;
        }
        if net.is_floating {
            floating_nets += 1;
        }
        if net.is_mixed_region {
            mixed_region_nets += 1;
        }
        max_fanout = max_fanout.max(net.fanout_max);
        let bucket = (net.fanout_max as usize).min(FANOUT_BUCKET_MAX);
        fanout_histogram[bucket] = fanout_histogram[bucket].saturating_add(1);
    }

    GlobalSummary {
        total_nets,
        clock_nets,
        floating_nets,
        mixed_region_nets,
        max_fanout,
        fanout_histogram,
    }
}

fn region_of_net(net: &NetStats, region_field: &FieldGrid<u32>) -> u32 {
    if let Some(r) = net.region {
        return r;
    }
    // Derive deterministically: sample bbox corners and pick the smallest non-zero, else 0.
    let mut candidates = Vec::with_capacity(4);
    let coords = [
        (net.x0 as usize, net.y0 as usize),
        (net.x1 as usize, net.y0 as usize),
        (net.x0 as usize, net.y1 as usize),
        (net.x1 as usize, net.y1 as usize),
    ];
    for (x, y) in coords {
        candidates.push(region_field.get(x, y));
    }
    candidates
        .into_iter()
        .filter(|r| *r != 0)
        .min()
        .unwrap_or(0)
}

pub(crate) fn summarize_with_regions(
    report: &NetReport,
    region_field: &FieldGrid<u32>,
) -> SummaryReport {
    let global = summarize_nets(report);
    let mut regions: Vec<RegionSummary> = Vec::new();

    for net in &report.nets {
        let rid = region_of_net(net, region_field);
        let entry = match regions.iter_mut().find(|r| r.region_id == rid) {
            Some(r) => r,
            None => {
                regions.push(RegionSummary {
                    region_id: rid,
                    nets: 0,
                    drivers: 0,
                    sinks: 0,
                    clock_nets: 0,
                    floating_nets: 0,
                });
                regions.last_mut().unwrap()
            }
        };
        entry.nets += 1;
        entry.drivers = entry.drivers.saturating_add(net.driver_count);
        entry.sinks = entry.sinks.saturating_add(net.sink_count);
        if net.kind == crate::net::NetKind::Clock || net.has_clock {
            entry.clock_nets += 1;
        }
        if net.is_floating {
            entry.floating_nets += 1;
        }
    }

    SummaryReport { global, regions }
}

#[allow(dead_code)]
pub fn summarize(report: &NetReport) -> GlobalSummary {
    summarize_nets(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::FieldGrid;
    use crate::net::{NetId, NetKind, NetReport, NetStats};

    fn make_net(
        id: NetId,
        kind: NetKind,
        node_count: u32,
        driver_count: u32,
        sink_count: u32,
        fanout_max: u32,
        has_clock: bool,
        is_floating: bool,
        is_mixed_region: bool,
        region: Option<u32>,
    ) -> NetStats {
        NetStats {
            id,
            kind,
            node_count,
            driver_count,
            sink_count,
            fanout_max,
            region,
            x0: 0,
            y0: 0,
            x1: 0,
            y1: 0,
            has_clock,
            is_floating,
            is_mixed_region,
        }
    }

    #[test]
    fn summary_empty_netlist() {
        let report = NetReport { nets: Vec::new() };
        let summary = summarize_nets(&report);
        assert_eq!(summary.total_nets, 0);
        assert_eq!(summary.clock_nets, 0);
        assert_eq!(summary.floating_nets, 0);
        assert_eq!(summary.mixed_region_nets, 0);
        assert_eq!(summary.max_fanout, 0);
        assert!(summary.fanout_histogram.iter().all(|v| *v == 0));
    }

    #[test]
    fn summary_single_data_net() {
        let net = make_net(1, NetKind::Data, 3, 1, 2, 4, false, false, false, Some(0));
        let report = NetReport { nets: vec![net] };
        let summary = summarize_nets(&report);
        assert_eq!(summary.total_nets, 1);
        assert_eq!(summary.clock_nets, 0);
        assert_eq!(summary.floating_nets, 0);
        assert_eq!(summary.mixed_region_nets, 0);
        assert_eq!(summary.max_fanout, 4);
        assert_eq!(summary.fanout_histogram[4], 1);
    }

    #[test]
    fn summary_clock_and_floating() {
        let net_clock = make_net(1, NetKind::Clock, 2, 1, 1, 2, true, false, false, Some(0));
        let net_floating = make_net(2, NetKind::Data, 3, 0, 2, 1, false, true, false, Some(0));
        let report = NetReport {
            nets: vec![net_clock, net_floating],
        };
        let summary = summarize_nets(&report);
        assert_eq!(summary.total_nets, 2);
        assert_eq!(summary.clock_nets, 1);
        assert_eq!(summary.floating_nets, 1);
        assert_eq!(summary.max_fanout, 2);
        assert_eq!(summary.fanout_histogram[2], 1);
        assert_eq!(summary.fanout_histogram[1], 1);
    }

    #[test]
    fn summary_fanout_overflow_bucket() {
        let net = make_net(
            1,
            NetKind::Data,
            1,
            1,
            0,
            (FANOUT_BUCKET_MAX as u32) + 5,
            false,
            false,
            false,
            Some(0),
        );
        let report = NetReport { nets: vec![net] };
        let summary = summarize_nets(&report);
        assert_eq!(summary.max_fanout, (FANOUT_BUCKET_MAX as u32) + 5);
        assert_eq!(summary.fanout_histogram[FANOUT_BUCKET_MAX], 1);
    }

    #[test]
    fn region_summary_single_region() {
        let net1 = make_net(1, NetKind::Data, 2, 1, 1, 1, false, false, false, Some(1));
        let net2 = make_net(2, NetKind::Data, 2, 1, 1, 1, false, false, false, Some(1));
        let report = NetReport {
            nets: vec![net1, net2],
        };
        let region_field = FieldGrid::new(2, 2, 1);
        let summary = summarize_with_regions(&report, &region_field);
        assert_eq!(summary.global.total_nets, 2);
        assert_eq!(summary.regions.len(), 1);
        let r = &summary.regions[0];
        assert_eq!(r.region_id, 1);
        assert_eq!(r.nets, 2);
        assert_eq!(r.drivers, 2);
        assert_eq!(r.sinks, 2);
    }

    #[test]
    fn region_summary_multiple_regions() {
        let net1 = make_net(1, NetKind::Data, 2, 1, 1, 1, false, false, false, Some(1));
        let net2 = make_net(2, NetKind::Data, 2, 1, 1, 1, false, false, false, Some(2));
        let report = NetReport {
            nets: vec![net1, net2],
        };
        let region_field = FieldGrid::new(2, 2, 0);
        let summary = summarize_with_regions(&report, &region_field);
        assert_eq!(summary.global.total_nets, 2);
        assert_eq!(summary.regions.len(), 2);
        assert!(summary.regions.iter().any(|r| r.region_id == 1));
        assert!(summary.regions.iter().any(|r| r.region_id == 2));
    }

    #[test]
    fn region_summary_mixed_region_net() {
        let net = make_net(1, NetKind::Data, 2, 0, 1, 1, false, false, true, None);
        let report = NetReport { nets: vec![net] };
        let mut region_field = FieldGrid::new(2, 2, 0);
        region_field.set(0, 0, 2);
        region_field.set(1, 1, 3);
        let summary = summarize_with_regions(&report, &region_field);
        assert_eq!(summary.global.mixed_region_nets, 1);
        assert_eq!(summary.regions.len(), 1); // home region chosen deterministically
    }
}
