//! Opt-in phase profiler for the ML hot paths (DAVI train_step, BWAS search).
//!
//! Zero overhead when disabled (a single relaxed atomic load guards every
//! record). When enabled, `sync()` forces device completion at phase boundaries
//! so each phase's GPU work is attributed to *that* phase rather than to a later
//! sync point (Metal executes lazily) — this makes profiled totals larger than
//! unprofiled ones, which is the expected trade for accurate attribution.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use candle_core::Device;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ACC: OnceLock<Mutex<BTreeMap<&'static str, (Duration, u64)>>> = OnceLock::new();

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

#[inline]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

fn acc() -> &'static Mutex<BTreeMap<&'static str, (Duration, u64)>> {
    ACC.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Add `start.elapsed()` to the running total for `name` (and bump its call
/// count) iff profiling is enabled.
#[inline]
pub fn record_if(name: &'static str, start: Instant) {
    if is_enabled() {
        let mut m = acc().lock().unwrap();
        let e = m.entry(name).or_insert((Duration::ZERO, 0));
        e.0 += start.elapsed();
        e.1 += 1;
    }
}

/// Force device completion iff profiling is enabled (no-op otherwise, so normal
/// runs are unaffected). Call at a phase boundary before `record_if` when the
/// phase enqueues GPU work that would otherwise not execute until a later sync.
#[inline]
pub fn sync(device: &Device) {
    if is_enabled() {
        let _ = device.synchronize();
    }
}

pub fn reset() {
    acc().lock().unwrap().clear();
}

/// Tab-separated report, phases sorted by total time descending.
pub fn report() -> String {
    let m = acc().lock().unwrap();
    let mut rows: Vec<(&'static str, (Duration, u64))> = m.iter().map(|(k, v)| (*k, *v)).collect();
    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    let total_ms: f64 = rows.iter().map(|(_, (d, _))| d.as_secs_f64() * 1000.0).sum();
    let mut out = String::from(
        "── phase profile (profiled runs force per-phase syncs; totals exceed unprofiled) ──\n\
         phase\ttotal_ms\t%\tcalls\tms/call\n",
    );
    for (name, (dur, n)) in rows {
        let ms = dur.as_secs_f64() * 1000.0;
        let per = if n > 0 { ms / n as f64 } else { 0.0 };
        let pct = if total_ms > 0.0 { 100.0 * ms / total_ms } else { 0.0 };
        out += &format!("{name}\t{ms:.1}\t{pct:.1}\t{n}\t{per:.3}\n");
    }
    out
}
