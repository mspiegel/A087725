//! Named CPU performance counters via Apple's private `kperf`/`kperfdata` APIs.
//!
//! # Why this exists
//!
//! The search runs ~18% slower per node at exhaust-146 than at exhaust-144
//! (FINDINGS_R §8x). Top-down counters attribute the extra cycles to the
//! back-end, and a line-level diff of the two regimes shows the cost spread
//! *proportionally* across every line rather than concentrated anywhere — the
//! hash probe's share moves 6.0% → 6.1%. Timer-based sampling cannot separate
//! "genuinely diffuse" from "one stall smeared across neighbours by
//! out-of-order retirement", so it cannot name the mechanism.
//!
//! Raw counter deltas can. `xctrace`'s CPU Counters template records 12 unnamed
//! PMU slots, and normalising them per instruction retired across the two
//! regimes shows slot 4 growing **1.75×** where cycles grow 1.18× — four times
//! more than anything else. But the trace does not record which event each slot
//! holds, and neither `xctrace` nor `ktrace artrace --kperf` can select events
//! by name from the command line (the former's counting mode is runtime plugin
//! state; the latter's `--kperf` takes only timers).
//!
//! This module asks for events **by name** instead, so the mechanism can be
//! identified rather than inferred.
//!
//! # Why counts, not times
//!
//! Event counts per node are independent of clock frequency, so this is valid
//! on battery — unlike every timing measurement in this session. And because
//! the counters are read at each threshold boundary inside a *single* process,
//! the 144-vs-146 comparison carries no cross-run drift at all.
//!
//! # Requires root
//!
//! `kpc_force_all_ctrs_set` needs privileges. Run the driver under `sudo`. If
//! any step fails the caller gets an error and the search still runs — the
//! counters are diagnostic, never load-bearing.

use std::ffi::{c_char, c_int, c_void, CString};

// ---------------------------------------------------------------- FFI ------

type KpepDb = c_void;
type KpepConfig = c_void;
type KpepEvent = c_void;

const KPC_CLASS_FIXED_MASK: u32 = 1 << 0;
const KPC_CLASS_CONFIGURABLE_MASK: u32 = 1 << 1;

struct Api {
    // kperf
    kpc_force_all_ctrs_set: unsafe extern "C" fn(c_int) -> c_int,
    kpc_set_config: unsafe extern "C" fn(u32, *mut u64) -> c_int,
    kpc_set_counting: unsafe extern "C" fn(u32) -> c_int,
    kpc_set_thread_counting: unsafe extern "C" fn(u32) -> c_int,
    kpc_get_thread_counters: unsafe extern "C" fn(u32, u32, *mut u64) -> c_int,
    // kperfdata
    kpep_db_create: unsafe extern "C" fn(*const c_char, *mut *mut KpepDb) -> c_int,
    kpep_db_event: unsafe extern "C" fn(*mut KpepDb, *const c_char, *mut *mut KpepEvent) -> c_int,
    kpep_config_create: unsafe extern "C" fn(*mut KpepDb, *mut *mut KpepConfig) -> c_int,
    kpep_config_force_counters: unsafe extern "C" fn(*mut KpepConfig) -> c_int,
    kpep_config_add_event:
        unsafe extern "C" fn(*mut KpepConfig, *mut *mut KpepEvent, u32, *mut u32) -> c_int,
    kpep_config_kpc_classes: unsafe extern "C" fn(*mut KpepConfig, *mut u32) -> c_int,
    kpep_config_kpc_count: unsafe extern "C" fn(*mut KpepConfig, *mut usize) -> c_int,
    kpep_config_kpc_map: unsafe extern "C" fn(*mut KpepConfig, *mut usize, usize) -> c_int,
    kpep_config_kpc: unsafe extern "C" fn(*mut KpepConfig, *mut u64, usize) -> c_int,
}

unsafe fn sym<T>(h: *mut c_void, name: &str) -> Result<T, String> {
    let c = CString::new(name).unwrap();
    let p = libc_dlsym(h, c.as_ptr());
    if p.is_null() {
        return Err(format!("symbol {name} not found"));
    }
    Ok(std::mem::transmute_copy(&p))
}

// Minimal dlopen/dlsym bindings (avoids pulling in a crate for three calls).
extern "C" {
    #[link_name = "dlopen"]
    fn libc_dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    #[link_name = "dlsym"]
    fn libc_dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_LAZY: c_int = 1;

impl Api {
    fn load() -> Result<Self, String> {
        unsafe {
            let kperf_path =
                CString::new("/System/Library/PrivateFrameworks/kperf.framework/kperf").unwrap();
            let kpd_path =
                CString::new("/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata")
                    .unwrap();
            let kperf = libc_dlopen(kperf_path.as_ptr(), RTLD_LAZY);
            if kperf.is_null() {
                return Err("cannot dlopen kperf".into());
            }
            let kpd = libc_dlopen(kpd_path.as_ptr(), RTLD_LAZY);
            if kpd.is_null() {
                return Err("cannot dlopen kperfdata".into());
            }
            Ok(Api {
                kpc_force_all_ctrs_set: sym(kperf, "kpc_force_all_ctrs_set")?,
                kpc_set_config: sym(kperf, "kpc_set_config")?,
                kpc_set_counting: sym(kperf, "kpc_set_counting")?,
                kpc_set_thread_counting: sym(kperf, "kpc_set_thread_counting")?,
                kpc_get_thread_counters: sym(kperf, "kpc_get_thread_counters")?,
                kpep_db_create: sym(kpd, "kpep_db_create")?,
                kpep_db_event: sym(kpd, "kpep_db_event")?,
                kpep_config_create: sym(kpd, "kpep_config_create")?,
                kpep_config_force_counters: sym(kpd, "kpep_config_force_counters")?,
                kpep_config_add_event: sym(kpd, "kpep_config_add_event")?,
                kpep_config_kpc_classes: sym(kpd, "kpep_config_kpc_classes")?,
                kpep_config_kpc_count: sym(kpd, "kpep_config_kpc_count")?,
                kpep_config_kpc_map: sym(kpd, "kpep_config_kpc_map")?,
                kpep_config_kpc: sym(kpd, "kpep_config_kpc")?,
            })
        }
    }
}

// --------------------------------------------------------------- driver ----

/// A configured set of named PMU events, counted on the calling thread.
pub struct Pmu {
    api: Api,
    /// Index into the raw counter buffer for each requested event, in order.
    map: Vec<usize>,
    names: Vec<String>,
    counter_count: usize,
    classes: u32,
}

impl Pmu {
    /// Configure the given events by name (see `/usr/share/kpep/*.plist` for the
    /// per-chip event list; this machine is `as2`). Requires root.
    pub fn new(events: &[&str]) -> Result<Self, String> {
        let api = Api::load()?;
        unsafe {
            let mut db: *mut KpepDb = std::ptr::null_mut();
            if (api.kpep_db_create)(std::ptr::null(), &mut db) != 0 {
                return Err("kpep_db_create failed".into());
            }
            let mut cfg: *mut KpepConfig = std::ptr::null_mut();
            if (api.kpep_config_create)(db, &mut cfg) != 0 {
                return Err("kpep_config_create failed".into());
            }
            if (api.kpep_config_force_counters)(cfg) != 0 {
                return Err("kpep_config_force_counters failed".into());
            }
            for name in events {
                let cname = CString::new(*name).unwrap();
                let mut ev: *mut KpepEvent = std::ptr::null_mut();
                if (api.kpep_db_event)(db, cname.as_ptr(), &mut ev) != 0 {
                    return Err(format!("unknown PMU event {name:?} on this CPU"));
                }
                if (api.kpep_config_add_event)(cfg, &mut ev, 0, std::ptr::null_mut()) != 0 {
                    return Err(format!("cannot add event {name:?} (too many counters?)"));
                }
            }
            let mut classes: u32 = 0;
            (api.kpep_config_kpc_classes)(cfg, &mut classes);
            let mut count: usize = 0;
            (api.kpep_config_kpc_count)(cfg, &mut count);
            let mut map = vec![0usize; events.len()];
            (api.kpep_config_kpc_map)(cfg, map.as_mut_ptr(), map.len() * 8);
            let mut regs = vec![0u64; count.max(1)];
            (api.kpep_config_kpc)(cfg, regs.as_mut_ptr(), regs.len() * 8);

            if (api.kpc_force_all_ctrs_set)(1) != 0 {
                return Err("kpc_force_all_ctrs_set failed — run under sudo".into());
            }
            if (api.kpc_set_config)(classes, regs.as_mut_ptr()) != 0 {
                return Err("kpc_set_config failed".into());
            }
            if (api.kpc_set_counting)(classes) != 0 {
                return Err("kpc_set_counting failed".into());
            }
            if (api.kpc_set_thread_counting)(classes) != 0 {
                return Err("kpc_set_thread_counting failed".into());
            }
            let counter_count = (KPC_CLASS_FIXED_MASK | KPC_CLASS_CONFIGURABLE_MASK) as usize;
            let _ = counter_count;
            Ok(Pmu {
                api,
                map,
                names: events.iter().map(|s| s.to_string()).collect(),
                counter_count: count.max(events.len()) + 8,
                classes,
            })
        }
    }

    /// Current counts for the configured events, in the order they were given.
    pub fn read(&self) -> Vec<u64> {
        let mut buf = vec![0u64; self.counter_count + 8];
        unsafe {
            (self.api.kpc_get_thread_counters)(0, buf.len() as u32, buf.as_mut_ptr());
        }
        self.map
            .iter()
            .map(|&i| buf.get(i).copied().unwrap_or(0))
            .collect()
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn classes(&self) -> u32 {
        self.classes
    }
}
