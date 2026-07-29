//! Page-locality of the merged-table probe, behind the `probe-locality` feature.
//!
//! # The question this answers
//!
//! At exhaust-146 the search takes 0.403 L2 TLB misses per node against 0.264 at
//! exhaust-144 (+52.7%, `puzzle24::pmu`), and there is ~1 probe per node — so
//! roughly 40% of probes miss the L2 TLB. The obvious fix is a smaller table,
//! but whether that helps depends on something nobody has measured: TLB misses
//! are driven by the *hot* subset of the table, not its total size. If the hot
//! set is far beyond TLB reach, shrinking the table 44% barely moves the miss
//! rate; if it sits near the boundary, shrinking it could move a lot.
//!
//! # Method
//!
//! Record the 16 KiB page of every probed cell — the real address of the value
//! hashbrown returns, not a model of it — and push it through several fixed-size
//! LRU caches at once. Each cache stands in for a TLB of that many entries, so
//! the output is a hit-rate-vs-reach curve for the actual access sequence.
//!
//! Reading a smaller table off that curve needs no second run: a table shrunk by
//! factor `f` spans `f x` as many pages, so a TLB of `N` entries covers the same
//! *fraction* of it as a TLB of `N/f` entries covers of the current one. The
//! 2.48 GB table (56% of today's) at the machine's ~3000-entry L2 TLB therefore
//! behaves like today's table at ~5360 entries — both points are on the curve.
//!
//! Off by default: an LRU probe per cache per node is far too slow to ship.

use std::collections::HashMap;

/// 16 KiB pages — the Apple Silicon page size (`vm_stat` reports 16384).
const PAGE_SHIFT: u32 = 14;

/// TLB reaches to simulate. 3072 brackets the M2's L2 TLB; the larger entries
/// are what a proportionally smaller table would look like at that same reach.
const SIZES: &[usize] = &[256, 512, 1024, 2048, 3072, 4096, 5360, 8192, 16384, 65536];

/// Software-cache sizes to simulate, in **entries** (32 B each). This is a
/// different question from the TLB curve: a compact cache in front of the probe
/// lives or dies on entry-level reuse, not page-level. Recorded at exhaust-144 in
/// an earlier session: 77.4 / 89.4 / 96.4 / 98.9% at 1K / 4K / 16K / 64K.
const ENTRY_SIZES: &[usize] = &[1024, 4096, 16384, 65536, 262144, 1048576];

/// `log2` of the merged-table entry stride, so an address becomes an entry index.
const ENTRY_SHIFT: u32 = 5;

/// An O(1) LRU over page numbers: hash map to arena slot, intrusive list.
struct Lru {
    cap: usize,
    map: HashMap<u64, u32>,
    prev: Vec<u32>,
    next: Vec<u32>,
    page: Vec<u64>,
    head: u32,
    tail: u32,
    free: Vec<u32>,
    hits: u64,
    misses: u64,
}

const NIL: u32 = u32::MAX;

impl Lru {
    fn new(cap: usize) -> Self {
        Lru {
            cap,
            map: HashMap::with_capacity(cap * 2),
            prev: Vec::with_capacity(cap),
            next: Vec::with_capacity(cap),
            page: Vec::with_capacity(cap),
            head: NIL,
            tail: NIL,
            free: Vec::new(),
            hits: 0,
            misses: 0,
        }
    }

    #[inline]
    fn unlink(&mut self, i: u32) {
        let (p, n) = (self.prev[i as usize], self.next[i as usize]);
        if p != NIL {
            self.next[p as usize] = n;
        } else {
            self.head = n;
        }
        if n != NIL {
            self.prev[n as usize] = p;
        } else {
            self.tail = p;
        }
    }

    #[inline]
    fn push_front(&mut self, i: u32) {
        self.prev[i as usize] = NIL;
        self.next[i as usize] = self.head;
        if self.head != NIL {
            self.prev[self.head as usize] = i;
        }
        self.head = i;
        if self.tail == NIL {
            self.tail = i;
        }
    }

    #[inline]
    fn touch(&mut self, pg: u64) {
        if let Some(&i) = self.map.get(&pg) {
            self.hits += 1;
            if self.head != i {
                self.unlink(i);
                self.push_front(i);
            }
            return;
        }
        self.misses += 1;
        let i = if self.map.len() >= self.cap {
            // Evict the least-recently-used entry and reuse its slot.
            let victim = self.tail;
            self.unlink(victim);
            let old = self.page[victim as usize];
            self.map.remove(&old);
            victim
        } else if let Some(i) = self.free.pop() {
            i
        } else {
            self.prev.push(NIL);
            self.next.push(NIL);
            self.page.push(0);
            (self.prev.len() - 1) as u32
        };
        self.page[i as usize] = pg;
        self.map.insert(pg, i);
        self.push_front(i);
    }
}

/// Simulates every reach in [`SIZES`] against one probe stream.
pub struct ProbeLocality {
    lrus: Vec<Lru>,
    entry_lrus: Vec<Lru>,
    distinct: HashMap<u64, u64>,
    probes: u64,
    limit: u64,
}

impl ProbeLocality {
    pub fn new(limit: u64) -> Self {
        ProbeLocality {
            lrus: SIZES.iter().map(|&c| Lru::new(c)).collect(),
            entry_lrus: ENTRY_SIZES.iter().map(|&c| Lru::new(c)).collect(),
            distinct: HashMap::new(),
            probes: 0,
            limit,
        }
    }

    /// Record one probe by the **address of the returned value**, so this
    /// measures where the table actually lives rather than a model of it.
    #[inline]
    pub fn note(&mut self, addr: usize) {
        if self.probes >= self.limit {
            return;
        }
        self.probes += 1;
        let pg = (addr >> PAGE_SHIFT) as u64;
        *self.distinct.entry(pg).or_insert(0) += 1;
        for l in &mut self.lrus {
            l.touch(pg);
        }
        let ent = (addr >> ENTRY_SHIFT) as u64;
        for l in &mut self.entry_lrus {
            l.touch(ent);
        }
    }

    pub fn saturated(&self) -> bool {
        self.probes >= self.limit
    }

    pub fn report(&self) -> String {
        let mut s = format!(
            "probe-locality: {} probes, {} distinct 16 KiB pages ({:.2} GB spanned)\n",
            self.probes,
            self.distinct.len(),
            self.distinct.len() as f64 * 16384.0 / 1e9
        );
        // How concentrated is the traffic? A small hot core would mean shrinking
        // the table helps little; a flat distribution means it helps a lot.
        let mut counts: Vec<u64> = self.distinct.values().copied().collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        let total: u64 = counts.iter().sum();
        let mut acc = 0u64;
        for (frac, label) in [(0.5, "50%"), (0.8, "80%"), (0.9, "90%"), (0.99, "99%")] {
            let need = (total as f64 * frac) as u64;
            let mut n = 0usize;
            acc = 0;
            for &c in &counts {
                acc += c;
                n += 1;
                if acc >= need {
                    break;
                }
            }
            s += &format!(
                "  {label} of probes land in {n} pages ({:.1}% of pages, {:.0} MB)\n",
                n as f64 / counts.len() as f64 * 100.0,
                n as f64 * 16384.0 / 1e6
            );
        }
        let _ = acc;
        s += "  TLB reach      hit rate   miss rate\n";
        for (i, &cap) in SIZES.iter().enumerate() {
            let l = &self.lrus[i];
            let t = (l.hits + l.misses).max(1);
            let note = match cap {
                3072 => "   <- ~M2 L2 TLB",
                5360 => "   <- equivalent reach for a 2.48 GB table",
                _ => "",
            };
            s += &format!(
                "  {cap:>9}  {:>9.2}%  {:>9.2}%{note}\n",
                l.hits as f64 / t as f64 * 100.0,
                l.misses as f64 / t as f64 * 100.0
            );
        }
        s += "  software cache (entries, 32 B each)\n";
        s += "     entries        RAM   hit rate\n";
        for (i, &cap) in ENTRY_SIZES.iter().enumerate() {
            let l = &self.entry_lrus[i];
            let t = (l.hits + l.misses).max(1);
            s += &format!(
                "  {cap:>10}  {:>7.1} MB  {:>8.2}%\n",
                cap as f64 * 32.0 / 1e6,
                l.hits as f64 / t as f64 * 100.0
            );
        }
        s
    }
}

// ------------------------------- global hook --------------------------------

use std::cell::RefCell;

thread_local! {
    static TRACKER: RefCell<Option<ProbeLocality>> = const { RefCell::new(None) };
}

/// Begin recording, capped at `limit` probes (the LRU simulation is far slower
/// than the search, so a cap keeps the run bounded).
pub fn start(limit: u64) {
    TRACKER.with(|t| *t.borrow_mut() = Some(ProbeLocality::new(limit)));
}

#[inline]
pub fn note_probe(addr: usize) {
    TRACKER.with(|t| {
        if let Some(p) = t.borrow_mut().as_mut() {
            p.note(addr);
        }
    });
}

pub fn report() -> String {
    TRACKER.with(|t| {
        t.borrow()
            .as_ref()
            .map(|p| p.report())
            .unwrap_or_else(|| "probe-locality: not started".into())
    })
}

pub fn saturated() -> bool {
    TRACKER.with(|t| t.borrow().as_ref().is_some_and(|p| p.saturated()))
}
