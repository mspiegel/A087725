//! Table loading with an optional huge-page path.
//!
//! The engine's tables (merged cWD, the LM/LM2 artifact, the cLM2 joint table
//! and the three k8 zPDBs — ~49 GB together) are normally `mmap`ed straight
//! from their files. That is the right default: startup is a page-table
//! operation, and the page cache is shared between processes.
//!
//! It also forfeits the TLB, which this workload cares about more than most.
//! Search probes are random and dependent across the whole 49 GB, so nearly
//! every probe misses the TLB and pays a page walk. Measured on the 64-core
//! Azure box (kernel 7.0, 4 KiB base pages):
//!
//! ```text
//! FileHugePages:  38023168 kB   <- 38 GB of tables held in 2 MiB folios
//! FilePmdMapped:         0 kB   <- none of it PMD-mapped into the process
//! ```
//!
//! The page cache already stores the data in huge folios, but the mapping is
//! still built from 4 KiB PTEs, so the reach never materialises;
//! `CONFIG_READ_ONLY_THP_FOR_FS` is off on that kernel, so `khugepaged` will
//! not collapse a read-only file mapping either, and no `madvise` on the file
//! mapping can help.
//!
//! What does work is owning the pages: copy each table into an *anonymous*
//! region marked `MADV_HUGEPAGE`, which THP will back with 2 MiB pages
//! (`transparent_hugepage/enabled` must be `madvise` or `always`). 49 GB then
//! needs ~25 K TLB entries instead of ~12.8 M.
//!
//! The cost is real RSS: the tables stop being evictable page cache and become
//! resident anonymous memory. That is free on a 500 GB machine and fatal on a
//! 32 GB laptop, which is why this is opt-in via `solve24 --hugepages` rather
//! than a default.
//!
//! The flag is stored process-globally rather than threaded through the
//! loaders: they have 70+ call sites across binaries, examples and tests, and
//! none of the others want the behaviour. It is set once from the CLI before
//! any table is opened, so there is no race.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

static USE_HUGEPAGES: AtomicBool = AtomicBool::new(false);

/// Route later [`map_table`] calls through the anonymous huge-page path.
/// Call once at startup, before loading any table.
pub fn set_hugepages(on: bool) {
    USE_HUGEPAGES.store(on, Ordering::Relaxed);
}

pub fn hugepages_enabled() -> bool {
    USE_HUGEPAGES.load(Ordering::Relaxed)
}

/// Read-only view of `path`: an ordinary file mapping, or — when
/// [`set_hugepages`] asked for it — a private anonymous copy backed by
/// transparent huge pages. Both return the same `Mmap` type, so callers are
/// unchanged.
pub fn map_table(path: &Path) -> std::io::Result<memmap2::Mmap> {
    let f = std::fs::File::open(path)?;
    if !hugepages_enabled() {
        // SAFETY: write-once-then-immutable build artifact.
        return unsafe { memmap2::Mmap::map(&f) };
    }

    let len = f.metadata()?.len() as usize;
    let mut anon = memmap2::MmapMut::map_anon(len)?;
    // Ask for 2 MiB backing before the region is populated — THP can only
    // honour this while the pages are still unfaulted.
    #[cfg(target_os = "linux")]
    if let Err(e) = anon.advise(memmap2::Advice::HugePage) {
        eprintln!(
            "hugepages: MADV_HUGEPAGE on {} failed ({e}); continuing with base pages",
            path.display()
        );
    }

    use std::io::Read;
    let mut r = std::io::BufReader::with_capacity(1 << 22, f);
    r.read_exact(&mut anon[..])?;
    eprintln!(
        "hugepages: {} copied into anonymous memory ({:.1} GB resident)",
        path.display(),
        len as f64 / 1e9
    );
    anon.make_read_only()
}
