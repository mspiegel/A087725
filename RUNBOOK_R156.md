# RUNBOOK — proving optimal(R) = 156 on a 64-core machine

> **Status 2026-08-10.** Steps 1–8 have been executed on Azure VM `r156`
> (Standard_F64ams_v6, New Zealand North). Tables built and all ten SHA-256
> pins verified; all four node-identity canaries exact; E1/E2/E3 measured.
> Outstanding: E5 (checkpoint rehearsal), step 9 (exhaust-150 calibration),
> two new A/Bs added below (E6 huge pages, E7 k8 cache size), then step 10.
> Measured numbers replace the projections throughout.

The goal: prove `dist(R) ≥ 156` by exhaustive IDA\* search. The upper bound 156
is already published and replay-verified (`FINDINGS_R.md` §1), so `≥ 156`
closes the problem: **optimal(R) = 156**.

Mechanics: `cWD(R) = 144` and thresholds advance by parity (+2), so the ladder
is 144, 146, 148, 150, 152, 154. Exhausting threshold 154 proves ≥ 156.
`solve24 --prove-at-least 156` caps the ladder at 155, which by parity means
"exhaust through 154". Engine ceiling `MAX_DEPTH = 210` — no limit concerns.

The production stack is the cascade `--clm2 --zpdb8` (consult order
cWD → cLM2 → k8). On the 32 GB dev machine the cascade lost to `--lm2` at 148
**only because of paging** (~194 GB of pageins against ~49 GB of mapped
tables). With 503 GB that toll is gone, confirmed: the cascade exhausted 148
in 592.8 s here.

Rungs — measured through 148, extrapolated beyond:

| threshold | nodes | growth | wall @ W=64 |
|---|---:|---:|---:|
| 144 | 115,436,814 | — | 1.16 s |
| 146 | 4,363,759,350 | ×37.8 | 19.95 s |
| 148 | 114,245,221,757 | ×26.2 | 592.8 s |
| 150 | ~3 × 10¹² | ×~26 | ~4.3 h |
| 152 | ~8 × 10¹³ | ×~26 | ~4.7 days |
| 154 | ~2 × 10¹⁵ | ×~26 | **~4 months** |

**Exhaust-154 is 96% of the total cost**, so the growth ratio — currently one
measured transition — dominates every other variable. Step 9 gives it a
second point before any real money is committed. At $0.719/h that projects
**~$2,100** for the proof.

Machine (measured, `r156`): **x86-64 AMD EPYC 9V74 (Genoa), 64 physical cores,
1 thread/core (no SMT), 503 GiB RAM, single NUMA node, 4 KiB pages, THP =
madvise**, Ubuntu 26.04, 256 GiB Standard SSD. Requirements for a substitute
machine: ≥ 96 GB RAM (hard floor 64 GB — below that the cascade pages and
`--lm2` becomes the better engine), ~130 GB free disk, `tmux` for long runs
(the `caffeinate` habit is macOS-only). `numactl` is **not** needed on a
single-socket box.

Portability: the codebase is developed on ARM64 macOS; the one macOS-only
symbol (`_dyld_get_image_vmaddr_slide` in `solve24`) is cfg-gated to macOS,
so the default-feature build links cleanly on Linux. Do **not** enable the
`pmu-counters` feature on Linux (Apple kperf only).

Cross-architecture reproducibility is **verified**: every table artifact
rebuilt on x86-64 Linux matches its ARM64-macOS SHA-256 pin, and all four
node-identity canaries are exact. Per-core throughput ratio against the dev
Mac, for future machine sizing: **1.69× slower per core for the search**
(the number that matters), 1.39× for the k8 builds, 2–3.5× for the
latency-bound table builders.

---

## 1. Install Rust

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
rustc --version    # stable; repo is edition 2021, no nightly features
```

## 2. Clone the repository

```sh
git clone https://github.com/mspiegel/A087725.git   # or git@github.com:mspiegel/A087725.git
cd A087725
git checkout flat-parallel   # or wherever the cascade + Linux-gate work lives when you run this
```

The table binaries are gitignored; only the SHA-256 sidecars (`data/*.sha256`)
are tracked. Everything is rebuilt on this machine (Step 4) and verified
against the pinned sidecars (Step 5).

## 3. Build in release mode

```sh
cargo build --release --features sha
```

Default features (`parallel`, `mmap`, `cli`) plus `sha` cover every binary this
runbook uses: `solve24`, `build_wd24`, `build_cwd_table`, `build_cwd_artifacts`,
`build_pdb24`. Binaries land in `target/release/`.

## 4. Build the table artifacts

All commands run from the repo root. Order matters — later stages consume
earlier outputs. Every builder runs its own validation gate (round-trip,
dominance, or A\* oracle) and panics on failure.

### 4a. WD base table and the cWD overlay prerequisite

```sh
tmux new -s tables
target/release/build_wd24 --out data/wd24.bin --verify-sha data/wd24.bin.sha256
target/release/build_cwd_table        # → data/cwd_single.bin (reads data/wd24.bin)
```

`build_cwd_table` runs five backward product-graph BFS passes (one per goal
line) and checks the WD-layer invariant for every contingency. Single-threaded
but fast: **~7 minutes** measured on the dev machine (~83 s per line;
ledger entry 2026-08-04 in `data/r_flat_k8_lazy.txt`).

### 4b. The cWD table family (five artifacts, dependency order)

```sh
target/release/build_cwd_artifacts all
```

Stages and rough costs (dev-machine timings; the lm/lm2 stages are compute-bound
and will go faster here):

| stage | consumes | produces | cost |
|---|---|---|---|
| `mm` | wd24.bin + cwd_single.bin | `cwd_mm.bin` (3.0 GB) | minutes |
| `lm` | (pure compute) | `cwd_lm.bin` (0.85 GB) | minutes, ~3 GB peak |
| `lm2` | cwd_lm.bin (dominance gate) | `cwd_lm2.bin` (2.2 GB) | tens of minutes, ~9 GB peak |
| `lm-mm` | cwd_lm.bin + cwd_lm2.bin | `cwd_lm_mm.bin` (3.0 GB) | minutes |
| `lm1l` | cwd_mm.bin | `cwd_lm1l_mm.bin` (10.5 GB) | ~35 min, ~18 GB peak |

### 4c. The three 8-tile zero-aware PDBs (k8)

The 8-8-8 partition (tile groups from the pinned build,
`FINDINGS_R.md` / commit 6826e2a):

```sh
target/release/build_pdb24 --zero-aware --tiles 1,2,3,4,6,7,8,9        --out data/pdb24_k8_a.zbin --verify-sha data/pdb24_k8_a.sha256
target/release/build_pdb24 --zero-aware --tiles 5,10,14,15,19,20,23,24 --out data/pdb24_k8_b.zbin --verify-sha data/pdb24_k8_b.sha256
target/release/build_pdb24 --zero-aware --tiles 11,12,13,16,17,18,21,22 --out data/pdb24_k8_c.zbin --verify-sha data/pdb24_k8_c.sha256
```

Each group was ~8 h at 8 threads via the frontier-free 2-bit builder
(~20 GB peak per group). **Measured on 64 cores: 83.5 / 91.7 / 88.7 min**
(a / b / c) — the builder is fully rayon-parallel and sweeps the whole packed
array once per BFS depth level, so wall time is ~109-116 s per level and
tracks eccentricity almost exactly. Groups must report eccentricities
**46 / 50 / 46** (Σ-ecc = 142); b runs deepest because its tiles are spread
along the right and bottom edges rather than clustered. Run them serially —
they already use every core.

## 5. Verify artifact SHA-256 against the pinned references

`--verify-sha` on `build_wd24`/`build_pdb24` already checked those four
inline. This step re-checks everything from the files, independent of the
builders:

```sh
for a in wd24.bin cwd_single.bin cwd_mm.bin cwd_lm.bin cwd_lm2.bin cwd_lm_mm.bin cwd_lm1l_mm.bin; do
  want=$(tr -d ' \n' < data/$a.sha256); got=$(sha256sum data/$a | cut -d' ' -f1)
  [ "$want" = "$got" ] && echo "OK   $a" || echo "FAIL $a  want $want  got $got"
done
for g in a b c; do
  want=$(tr -d ' \n' < data/pdb24_k8_$g.sha256); got=$(sha256sum data/pdb24_k8_$g.zbin | cut -d' ' -f1)
  [ "$want" = "$got" ] && echo "OK   pdb24_k8_$g.zbin" || echo "FAIL pdb24_k8_$g.zbin  want $want  got $got"
done
```

Reference values (the tracked sidecars, 2026-08-08 state):

| artifact | bytes | sha256 |
|---|---:|---|
| `data/wd24.bin` | 590,854,479 | `c1652299a8a098b71d543bb9cd20e6b7d8dc9dd6dfee44d89b64e0accc37d734` |
| `data/cwd_single.bin` | 1,181,708,926 | `5a5b15a1bd8279803ded809fd599dc04b9912545fc0c95fe47f99d093190ad08` |
| `data/cwd_mm.bin` | 3,001,165,488 | `7bd7e712877c7fa331130b48a72ccc7c60e73b15c07a384096cbfa4ba573d3a4` |
| `data/cwd_lm.bin` | 853,456,447 | `b5e6c0daaf4d1b456ad2bc605150adf4b2232f9c0b1eaf125be141143662fd58` |
| `data/cwd_lm2.bin` | 2,166,466,347 | `2e9304c8d8227d09c0604301f3e63877797847c2421dbb5432e5bf252bb08681` |
| `data/cwd_lm_mm.bin` | 3,001,165,488 | `0e7ba985b596415a604dea21e262a10e880b25c21199405a6c9884150d9d90dd` |
| `data/cwd_lm1l_mm.bin` | 10,504,079,168 | `fd199a82e3886d3e54544fce6caefc0a7f65905efa5f87232936b6b40b201915` |
| `data/pdb24_k8_a.zbin` | 10,919,800,104 | `7e4772e79d7c4d369eec96d2145fa008a8c15f5bdf67a066e06e2efefd2ac5f1` |
| `data/pdb24_k8_b.zbin` | 10,919,800,104 | `ccf23af56a62092967e4cd80481d2b406268b38d2b8237b9088f37c1aea77344` |
| `data/pdb24_k8_c.zbin` | 10,919,800,104 | `f421e9e1699b2bd99d1069b023dd028a4ec096c93c885b0f3af3b8f060e44fcf` |

All ten pins are platform-independent: every builder serializes in sorted-key
(or fixed-layout) order. The `cwd_single.bin` pin above supersedes the
pre-2026-08-09 value `32db8593…`, which was raw-HashMap-order serialization —
reproducible only on aarch64, where it was built (content proven equal via the
round-trip-verified `cwd_mm.bin` pin matching on both architectures).

A sha mismatch on a bit-identical rebuild means a real problem (toolchain
nondeterminism or corruption) — stop and compare against the dev machine
before proceeding. The node-identity gates in Step 6 are the deeper check:
they exercise the tables through the search itself.

## 6. Engine gates — node-identity canaries

The R position (blank at cell 0, tile 25−i at cell i):

```sh
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
```

Run each gate and compare the per-threshold node counts printed on stderr.
These are exact — any deviation is a red flag (table corruption, build skew):

```sh
# gate 1: k8 2-tier ladder through 146
target/release/solve24 --position "$R" --prove-at-least 147 --zpdb8
#   threshold 144: 269,180,917   threshold 146: 8,539,130,554

# gate 2: clm2 standalone at 144
target/release/solve24 --position "$R" --prove-at-least 145 --clm2
#   threshold 144: 134,801,951

# gate 3: full cascade through 146
target/release/solve24 --position "$R" --prove-at-least 147 --clm2 --zpdb8
#   threshold 146: 4,363,759,350

# gate 4: parallel driver is node-identical — repeat gate 3 with --parallel
target/release/solve24 --position "$R" --prove-at-least 147 --clm2 --zpdb8 --parallel
#   same counts, bit for bit
```

Optional deeper gate (≈114 B nodes, worth one run to also exercise memory at
breadth): `--prove-at-least 149 --clm2 --zpdb8 --parallel` → threshold 148 =
114,245,221,757.

## 7. Machine preparation for timed runs

- **Idle box.** No editors, no rust-analyzer, no other jobs during timed A/Bs.
  Bracket every batch with an untouched-engine canary run (gate 3 above is a
  good ~90 s canary) — if the canary drifts more than a couple of percent,
  the measurements in between are suspect.
- **NUMA — not applicable.** `r156` is single-socket, one NUMA node, so
  `numactl` buys nothing. Re-check with `numactl --hardware` on any
  substitute machine.
- **SMT — none.** The Fasv6 family runs 1 thread per core, so all 64 vCPUs
  are physical cores. `lscpu` confirms `Thread(s) per core: 1`.
- **Page size — 4 KiB, and it costs us.** x86-64 fixes the base page size in
  hardware; no kernel option changes it (64 KiB base pages are ARM64-only).
  The kernel already caches the tables in 2 MiB folios but never PMD-maps
  them into the process, and `CONFIG_READ_ONLY_THP_FOR_FS` is off, so no
  `madvise` on a file mapping helps:

  ```text
  FileHugePages:  38023168 kB   <- 38 GB of tables in huge folios
  FilePmdMapped:         0 kB   <- none huge-mapped into the process
  ```

  The lever is `solve24 --hugepages`, which copies the tables into anonymous
  `MADV_HUGEPAGE` memory (needs `transparent_hugepage/enabled` = `madvise`
  or `always`, and enough RAM to hold ~49 GB resident). Quantified in E6.
- Long runs: `tmux` + `--checkpoint` (below). No sleep management needed on a
  server.

## 8. Tuning experiments for 64 cores

Goal: *good enough to exploit the parallelism*, not perfect. All experiments
at exhaust-148 (`--prove-at-least 149`) with the cascade — deep enough to see
contention, cheap enough to iterate at ~10 min per run.

### E1 — thread scaling curve — **DONE** (re-measured at final constants)

Node count identical at every W (114,245,221,757), re-confirming node-identity
across thread counts. Measured twice: first at the original constants
(SPLIT_TARGET 4096, K8_SHARED_BITS 21), then again at the final ones
(32768, 27) after E2 and E7:

| W | wall | Mn/s | per core | efficiency | (original wall) |
|---:|---:|---:|---:|---:|---:|
| 16 | 1486.64 s | 76.85 | 4.803 | 100% | 1929.41 s |
| 32 | 812.45 s | 140.62 | 4.394 | 91.5% | 1045.82 s |
| 48 | 570.95 s | 200.10 | 4.169 | 86.8% | 743.62 s |
| **64** | **444.73 s** | **256.88** | 4.014 | **83.6%** | 592.83 s |

**Run proofs at W=64** — rayon's default here, so no `RAYON_NUM_THREADS`
needed. The tuning work cut exhaust-148 by 25.0% (592.83 -> 444.73 s).

Two things this settles. There is no memory-bandwidth cliff — the curve
declines smoothly rather than flattening — which is what retires E4/E9's
size bump. And the ~16% parallel loss at W=64 is **not** explained by the
shared-cache pressure fixed in E7: efficiency moved only 81.4% -> 83.6%,
because the larger cache lifted every thread count nearly equally (+29.8% at
W=16, +33.3% at W=64). The loss is a separate, still-unidentified mechanism;
see E8.

### E2 — SPLIT_TARGET — **DONE** (4096 → 32768)

`SPLIT_TARGET` (`src/puzzle24/search/flat.rs`) is the number of subtree-root
work units the sequential frontier grows before going parallel. 4096 was
chosen for W=8 — ~500 units/worker of stealing slack, needed because subtree
sizes vary by orders of magnitude. At W=64 that is only ~64 units/worker,
thin enough that a few monster subtrees serialize each threshold's tail.

Swept at exhaust-148, W=64, via the `parallel-profile` build (that feature is
the only thing exposing the `FLAT_SPLIT_TARGET` env knob, so the production
binary stays knob-free):

| SPLIT_TARGET | 4096 | 16384 | 32768 | 65536 |
|---|---:|---:|---:|---:|
| wall (s) | 595.1 | 576.6 | 580.1 | 577.5 |

Everything at or above 16 K is one plateau within 0.6% — tied — while 4096
costs ~3%. **Set to 32768** (commit `5be4aec`), the middle of the plateau,
which restores the original ~500-units-per-worker ratio at W=64. Rebuilt
without `parallel-profile` afterwards.

### E3 — shared k8 cache under 64 threads — **DONE, and it found pressure**

The `--zpdb8` tier fronts the 33 GB zPDB mmaps with one process-wide shared
cache (`K8SharedCache`, 2²¹ × AtomicU64 = 16 MB). It is lock-free — relaxed
`u64` load on probe, relaxed store on miss-fill, no CAS or retry — so it was
never a *serialization* risk. The question was capacity under 64-way sharing,
and the answer is that it matters:

| | W=16 | W=64 |
|---|---:|---:|
| k8 hit rate | **72.905%** | **66.271%** |
| k8 misses | 44.3 B | 55.2 B (**+24.6%**) |
| LM2 per-worker cache | 99.757% | 99.749% |

Every extra miss is a rank walk plus a probe into the 33 GB mmap. The
dev-machine finding that cache size was irrelevant does **not** transfer —
that was 8 threads sharing the key stream. The per-worker LM2 cache is
unaffected, so this is specific to the shared structure. Follow-up is **E7**.

Method note: `probe-cache-stats` costs ~14× wall time at W=64 (atomic counter
contention across all workers). Use it for hit rates only — **never** read a
timing off an instrumented build.

### E4/E9 — per-worker cache sizing — **MEASURED, no change**

Each worker holds ~27 MB of private caches (cWD probe 8.4 + LM2 12.6 + LM 4 +
LM1L 3.4), so 64 workers hold ~1.7 GB against 256 MiB of L3 (32 MiB per CCD,
8 cores each) — about 7x oversubscribed, versus 1.7x at W=16. The hypothesis
was that 64 copies now evict each other and the caches should *shrink*.
Measured at exhaust-148, W=64, scaling all four constants together:

| ~MB/worker | 7 | 14 | **27 (current)** | 54 |
|---|---:|---:|---:|---:|
| wall (s) | 522.3 | 507.1 | **496.5** | 495.1 |

Monotone penalty for shrinking, a tie for growing: **the existing sizing is
already the knee at 64 threads.** Keep `WORKER_CACHE_BITS = 18`,
`LM_CACHE_BITS = 18`, `LM2_CACHE_BITS = 19`, `LM1L_CACHE_BITS = 15`.

The hypothesis was wrong for the reason the source already gave: these caches
exist to avoid probing the multi-GB tables, not to achieve cache residency,
and an L3-missing cache lookup still beats a random probe into a 4.4 GB or
33 GB mmap.

**The generalisable result**, from E7 and E9 together: constants guarding
*per-thread* state transfer across machines unchanged; constants guarding
*shared* state must be re-measured at every thread count, because unique-key
pressure through a shared structure scales with W. The shared k8 cache's
8-thread sizing left 21% on the table at 64; the per-worker sizing lost
nothing.

### E5 — checkpoint rehearsal

Before the long rungs, prove the recovery path at scale:

```sh
mkdir -p runs/ckpt156
target/release/solve24 --position "$R" \
  --prove-at-least 151 --clm2 --zpdb8 --parallel --checkpoint runs/ckpt156
# kill it ~30 min in (Ctrl-C), then re-run the identical command:
# completed thresholds and finished units restore; only unfinished work re-searches.
```

**Fold this into step 9 rather than running it separately.** Start the
exhaust-150 calibration with `--checkpoint`, kill it half an hour in, resume,
confirm the restore line, and let the resumed run continue to completion as
the calibration itself. That buys the crash-recovery proof for ~5 minutes of
re-searched work instead of a separate multi-hour run, and proves it at the
scale where it actually matters.

Checkpoint records are keyed to position + tier config + tree version (NOT to
`--prove-at-least`), so the same directory carries across the staged proof
runs below. At most one in-flight unit per worker is lost on a kill.

### E6 — huge pages for the tables (NEW — highest-value untested lever)

`solve24 --hugepages` copies each table into anonymous `MADV_HUGEPAGE` memory
instead of mapping its file, so probes translate through 2 MiB pages. The
motivation is measured (step 7): the kernel already holds 38 GB of tables in
huge folios but PMD-maps none of them, and `CONFIG_READ_ONLY_THP_FOR_FS` is
off, so the automatic paths cannot help. 49 GB at 4 KiB needs ~12.8 M TLB
entries against a ~3,000-entry L2 TLB; at 2 MiB it needs ~25 K.

E3's 66% k8 hit rate means ~55 B misses into the 33 GB zPDBs per exhaust-148
run, each a random dependent access — exactly the traffic huge pages help.

```sh
# A/B at exhaust-148, W=64, back to back, idle box
target/release/solve24 --position "$R" --prove-at-least 149 --clm2 --zpdb8 \
  --parallel > runs/e6_base.log 2>&1
target/release/solve24 --position "$R" --prove-at-least 149 --clm2 --zpdb8 \
  --parallel --hugepages > runs/e6_huge.log 2>&1
```

Baseline to beat: **592.8 s / 192.71 Mn/s**. Node counts must stay identical
(huge pages change only translation). Costs ~49 GB of resident anonymous
memory and adds a one-time ~2–4 min load; check `AnonHugePages` in
`/proc/meminfo` during the run to confirm THP actually backed the region.
If this wins, use `--hugepages` for step 9 and the proof runs.

### E7 — k8 shared-cache size (NEW — motivated by E3)

E3 showed 64-way sharing costs 24.6% more k8 misses than 16-way. Raise
`K8_SHARED_BITS` (`src/puzzle24/search/flat.rs`) from 21 (16 MB) to 24
(128 MB) — a one-line const edit plus rebuild — and re-run the exhaust-148
baseline. 128 MB is nothing against 503 GB.

Two outcomes worth distinguishing: if wall time improves, keep it and record
the new hit rate; if the hit rate rises but wall time does not, the misses
were not on the critical path and the earlier dev-machine conclusion still
holds at scale — record that and revert, because a bigger cache costs L3
residency.

Run E6 and E7 **before** step 9, so the calibration measures the final
configuration.

## 9. Calibration gate — before committing to 152/154

Everything downstream rests on the per-+2 growth ratio, which currently has
**one** measured value. Run exhaust-150 on the final configuration (after E6
and E7 decide `--hugepages` and `K8_SHARED_BITS`), with `--checkpoint` so E5
folds in, and record nodes and wall.

Current model, from measured 148 (114,245,221,757 nodes / 592.8 s at W=64):

| ratio | 154 nodes | wall | cost @ $0.719/h |
|---:|---:|---:|---:|
| 20 | 7.3 × 10¹⁴ | 1.5 months | ~$780 |
| **26.2 (measured 146→148)** | **2.0 × 10¹⁵** | **~4 months** | **~$2,100** |
| 33 | 4.1 × 10¹⁵ | 8.2 months | ~$4,300 |

The ratio enters *cubed* from 148, so it swings the total by more than every
hardware decision available — Hetzner vs Azure, buying vs renting, and every
alternative architecture all live inside a 2× band. This one measurement,
which costs about **$3**, is worth more than all of them.

Decision points:

- 154 lands in weeks-to-a-few-months → proceed to step 10.
- 154 lands in many months → reassess before spending. Options: more machines
  (needs an Azure quota increase beyond 64 regional vCPUs *and* multi-host
  sharding, which is not built — though the driver's deterministic indexed
  work units make a `--shard K/N` filter a small change), or a cheaper
  provider (Hetzner AX162 bare metal is ~1.8× cheaper per core), or stopping.

The 152 and 154 rungs are **calibration data, not results** — only
exhaust-154 proves anything new; each completed rung refines the estimate,
and everything banks in the checkpoint dir regardless.

## 10. The proof runs

Staged, all sharing one checkpoint dir (each stage banks the ladder for the
next; re-running with a higher `--prove-at-least` resumes, never repeats).
The intermediate stages exist for calibration and risk-reduction only — the
result is stage 3 or nothing:

```sh
# W=64 is rayon's default here; add --hugepages if E6 wins.
RUN="target/release/solve24 --position \"$R\" \
     --clm2 --zpdb8 --parallel --checkpoint runs/ckpt156"

# stage 1 (calibration): exhausts through 150; re-fit the growth ratio
$RUN --prove-at-least 152
# stage 2 (calibration): exhausts 152; final go/no-go on 154's cost
$RUN --prove-at-least 154
# stage 3 (the result): exhausts 154 → optimal(R) = 156
$RUN --prove-at-least 156
```

Deallocate between stages if there is a gap before the next decision:
`az vm deallocate -g A087725 -n r156` stops compute billing (a guest-OS
shutdown does **not**); `az vm start` resumes with the disk and IP intact.

Success criterion per stage: `Lower bound: depth >= T` on stdout with no
`NOT A PROOF` marker. Keep the full stderr logs (per-threshold node counts
are the proof's audit trail — archive `runs/` and the logs).

The root σ-orbit split stays on throughout (it auto-enables on R and is a
~2× saving; decision 2026-08-09). Its soundness has already been
cross-checked end-to-end on the dev machine's proven rungs, so no
`--no-root-orbit-split` arm is scheduled here.

## 11. Open questions

Resolved by the 2026-08-09/10 run: DRAM ceiling (no plateau; 81% efficiency
at W=64), topology (single socket, one NUMA node, no SMT), and k8 cache
capacity (real pressure at 64-way — see E3/E7).

Still open:

1. **Growth-ratio drift** — ×26.2 is measured only at 146→148. FINDINGS
   expects it to climb with depth; if it reaches 33 the proof doubles in
   cost. Step 9 is the gate, and it is the single highest-leverage number
   remaining.
2. **Huge pages** — quantified motivation exists (step 7) but no A/B yet.
   E6 settles it; potentially the largest single throughput lever left.
3. **Bare metal vs virtualized** — every TLB miss inside a VM pays a
   two-dimensional page walk through the hypervisor's nested tables. A
   Hetzner AX162 (48 Genoa cores, €199/mo, ~1.8× cheaper per core) would
   remove that, but 48 bare cores vs 64 virtualized is unmeasured. The clean
   test is one hourly-billed AX162 running the identical exhaust-148
   benchmark against the 592.8 s baseline.
4. **Multi-host sharding** — not built, but cheaper than previously thought:
   the driver already splits each threshold into deterministic, stably
   indexed work units with per-unit checkpoint records, so `--shard K/N`
   filtering on `idx % N` is a small change. The real work is making shards
   agree on the threshold ladder (an explicit `--bounds` list, or a sync
   point between rungs). Only worth building if step 9 says 154 is too slow
   on one machine.
