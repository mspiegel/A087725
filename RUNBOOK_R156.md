# RUNBOOK — proving optimal(R) = 156 on a 144-core machine

The goal: prove `dist(R) ≥ 156` by exhaustive IDA\* search. The upper bound 156
is already published and replay-verified (`FINDINGS_R.md` §1), so `≥ 156`
closes the problem: **optimal(R) = 156**.

Mechanics: `cWD(R) = 144` and thresholds advance by parity (+2), so the ladder
is 144, 146, 148, 150, 152, 154. Exhausting threshold 154 proves ≥ 156.
`solve24 --prove-at-least 156` caps the ladder at 155, which by parity means
"exhaust through 154". Engine ceiling `MAX_DEPTH = 210` — no limit concerns.

The production stack is the cascade `--clm2 --zpdb8` (consult order
cWD → cLM2 → k8). On the 32 GB dev machine the cascade loses to `--lm2` at
148 **only because of paging** (~194 GB of pageins; the mapped tables total
~49 GB). On a ≥ 96 GB machine that toll disappears and the cascade's node cut
(2.52× at 148, growing with depth) should convert to wall clock.

Known measured rungs (sequential, dev machine, for scale):

| threshold | stack | nodes | growth |
|---|---|---:|---:|
| 144 | `--clm2` | 134,801,951 | — |
| 146 | `--clm2 --zpdb8` | 4,363,759,350 | — |
| 148 | `--clm2 --zpdb8` | 114,245,221,757 | ×26.2 |
| 150 | extrapolated | ~3 × 10¹² | ×~26 |
| 152 | extrapolated | ~8 × 10¹³ | ×~26 |
| 154 | extrapolated | ~2 × 10¹⁵ | ×~26 |

At a plausible 144-core cascade rate (0.5–1.5 Gn/s — Step 8 measures the real
number), 150 is ~an hour, 152 is ~a day or two, and **154 is weeks-to-months
and dominates everything**. Step 9's calibration gate re-fits the growth ratio
from rungs measured on this machine before committing to 152/154.

Assumptions: **ARM64 (aarch64) Linux server**, ≥ 96 GB RAM (hard floor 64 GB —
below that the cascade pages and `--lm2` becomes the better engine), ~130 GB
free disk (~50 GB artifacts + ~60 GB transient build peak + checkpoints),
`tmux` or `nohup` for long runs (the `caffeinate` habit is macOS-only).
`numactl` installed if the box is multi-socket.

Portability: the codebase is developed on ARM64 macOS; the one macOS-only
symbol (`_dyld_get_image_vmaddr_slide` in `solve24`) is now cfg-gated to
macOS, so the default-feature build links cleanly on Linux. Do **not** enable
the `pmu-counters` feature on Linux (Apple kperf only). Make sure the branch
with that gate is pushed before cloning on the big machine.

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
(~20 GB peak per group); 144 cores should cut that substantially, and the
three groups can also run concurrently in separate shells (3 × 20 GB peak +
whatever else is resident — fine at ≥ 96 GB). Build log should report
eccentricities 46 / 50 / 46 (Σ-ecc = 142).

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
| `data/cwd_single.bin` | 1,181,708,926 | `32db859345da0a9e736887f766db9f173264e5e7dd8237c16d1510da5a5012fa` |
| `data/cwd_mm.bin` | 3,001,165,488 | `7bd7e712877c7fa331130b48a72ccc7c60e73b15c07a384096cbfa4ba573d3a4` |
| `data/cwd_lm.bin` | 853,456,447 | `b5e6c0daaf4d1b456ad2bc605150adf4b2232f9c0b1eaf125be141143662fd58` |
| `data/cwd_lm2.bin` | 2,166,466,347 | `2e9304c8d8227d09c0604301f3e63877797847c2421dbb5432e5bf252bb08681` |
| `data/cwd_lm_mm.bin` | 3,001,165,488 | `0e7ba985b596415a604dea21e262a10e880b25c21199405a6c9884150d9d90dd` |
| `data/cwd_lm1l_mm.bin` | 10,504,079,168 | `fd199a82e3886d3e54544fce6caefc0a7f65905efa5f87232936b6b40b201915` |
| `data/pdb24_k8_a.zbin` | 10,919,800,104 | `7e4772e79d7c4d369eec96d2145fa008a8c15f5bdf67a066e06e2efefd2ac5f1` |
| `data/pdb24_k8_b.zbin` | 10,919,800,104 | `ccf23af56a62092967e4cd80481d2b406268b38d2b8237b9088f37c1aea77344` |
| `data/pdb24_k8_c.zbin` | 10,919,800,104 | `f421e9e1699b2bd99d1069b023dd028a4ec096c93c885b0f3af3b8f060e44fcf` |

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
- **NUMA** (if multi-socket): run everything under `numactl --interleave=all`.
  The ~49 GB of mmap'd tables and the 16 MB shared k8 cache otherwise land on
  whichever node touches them first, and the far socket eats the latency.
- **Page size**: check `getconf PAGESIZE`. ARM64 Linux kernels ship with
  4 KiB or 64 KiB pages; a 64 KiB kernel natively cuts TLB pressure on the
  multi-GB random-probe tables (a measured dominant cost of this workload)
  16-fold — worth knowing which regime the box is in before interpreting any
  throughput number. (All working-set arithmetic in the repo's notes assumes
  the dev machine's 16 KiB pages.)
- **Huge pages** (4 KiB kernels): check
  `cat /sys/kernel/mm/transparent_hugepage/enabled`; `always` is worth an
  A/B against `madvise` (the engine does not madvise).
- **SMT**: most ARM server parts (Ampere, Graviton, Grace) are one thread
  per core, so 144 is likely 144 physical — confirm with `lscpu`, and if SMT
  does exist, E1's W sweep covers it.
- Long runs: `tmux` + `--checkpoint` (below). No sleep management needed on a
  server.

## 8. Tuning experiments for 144 cores

Goal: *good enough to exploit the parallelism*, not perfect. Expected total
tuning budget: a day. All experiments at exhaust-148 (`--prove-at-least 149`)
with the cascade — deep enough to see contention (144's tree is too small to
occupy 144 workers; 146 is marginal), cheap enough to iterate.

### E1 — thread scaling curve (run first)

```sh
for W in 18 36 72 108 144; do
  RAYON_NUM_THREADS=$W target/release/solve24 --position "$R" \
    --prove-at-least 149 --clm2 --zpdb8 --parallel 2>&1 | tee runs/e1_w$W.log
done
```

Record threshold-148 wall and Mn/s per W. This one curve answers most
questions at once: where DRAM bandwidth saturates, whether SMT helps, and
what the realistic proof-run rate is (feeds Step 9's extrapolation). If the
curve flattens hard before 144, the memory system — not any tuning constant —
is the ceiling, and the remaining experiments matter less.

### E2 — SPLIT_TARGET (answer: raise it to ~64 K)

`SPLIT_TARGET` (`src/puzzle24/search/flat.rs`, const, currently **4096**) is
the number of subtree-root work units the sequential frontier grows before
going parallel. 4096 was chosen for W=8 — ~500 units/worker of work-stealing
slack, needed because subtree sizes vary by orders of magnitude. At W=144,
4096 is only ~28 units/worker — thin enough that a handful of monster subtrees
serialize the tail of every threshold.

**Recommendation: 65,536** (2¹⁶ ≈ 455 units/worker, preserving the measured
slack ratio). Split cost is negligible at proof depths: at exhaust-148's
114 B nodes, 64 K units still average ~1.7 M nodes each; unit records and
checkpoint lines scale linearly and stay trivial. The value is not sharp —
anything in 32 K–128 K should sit on the same plateau. Confirm with one sweep
using the env-gated profile build (the env knob exists only under the
`parallel-profile` feature, so the production binary stays knob-free):

```sh
cargo build --release --features "sha parallel-profile"
for S in 4096 16384 65536 131072; do
  FLAT_SPLIT_TARGET=$S RAYON_NUM_THREADS=144 target/release/solve24 --position "$R" \
    --prove-at-least 149 --clm2 --zpdb8 --parallel 2>&1 | tee runs/e2_s$S.log
done
```

Then hard-set the winner as the `SPLIT_TARGET` const and **rebuild without
`parallel-profile`** — proof runs use the constant, per the no-env-knobs
policy.

### E3 — the shared k8 cache under 144 threads (answer: likely fine; one A/B)

The `--zpdb8` tier fronts the 33 GB zPDB mmaps with **one process-wide
shared cache** (`K8SharedCache`, 2²¹ × AtomicU64 = 16 MB). Design facts that
bound the contention risk:

- Lock-free: relaxed single-`u64` load on probe, relaxed store on miss-fill.
  No CAS, no locks, no retry loops — nothing serializes. Correctness is
  interleaving-proof (tag check rejects foreign entries), so this is purely
  a throughput question.
- Writes are rare per node: k8 is consulted only at children the earlier
  tiers fail to prune (~6–12% of nodes), and only cache *misses* store
  (8–24% of consults at the measured 76–92% hit rates) → a store on roughly
  1–2% of nodes. Coherence-line invalidations at that rate, spread over 2M
  slots, are noise next to the DRAM traffic the misses themselves generate.

So: **not expected to be a contention source in the lock sense; the real
exposure is NUMA placement and aggregate hit rate.** With 144 threads sharing
2M slots, unique-key pressure rises (more distinct subtrees in flight →
more capacity evictions). Two cheap mitigations to A/B once:

1. `K8_SHARED_BITS` 21 → **24** (16 MB → 128 MB; one-line const edit +
   rebuild). On the dev machine size was measured irrelevant (2¹⁹–2²⁵ moved
   wall less than canary spread) — but that was 8 threads sharing the key
   stream; 144 threads is a different regime, and 128 MB costs nothing here.
2. `numactl --interleave=all` (Step 7) so the cache and the mmaps stripe
   across nodes.

Diagnostic if suspicion remains: build with `--features probe-cache-stats`
and compare the k8 hit rate at W=8 vs W=144 — a large drop confirms capacity
pressure (fix: more bits); a stable hit rate with poor E1 scaling points at
DRAM/NUMA instead.

### E4 — per-worker ("split") cache sizing (answer: adequate; one optional sweep)

Each rayon worker owns private front caches; nothing is shared, so there is
no contention — the question is only RAM footprint and hit rate:

| cache | const | size/worker | ×144 |
|---|---|---:|---:|
| cWD probe cache | `WORKER_CACHE_BITS = 18` | 8.4 MB | 1.2 GB |
| LM2 branch cache | `LM2_CACHE_BITS = 19` | ~15 MB | ~2.2 GB |
| LM1L joint cache | `LM1L_CACHE_BITS = 15` | 3.4 MB | 0.5 GB |

≈ 27 MB/worker, ≈ 3.9 GB total at W=144 — trivial next to 49 GB of tables on
a ≥ 96 GB box. **Keep the defaults.** The 18-bit choice was validated by the
insight that these caches exist to avoid probing the multi-GB tables (DRAM +
TLB cost), *not* to fit in L2 — that reasoning transfers to any machine. The
one thing 144 threads changes: aggregate table-miss traffic scales ×18 vs the
measured W=8 config, so if E1 shows a memory-bound plateau, one notch larger
(`WORKER_CACHE_BITS` 18→19, `LM2_CACHE_BITS` 19→20; +25 MB/worker, ~3.6 GB
more total) is the cheapest lever to cut DRAM pressure. Sweep only if E1
motivates it; expect single-digit percent either way.

### E5 — checkpoint rehearsal

Before the long rungs, prove the recovery path at scale:

```sh
mkdir -p runs/ckpt156
RAYON_NUM_THREADS=144 target/release/solve24 --position "$R" \
  --prove-at-least 151 --clm2 --zpdb8 --parallel --checkpoint runs/ckpt156
# kill it mid-threshold-150 (Ctrl-C), then re-run the same command:
# completed thresholds and finished units restore; only unfinished work re-searches.
```

Checkpoint records are keyed to position + tier config + tree version (NOT to
`--prove-at-least`), so the same directory carries across the staged proof
runs below. At most one in-flight unit per worker is lost on a kill.

## 9. Calibration gate — before committing to 152/154

From E1/E5 output, record for this machine: nodes and wall at exhaust-148 and
exhaust-150, cascade, W=chosen. Fit the per-+2 growth ratio (dev-machine
ratio: ×26 at 146→148, drifting slowly; use the local 148→150 number) and
extrapolate 152 and 154. The 152 and 154 rungs are **calibration data, not
results** — only exhaust-154 proves anything new; each completed rung's
node count refines the 154 estimate (and everything banks in the checkpoint
dir regardless). Decision points:

- 154 lands in weeks → proceed.
- 154 lands in many months → stop and reassess (consider more machines —
  the checkpoint format is per-worker append-only, but multi-host sharding
  is *not* built, so that would be new work).

## 10. The proof runs

Staged, all sharing one checkpoint dir (each stage banks the ladder for the
next; re-running with a higher `--prove-at-least` resumes, never repeats).
The intermediate stages exist for calibration and risk-reduction only — the
result is stage 3 or nothing:

```sh
RUN="numactl --interleave=all target/release/solve24 --position \"$R\" \
     --clm2 --zpdb8 --parallel --checkpoint runs/ckpt156"

# stage 1 (calibration): exhausts through 150; re-fit the growth ratio
RAYON_NUM_THREADS=144 $RUN --prove-at-least 152
# stage 2 (calibration): exhausts 152; final go/no-go on 154's cost
RAYON_NUM_THREADS=144 $RUN --prove-at-least 154
# stage 3 (the result): exhausts 154 → optimal(R) = 156
RAYON_NUM_THREADS=144 $RUN --prove-at-least 156
```

Success criterion per stage: `Lower bound: depth >= T` on stdout with no
`NOT A PROOF` marker. Keep the full stderr logs (per-threshold node counts
are the proof's audit trail — archive `runs/` and the logs).

The root σ-orbit split stays on throughout (it auto-enables on R and is a
~2× saving; decision 2026-08-09). Its soundness has already been
cross-checked end-to-end on the dev machine's proven rungs, so no
`--no-root-orbit-split` arm is scheduled here.

## 11. Open questions this runbook can't settle from the dev machine

1. **DRAM bandwidth ceiling** — the dev machine's parallel loss was half
   memory contention at W=8. Whether 144 workers hammering ~49 GB of tables
   saturate the memory system at W≪144 is the single biggest unknown; E1
   answers it and everything else adjusts to that answer.
2. **Topology** — single- or dual-socket, and per-socket memory channel
   count (`lscpu`, `numactl --hardware`)? Changes E1's interpretation and
   whether the interleave advice matters at all.
3. **Page size / huge pages** — 4 KiB vs 64 KiB kernel, and on 4 KiB,
   `always` vs `madvise` THP: an easy A/B with possibly large payoff on
   this TLB-bound workload; untested anywhere so far.
4. **k8 shared-cache capacity at 144-way sharing** — hit rate was
   size-insensitive at W=8; E3's probe-cache-stats comparison says whether
   that survives ×18 thread count.
5. **Growth-ratio drift** — the ×26/rung ratio has only been measured
   through 148 on the cascade. If it climbs at 150+ (it is expected to),
   the 154 estimate moves materially; Step 9 gates on the locally measured
   number, not the dev-machine one.
