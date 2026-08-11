# A087725 — working notes

Proving `optimal(R) = 156` for the 24-puzzle (OEIS A087725). The upper bound is
published and replay-verified (`FINDINGS_R.md` §1), so exhausting threshold 154
closes the problem.

Everything below is enforced by a gate, a test, or a file you can point at. If
a rule here has no such backing, delete it rather than trusting it.

## Orientation

- `src/puzzle24/search/engine.rs` — **the** search engine. Flat (iterative)
  IDA\*, one hardcoded stack: cWD + move-DFA + neighbour-WD child pre-prune +
  root σ-orbit split. Bounded lower-bound mode only; it cannot solve. Consumed
  by `solve24`.
- `src/puzzle24/search/recursive.rs` — the generic heuristic-driven ladder.
  Solves optimally and honours a deadline, which the engine cannot. Two real
  callers: `gen_corridors` (`--mode exact`, `--mode audit`) and `ml/eval.rs`.
- `src/puzzle24/search/oracle.rs` — 180 frozen cases from the deleted recursive
  engine. **Cannot be regenerated.** A mismatch means the engine's tree changed,
  not that the fixture is stale.
- `RUNBOOK_R156.md` — the proof procedure, measured timings, artifact SHA pins.
- `records/r_flat_k8_lazy.txt` — the verdict ledger. Grep it before calling any
  lever untried; the measured graveyard is larger than the FINDINGS summaries.

## Gates — all green before any commit

```sh
cargo build --all-targets                    # 0 warnings
cargo build --all-targets --all-features     # 0 warnings
cargo clippy --all-targets                   # 0 warnings
cargo fmt --check
cargo test                                   # 493 pass, 32 ignored
```

`cargo build --all-targets --no-default-features` is **known-broken** and
predates this file: tests and examples reference `puzzle8::puzzle24`
unconditionally while the module sits behind a feature. `cargo build --lib
--no-default-features` is clean; keep it that way.

## Node identity is the gate for engine changes

The engine's tree is fixed. Nothing may change a pruning decision — only
evaluate the same predicates more cheaply. Two checks, in order of cost:

```sh
cargo test                       # includes the 180-case frozen oracle
R="0 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
target/release/solve24 --config large --position "$R" \
  --prove-at-least 145 --clm2 --zpdb8 --parallel     # → 115,436,814 nodes
```

That count is exact and is the value `runs/ckpt156/main.ckpt` records for
threshold 144. `RUNBOOK_R156.md` §6 has the full canary set.

`debug_assert!`s document the engine's carried-state invariants (XOR-maintained
k8 keys, front-cache transparency, `D ≥ WD`, a unit's carried `h`). A proof runs
`--release`, where they compile out, so the only thing that executes them is:

```sh
cargo test engine_cascade_assertions_hold_at_144 -- --ignored --nocapture
```

Run it in the **default (debug) profile** — `--release` still checks the node
count but skips every assertion. ~4 min.

## The proof record

`runs/ckpt156/` is the append-only evidence for thresholds 144–150. **Never
point `--checkpoint` at it** — the solver appends. Copy it somewhere scratch
first. `git status runs/ckpt156` must come back empty after any experiment.

Checkpoint records are keyed by a run id over (tree version, position,
orbit_split, tier flags, neighbor_prune) — *not* over `--config`. Resuming under
a different `--config` is safe (`ckpt_unit_fp` guards every record, so one is
never applied to a different subtree) but re-searches every banked unit of an
unfinished threshold. `runs/ckpt156` needs `--config large`.

## Conventions

- **No env knobs.** Configuration is clap flags; tuning values are consts.
  `--config standard|large` is required and has no default — the two tunings
  give identical answers and differ only in wall clock, so a wrong choice is
  invisible.
- **Constants that guard shared state must be re-measured per thread count**;
  constants that guard per-thread state transfer across machines. This is
  measured, not a guess: `K8_SHARED_BITS` (shared) moved 21% from W=8 to W=64,
  while the per-worker cache sizes were re-measured at W=64 and kept.
- **Slow tests are `#[ignore]`d** with artifacts, runtime and invocation in the
  string: `#[ignore = "needs data/foo.bin; ~4 min; run with --ignored"]`.
  Artifact-dependent tests skip with a message rather than failing.
- **Wrap long runs in `caffeinate -i`** so the machine does not idle-sleep
  mid-search.
- **Timed A/Bs need an idle box and a canary.** rust-analyzer's post-edit clippy
  steals a core; bracket every batch with an untouched-engine reference run.

## Commit messages

Verify each claim against the diff before committing. Two messages in this
repo's history describe intent rather than what landed — `9ab9acc` says it
dropped the macOS dyld print and did not (fixed in `bc779fd`). `git log -S
<symbol>` settles "was this actually removed?" in one command.

Historical docs (`PLAN.md`, `FINDINGS_R.md`, `PUZZLE24.md`) record what was
executed at the time and may show removed flags. Leave them; rewriting a
research log is not a mechanical rename.
