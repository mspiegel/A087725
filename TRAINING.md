# TRAINING — adversarial generator/solver co-training (15-puzzle proof of concept)

## Context

This project's original goal was pushing the 24-puzzle diameter lower bound. Extensive
investigation (WD heuristic analysis, ZPDB partition sizing, the `hb` wildcard-complement
heuristic, an exhaustive WD-maximization search) converged on a hard structural finding: every
cheap admissible heuristic caps out at WD = 140 on deep boards, the gap to the true diameter
(≥152) requires either infeasible exact search or Rokicki-scale pattern databases, and there is
no published *suboptimal* solution to even the best-known hard board (`R`) despite it being the
most-discussed instance in the field.

That reframed the goal: rather than chasing an unreachable lower-bound proof, build **two tools
that improve each other** — a **solver** that finds short (not necessarily optimal) solutions to
hard boards, graded by solution length, and a **generator** that learns to construct boards that
are maximally difficult for the current solver. This is a genuine ML research build (training
loop, learned models, non-admissible search), not an extension of the existing admissible-
heuristic/exact-proof machinery, which stays untouched.

Because the 24-puzzle has **no ground truth beyond depth ~30** (established this session), there
is no way to tell whether a trained system is learning anything real versus just gaming its own
search. The 15-puzzle, by contrast, has **complete, exact ground truth** already in this repo
(depth histogram, the 17 known depth-80 antipodes, and fully-enumerated depth-76–80 board sets —
see `ENUMERATION.md`). So this plan builds the system as a **15-puzzle-specific proof of concept**,
deliberately not generalized across puzzle sizes (no shared `PuzzleEnv`-style abstraction —
`puzzle15` and `puzzle24` are separate, non-generic modules in this codebase today, and
introducing an abstraction before a second concrete case exists would be premature). Porting to
the 24-puzzle is future, out-of-scope work; several design choices below are flagged as expected
**not** to transfer as-is.

The design was validated against real, verified precedent across a spread of adversarial-training
literature (not invented from scratch): DeepCubeA/DAVI (Agostinelli et al., *Nature Machine
Intelligence* 2019) for the value-learning half, and GANCO ("Generative Adversarial Training for
Neural Combinatorial Optimization Models") for the generator-training half — the closest-matching
precedent found, since it targets the same shape of problem (a non-differentiable solver being
challenged by a learned generator, trained via RL, with reward defined as regret against a fixed
non-learned baseline).

## Design

**Solver.** A learned cost-to-go value network `V(s)` (raw one-hot board state in, no admissible
heuristic anywhere in its input or training — WD/Manhattan/etc. are never fed into the network).
Trained via DAVI/Approximate Value Iteration: sample scramble depth `k` uniformly from a fixed
`[1, K]` (not an advancing curriculum gate — verified this is DeepCubeA's actual method; the
"learns near states before far states" behavior is an emergent property of Bellman-bootstrap
dynamics, not an engineered threshold). Regression target: `V_target(s) = min over legal neighbors
s' of (1 + V_target_net(s'))`, with `V(goal) = 0` fixed, using a periodically-synced target network
for stability. Deployed via a new **Batch Weighted A\* Search (BWAS)** driver — `f(x) = g(x) +
weight·V(x)` — since no weighted/non-admissible search exists in this codebase today (every
existing `idastar_inc*` hardcodes `f = g + h` for admissible-only search). Solver score: shorter
valid solution is better; failing within a fixed **node-expansion budget** (not wall-clock) is a
defined penalty, strictly worse than any real solution length.

**Generator.** A learned policy network that constructs a board by choosing a sequence of legal
moves starting from `GOAL` (like DeepCubeA's scrambling, but the moves are chosen by a trained
policy instead of uniformly at random), walk length drawn from a fixed range. Trained via
**REINFORCE-style policy gradient** — required because its reward depends on running the solver's
actual search, which involves discrete argmax/priority-queue decisions and cannot be
backpropagated through (confirmed: this is exactly why GANCO, and the broader
adversarial-curriculum literature generally, use RL for this side rather than direct backprop).
**Reward (GANCO-style regret against a non-learning baseline):** for a generated board `b`, run
both the current solver (`V`-driven BWAS) and a fixed baseline — the existing, already-tested
`idastar` + `WalkingDistanceHeuristic` (exact, admissible) — on `b`. Generator reward =
`(learned solver's cost) − (baseline's cost)`. This deliberately targets boards where the *learned*
solver underperforms a simple fixed reference, a more informative training signal than raw
difficulty. WD is used **only** as this external comparison; it never touches the solver's own
network or search.

**Training loop.** Simple alternation: train the solver for N steps, freeze it, train the generator
for M steps against the frozen solver, repeat. **Confirmed:** the solver's training distribution
each round is a mix of uniform-random scrambles *and* generator-produced boards (not 100%
generator-produced) — this degrades gracefully (an untrained generator round 1 looks like uniform
sampling) and avoids the solver's entire training diet depending on a generator that could
collapse.

**Tooling.** `candle` (Rust-native ML, `candle-core`/`candle-nn` 0.11) with the Metal backend
(this machine is an Apple M2 Pro — no CUDA path exists there, and `candle`'s Metal backend is
real, actively maintained, and well-suited to a network this size). `Device::metal_if_available`
gives CPU fallback for free if Metal has issues.

**Success criteria (this is a PoC, not a finished system):** the alternation loop runs stably for
a meaningful number of rounds with no crashes, NaNs, or generator collapse; the solver's average
solution length on a held-out random-board set trends down over training; evaluated against the
15-puzzle's 17 known depth-80 antipodes, the solver lands in a sane neighborhood of **DeepCubeA's
own verified benchmark on this exact experiment — solutions averaging +2.8 moves over optimal
(80)** on those boards. Matching or beating that isn't required; landing wildly outside it is the
signal something's broken.

**Known non-transferable choice (flag for future 24-puzzle work, not to solve now):** the exact
`idastar`+WD baseline is fast here (even the hardest known 15-puzzle board solves in ~9s) but is
expected to be **infeasible on the 24-puzzle** — WD is not even the right general-purpose 24-puzzle
heuristic (this project's own findings show pattern databases beat WD on typical/general boards;
WD only wins on maximally-adversarial boards like `R`), the state space is ~7.4×10¹¹× larger, and
an *improving* generator would make an *exact* baseline's solve time balloon unpredictably over
training. A 24-puzzle port would need a budgeted/best-effort baseline (this project's existing
`select` heuristic policy under a node budget, not plain exact WD).

## Implementation

All new code lives under a new `src/puzzle15/ml/` module, gated behind a new `ml` Cargo feature
(the whole module, not per-item — it cannot compile at all without `candle-core`/`candle-nn`).
Reuses existing, verified APIs throughout:

- `puzzle15::state::{State, Move, GOAL}` — `blank_pos`, `legal_moves_at`, `apply_at`
  (`src/puzzle15/state.rs`)
- `puzzle15::rank::{rank, unrank}` (`src/puzzle15/rank.rs`)
- `puzzle15::search::{idastar, WalkingDistanceHeuristic}` (`src/puzzle15/search/idastar.rs`,
  `search/walking_distance.rs`, re-exported in `search/mod.rs`) — the exact baseline, unmodified
- `puzzle15::enumerate::antipodes::load_ranks` (17 known antipodes) and
  `enumerate::store::Store::read_ranks_file` (the `data/enum15/depth{76..80}.ranks` files) for
  evaluation ground truth

New files:

```
src/puzzle15/ml/
  mod.rs            pub mod decls, #[cfg(feature = "ml")]
  device.rs         pick_device() -> candle_core::Device (Metal-if-available, else CPU)
  encoding.rs       State -> one-hot [256] f32 Tensor, single + batch
  value_net.rs       ValueNet (candle_nn Module): [256] -> scalar cost-to-go
  scramble.rs         dependency-free xorshift64 RNG (matches this repo's existing inline-PRNG
                      test convention) + random-walk-from-GOAL sampler, plain and policy-driven
  davi.rs             DAVI training loop: Bellman targets, target-net sync, train_step()
  bwas.rs             NEW weighted/batch search driver (f = g + weight*h, node-budgeted,
                      generic over any batched heuristic closure — serves the solver, the WD
                      baseline via idastar (see below), and is unit-tested against the existing
                      admissible idastar's output on shallow states before ever touching a
                      learned network)
  policy_net.rs       PolicyNet: [256] -> 4 move logits, legal-move masking, categorical sampling
  generator.rs        REINFORCE training loop + GANCO-style regret reward
  alternate.rs        orchestrates solver-round / freeze / generator-round loop + checkpointing
  eval.rs             antipodes / depth76-80 / random-holdout evaluation, reports vs. DIAMETER=80
  checkpoint.rs       thin VarMap::save/load wrapper + plain-text run metadata

src/bin/
  train_ml15.rs       launches the full alternation loop
  eval_ml15.rs        loads a checkpoint, runs eval.rs, prints the antipode/holdout report

tests/
  puzzle15_ml_smoke.rs   feature="ml" integration smoke test: tiny loop, asserts no NaNs
```

`src/puzzle15/mod.rs` gets one new line: `#[cfg(feature = "ml")] pub mod ml;`

**`Cargo.toml`:**
```toml
[features]
ml = ["parallel", "dep:candle-core", "dep:candle-nn"]

[dependencies]
candle-core = { version = "0.11", optional = true, default-features = false }
candle-nn = { version = "0.11", optional = true, default-features = false }

[target.'cfg(target_os = "macos")'.dependencies]
candle-core = { version = "0.11", optional = true, features = ["metal"] }

[[bin]]
name = "train_ml15"
required-features = ["ml"]

[[bin]]
name = "eval_ml15"
required-features = ["ml"]
```
Verify this dependency-table merge (base + macOS-target block for the same crate) resolves
cleanly with `cargo tree --features ml` as the very first step — this specific pattern hasn't
been used elsewhere in this Cargo.toml yet.

**Starting hyperparameters** (PoC defaults, all tunable, none load-bearing for correctness):
value/policy nets: 4 fully-connected layers, hidden width 512, ReLU, no residual connections.
BWAS: node budget 100,000, weight ≈ 2.0. DAVI: batch size 8,192, target-net hard sync every 1,000
steps. Alternation: ~10–20 rounds × (5,000 solver steps, 500 generator steps) as a starting scale,
adjusted once real wall-clock on this machine is known. Eval/checkpoint reports: plain text/TSV
(no new `serde` dependency needed for something this small).

## Build order (each step independently verifiable before the next)

1. **Dependency scaffolding only.** Add the `Cargo.toml` diff. A one-line `ml_probe` binary that
   calls `Device::metal_if_available(0)`, prints which backend it got, and allocates a trivial
   tensor. Run it on this machine before writing any puzzle logic — cheaply validates the whole
   Metal-linking premise.
2. `encoding.rs` — pure Rust + tensor construction, no model yet. Test: `encode_one(&GOAL)` has
   exactly 16 ones at the expected positions.
3. `value_net.rs` — forward pass + a `VarMap::save`/`load` round-trip test (save, reload into a
   fresh `VarMap`, confirm identical output on a fixed input).
4. `scramble.rs` — no candle dependency at all. Test: walk-length distribution, no immediate-undo
   moves, every output `is_solvable()`.
5. `davi.rs` `train_step` — get one call to run without panicking/NaN-ing on a tiny batch, then a
   short run confirming loss trends down on `k_max ≤ 5` scrambles specifically (cheap, falsifiable
   check since near-goal optimal move-counts are easy to sanity-check by hand).
6. `bwas.rs` — validate first against a **trivial heuristic** (Manhattan) by comparing its output
   length to the existing, already-tested `idastar`'s output on shallow states (reuse
   `search::tests_util::bfs_distances` as ground truth) — only wire it to a real `ValueNet` after
   that passes.
7. `eval.rs` + `eval_ml15` — wire up antipodes/depth76-80 loading and run the (still barely
   trained) solver through it purely to validate the I/O and reporting path; expect bad scores at
   this point, that's fine.
8. `policy_net.rs` — forward pass + legal-move masking + sampling. Test: masked-illegal-move
   probabilities are exactly zero; sampled moves are always legal.
9. `generator.rs` — wire REINFORCE using the step-5 `ValueNet` as the frozen "solver" side and
   `idastar`+WD as the baseline. Validate the reward arithmetic on hand-picked boards with known
   costs before trusting the gradient step.
10. `alternate.rs` + `train_ml15` — only after 1–9 are each independently verified, wire the full
    loop. First run: tiny scale (2 rounds, a few hundred steps each) purely to confirm orchestration
    and checkpoint save/load across a round boundary don't crash.
11. **Full PoC run.** Scale to the starting hyperparameters above, run to completion, use
    `eval_ml15` against the final checkpoint to produce the antipodes-vs-DeepCubeA's-+2.8
    comparison and the random-holdout trend-over-rounds report.

## Verification

- Steps 1–9 each carry their own unit/smoke test as described above — run `cargo test --features
  ml` after each step lands.
- `tests/puzzle15_ml_smoke.rs`: a full tiny end-to-end loop (few rounds, small budgets), asserting
  it completes without NaNs/panics — this is the regression guard once the system exists.
- Final acceptance (step 11): `cargo run --release --features ml --bin eval_ml15` against the
  completed run's checkpoint, reporting (a) the held-out random-board average solution length
  trend across saved checkpoints, and (b) the 17-antipode average-excess-over-80, compared against
  DeepCubeA's own +2.8 benchmark on this identical experiment. Report both numbers plainly; success
  is "stable loop + trending-down solver + a sane-neighborhood antipode result," not a specific
  target to beat.

## Status / Results (built and run 2026-07-01)

**All 11 steps implemented and independently tested.** Module `src/puzzle15/ml/` (behind the `ml`
feature): `encoding`, `value_net`, `scramble`, `davi`, `bwas`, `policy_net`, `generator`,
`alternate`, `eval`, `checkpoint`, `device`. Binaries: `ml_probe`, `train_ml15`, `eval_ml15`.
**24 lib unit tests + 1 integration smoke test all pass** (`cargo test --features ml`). Metal
backend confirmed working on the M2 Pro. Every sub-component verified in isolation before wiring:
BWAS matches exact `idastar` optimal lengths (batch=1, weight=1, Manhattan); DAVI loss trends down
on shallow scrambles; value-net save/load round-trips; policy masks illegal + banned moves; the
value net is trained purely on raw one-hot state (no admissible heuristic in the solver, as
designed); the generator's regret reward is computed against exact `idastar`+WD.

**Mechanism validated; solver under-trained (a capacity limitation, not a bug).** A right-sized
PoC run (8 rounds × 100 solver steps × batch 1024, hidden 256) showed:
- **Loop is stable** across all 8 rounds — no crash, NaN, or generator collapse. Checkpoints +
  `metrics.tsv` written to the (gitignored) `data/ml15/`.
- **Generator works**: consistently large regret (gen_reward +60…+97 every round) — it reliably
  constructs boards the current solver fails on.
- **Solver learns but plateaus shallow**: DAVI loss drops 0.033→0.003 by round 1; on the fixed
  depth-35 holdout it solves the low-effective-depth boards (mean solved length ≈8) but plateaus at
  ~57% fail (24→26/60 over training). A final eval at a 12.5× larger search budget (500k nodes)
  reached only 49/100 holdout and **0/17 antipodes** — confirming the value net is under-*trained*
  (accurate near the goal, inaccurate at depth), not under-*searched*.

**Interpretation.** This is the expected outcome at ~800 training steps versus DeepCubeA's ~10⁶: the
full pipeline and adversarial dynamics are demonstrably correct, but reaching DeepCubeA's +2.8-over-
optimal on the depth-80 antipodes needs orders of magnitude more solver training than is feasible in
one session on a laptop Metal backend. The generator's *persistently high* reward (never declining)
is itself the tell — the solver never got strong enough to close the adversarial curriculum, which
would require a much longer run. Compute (solver-step throughput), not correctness, is the binding
constraint — matching the earlier analysis that the 24-puzzle port would face this even harder.

**To push further** (future, not in scope now): far more solver steps (the dominant cost); a larger
value net once step throughput allows; and only then would the generator's reward begin to decline
as the curriculum bites. The code is structured so these are pure hyperparameter changes to
`train_ml15`, no new components.

### Ceiling characterization of the hidden-256 net (2026-07-02)

A resumed continuation (`--resume`, +24 rounds) and a search-vs-capacity diagnostic characterized the
hidden-256 PoC's ceiling:
- **More training helped**: fixed 60-board holdout climbed **26 → 33** solved (progress is lumpy and
  target-sync-gated — a jump coincided with the 10-round `target_sync_every`).
- **More search budget helped too**: same checkpoint on the same holdout went **33 → 41** at a 12.5×
  budget (40k→500k nodes), `mean_len` 9.48→11.32 — so the mid-depth regime is *partly* search-limited.
- **But the deep regime is purely value-limited**: antipodes stayed **0/17 at every budget** — the
  hidden-256 net does not represent depth-80 cost-to-go, so no search budget reaches them.

**Conclusion:** the approach is sound and improves along both cheap axes (training, search budget),
but the hidden-256 net caps at mid-depth boards; reaching the antipodes (and a useful transfer to the
24-puzzle R goal) needs more model capacity. That motivated the scale-up below. Whether hidden-1024
clears this ceiling is the pending capacity check.

### Scale-up architecture (2026-07-02)

Upgraded the value net from a fixed 4-layer MLP to a **DeepCubeA-style residual MLP**, configurable
by width and depth: `256 → hidden` input projection, then `blocks` residual blocks
(`y = relu(x + norm(l2(relu(norm(l1(x)))))`), then `hidden → 1`. Exposed via `--hidden` / `--blocks`
on `train_ml15` and `eval_ml15` (default hidden 512, blocks 4); `DaviConfig` carries `blocks`. The
target-net sync is architecture-agnostic (copies vars by name), so no change there.

- **Normalization is a hand-rolled RmsNorm — not BatchNorm, and not candle's fused norms.** It
  normalizes per-sample, so it's identical for batch=1 (BWAS root) and batch=10⁴ (DAVI) and needs no
  train/eval mode across the target sync — BatchNorm's batch statistics would break the variable-size
  search forwards. It's hand-rolled from primitive ops because **candle 0.9 ships no Metal kernel for
  fused layer-norm *or* fused rms-norm** (both crash on Metal); the primitive-op path runs on Metal
  and, per `ml_bench`, is ~1.5× cheaper per-op than a mean-centering LayerNorm. (candle's CPU rng also
  can't be `set_seed`'d, so training tests use a stable config + a sawtooth-robust "best loss in
  second half" metric instead of seeding.)
- **Training is stable at the big config** — loss decreases, no NaN/divergence.
- Real per-step throughput, backend choice, and feasibility were only measured correctly *after* the
  generator-rollout fix — see **"Perf breakthrough"** immediately below.

### Perf breakthrough (2026-07-02) — the real bottleneck was the generator rollouts

Every earlier "training is catastrophically slow" symptom (round 0 taking hours, the "62h /
infeasible" verdict, the CPU-vs-Metal flip-flopping) traced to **one bug in our own code**, not
candle, the net size, buffer pools, or sleep: the solver's per-step batch was built by re-rolling
`n_gen = batch·gen_frac` generator boards **every step**, and each board rolls the policy net **one
move at a time** (batch-1 forwards). At batch 1024 that's ~10k tiny forwards/step ≈ **975 ms/step**
(~65% of the step) — and it lived in `alternate.rs`, *outside* `train_step`, so every value-net
profile (which timed only `train_step`) missed it. That, not the value net, is why the real per-step
was ~11× my phase-sum estimates.

**Fix** (commit on `ml-scaleup`): (1) the generator is frozen for the whole solver phase, so roll its
boards **once per round into a pool** (`batch·8`) and sample the per-step generator boards from it;
(2) `Generator::sample_pool` rolls the whole pool in **lockstep** — one batched policy forward per
move-step instead of `n·k` tiny forwards. Pool gen dropped from ~15 s to **729 ms (CPU) / 387 ms
(Metal)**.

**Corrected numbers** (hidden 1024, batch 1024, real per-step via windowed timing, *not* phase sums):

| | blocks 2 | blocks 4 |
|---|--:|--:|
| **Metal** | 67 ms | **116 ms** |
| CPU | 276 ms | 537 ms |

So **Metal is ~4× faster than CPU for the value net**, and the **default backend is Metal** (`--cpu`
forces CPU). The earlier "CPU is faster" was entirely the batch-1 rollout tax, which is *worst* on
Metal (dispatch-bound); with it gone, the value net's large batched matmuls dominate, where Metal
wins (near-peak). Feasibility is now: **30k-step capacity check ≈ 1 h, 150k-step run ≈ 5 h** on Metal.

Diagnostic tooling added along the way: windowed per-step `ms/step` logging in the solver loop
(~10 windows/round) and pool-gen timing — both behind `verbose`. Lesson: **profile the actual
training loop's wall-clock, not just the instrumented sub-phases** — the phase timers hid the real
cost for a long time.
