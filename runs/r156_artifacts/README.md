# The r156 proof machine, as it stood when the search finished

Copied off Azure VM `r156` (Standard_F16ams_v6 at the end, F64ams_v6 for most
of the run; New Zealand North) on 2026-09-20, after threshold 154 was
exhausted. The VM was a spot instance and is not expected to outlive this
directory. `runs/ckpt156/` alongside holds the checkpoint itself; everything
here is the machinery that produced it.

## The result

```
[05:06:38.659 UTC] threshold 154 exhausted: 568,439,495,042,651 nodes in 15941.67s
Lower bound: depth >= 156
Nodes      : 609,193,630,407,023        Iterations : 6
Search time: 15997.05s                  Throughput : 38,081.62 Mnodes/s
```

Finished 2026-09-20 1:06 AM ET. The per-threshold totals are in
`../ckpt156/main.ckpt`; the last record is `154 156 568439495042651`.

The figures above are the final resumption only. The search ran from
2026-08-10 and was restarted 74 times (`logs/evictions.log`), so no single
wall-clock number covers the whole proof; the node counts are cumulative and
do, since the checkpoint restores them.

## Files

| file | what it is |
|---|---|
| `logs/proof_run.log` | every resumption's output, ending in the result above |
| `logs/evictions.log` | one line per solver start, with uptime at launch — the eviction record |
| `logs/*.log` | the August tuning campaign (E1-E9), kept because `RUNBOOK_R156.md` §8 cites it |
| `solve24` | the exact binary that ran, sha256 `3e28bd43e0625eafa1093515cb6d3b27109a63c8d0460c923b09e38cd65158a2` |
| `solver_run.sh` | the launcher, `ExecStart` of the unit below |
| `r156-solver.service` | the systemd unit; `Restart=on-failure`, so exit 0 at completion did not relaunch it |
| `flat_split_target.patch` | the one uncommitted source edit the binary carries |
| `table_sha256_verify.txt` | all ten table artifacts re-checked against the `RUNBOOK_R156.md` §5 pins on the VM, 2026-09-20 — ten OK |
| `vm_scripts/` | the table builders and tuning drivers, as they existed on the VM |

## Binary provenance

Built 2026-08-13 03:19 UTC from commit `a590305` plus `flat_split_target.patch`
(`SPLIT_TARGET` 32768 -> 262144, saved 11 seconds before the build). Local
history was rewritten afterwards: `a590305` has source identical to local
`7419107`, the trees differing only in `data/rho/pdb24_rho_{a,b,c,d}.zbin`,
which were purged and which the solver never reads. So the proof binary is
`7419107` plus that one line. `provenance_a590305.txt` is the recorded
comparison; `a590305` itself no longer exists locally, since the only ref
holding it was the fetch from the VM and that carried the purged `rho` blobs
back with it.

The VM never fetched from git — no cron, no timer, no `git` in the launcher —
so what ran is what was on disk, which is what is here.

`solver_run.sh` predates `--config` becoming a required flag (commit `98ba9ad`)
and passes no such argument. Rebuilding this binary from current source without
adding `--config large` to the script produces a solver that exits on a missing
argument.
