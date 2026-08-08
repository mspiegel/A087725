//! Build the cWD table family — the five artifacts the solve24 engine mmaps.
//!
//! Each stage builds one artifact and runs its validation gate (round-trip,
//! dominance, or A\* oracle checks) before returning; a gate failure panics.
//! Stages consume each other's outputs, so a from-scratch build runs them in
//! the `all` order:
//!
//! ```text
//! mm     wd24.bin + cwd_single.bin        -> cwd_mm.bin       (~4 GiB)
//! lm     (pure compute)                   -> cwd_lm.bin       (minutes, ~3 GB peak)
//! lm2    [cwd_lm.bin for dominance gate]  -> cwd_lm2.bin      (tens of minutes, ~9 GB peak)
//! lm-mm  cwd_lm.bin + cwd_lm2.bin        -> cwd_lm_mm.bin    (~4 GiB)
//! lm1l   cwd_mm.bin                       -> cwd_lm1l_mm.bin  (hours, ~18 GB peak, ~10.5 GB)
//! ```
//!
//! The prerequisites `data/wd24.bin` and `data/cwd_single.bin` come from
//! `build_wd24` and `build_cwd_single`. Wrap long stages in `caffeinate -i`.

use std::path::Path;

use clap::{Parser, Subcommand};
use puzzle8::puzzle24::search::cwd::build_cwd_mm_artifact;
use puzzle8::puzzle24::search::cwd_lm::{
    build_cwd_lm2_table, build_cwd_lm_mm_artifact, build_cwd_lm_table,
};
use puzzle8::puzzle24::search::cwd_lm1l::build_cwd_lm1l_artifact;

/// Build the cWD table family (merged cWD, LM, LM2, and the clm2 joint
/// table), each with its validation gate.
#[derive(Parser)]
#[command(name = "build_cwd_artifacts")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// The merged cWD mmap artifact from data/wd24.bin + data/cwd_single.bin
    /// (~4 GiB written; every cell round-trip verified).
    Mm {
        #[arg(long, value_name = "PATH", default_value = "data/cwd_mm.bin")]
        out: String,
    },
    /// The Last-Move refined table (~66 M keys; minutes, ~3 GB peak).
    Lm {
        #[arg(long, value_name = "PATH", default_value = "data/cwd_lm.bin")]
        out: String,
    },
    /// The last-two-moves pair table (~6.6 B product states; tens of
    /// minutes, ~9 GB peak). Gates pair >= single dominance against --lm-bin
    /// when that table is present.
    Lm2 {
        #[arg(long, value_name = "PATH", default_value = "data/cwd_lm.bin")]
        lm_bin: String,
        #[arg(long, value_name = "PATH", default_value = "data/cwd_lm2.bin")]
        out: String,
    },
    /// The combined LM/LM2 mmap artifact from the two canonical tables
    /// (~4 GiB written; every key round-trip verified).
    LmMm {
        #[arg(long, value_name = "PATH", default_value = "data/cwd_lm.bin")]
        lm_bin: String,
        #[arg(long, value_name = "PATH", default_value = "data/cwd_lm2.bin")]
        lm2_bin: String,
        #[arg(long, value_name = "PATH", default_value = "data/cwd_lm_mm.bin")]
        out: String,
    },
    /// The clm2 single-demanded-line joint table (hours, ~18 GB peak,
    /// ~10.5 GB artifact; sampled fields validated against the A* oracle).
    /// If the artifact already exists, skips the build and validates only.
    Lm1l {
        #[arg(long, value_name = "PATH", default_value = "data/cwd_mm.bin")]
        cwd_mm: String,
        #[arg(long, value_name = "PATH", default_value = "data/cwd_lm1l_mm.bin")]
        out: String,
    },
    /// All five stages in dependency order, at the default paths.
    All,
}

fn main() {
    match Args::parse().cmd {
        Cmd::Mm { out } => build_cwd_mm_artifact(Path::new(&out)),
        Cmd::Lm { out } => build_cwd_lm_table(Path::new(&out)),
        Cmd::Lm2 { lm_bin, out } => build_cwd_lm2_table(Path::new(&lm_bin), Path::new(&out)),
        Cmd::LmMm {
            lm_bin,
            lm2_bin,
            out,
        } => build_cwd_lm_mm_artifact(Path::new(&lm_bin), Path::new(&lm2_bin), Path::new(&out)),
        Cmd::Lm1l { cwd_mm, out } => build_cwd_lm1l_artifact(Path::new(&cwd_mm), Path::new(&out)),
        Cmd::All => {
            build_cwd_mm_artifact(Path::new("data/cwd_mm.bin"));
            build_cwd_lm_table(Path::new("data/cwd_lm.bin"));
            build_cwd_lm2_table(Path::new("data/cwd_lm.bin"), Path::new("data/cwd_lm2.bin"));
            build_cwd_lm_mm_artifact(
                Path::new("data/cwd_lm.bin"),
                Path::new("data/cwd_lm2.bin"),
                Path::new("data/cwd_lm_mm.bin"),
            );
            build_cwd_lm1l_artifact(
                Path::new("data/cwd_mm.bin"),
                Path::new("data/cwd_lm1l_mm.bin"),
            );
        }
    }
}
