//! Enumerate solvable boards with a fixed corner tuple (and optionally a fixed
//! interior multiset), filtered by Manhattan distance and the
//! ZPDB+WD+LC Korf-max heuristic. Emits 6-byte LE ranks to stdout.
//!
//! - Corner-only: `--corners A,B,C,D` → ~240M raw permutations.
//! - Corner + interior multiset: `--corners A,B,C,D --interior W,X,Y,Z` →
//!   ~480K raw permutations.
//!
//! Parallelism: rayon over (blank-position × first-free-tile) chunks for the
//! corner-only case. The constrained case is tiny and runs sequentially.
//!
//! ```text
//! enum_corner15 --pdb-dir data --corners 12,13,4,1 --min-md 50 --min-h 62 > out.ranks
//! ```

use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use rayon::prelude::*;

use puzzle8::puzzle15::pdb::{AdditiveZpdbHeuristic, ZPatternDb};
use puzzle8::puzzle15::rank::rank;
use puzzle8::puzzle15::search::{Heuristic, LinearConflictHeuristic, WalkingDistanceHeuristic};
use puzzle8::puzzle15::state::{State, N_CELLS, W};
use puzzle8::puzzle15::symmetry::reflect;

const CORNERS: [usize; 4] = [0, 3, 12, 15];
const INTERIOR: [usize; 4] = [5, 6, 9, 10];

#[inline]
fn pos_parity(p: usize) -> usize {
    (p / W + p % W) & 1
}

#[inline]
fn manhattan(b: &[u8; N_CELLS]) -> u32 {
    let mut total = 0u32;
    for (pos, &v) in b.iter().enumerate() {
        if v == 0 {
            continue;
        }
        let goal = (v as usize) - 1;
        total += (pos / W).abs_diff(goal / W) as u32;
        total += (pos % W).abs_diff(goal % W) as u32;
    }
    total
}

struct Args {
    pdb_dir: PathBuf,
    corners: [u8; 4],
    interior: Option<[u8; 4]>,
    min_md: u32,
    min_h: u8,
}

fn parse_tile_list<const N: usize>(s: &str, flag: &str) -> Result<[u8; N], String> {
    let parts: Vec<u8> = s
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<u8>()
                .map_err(|e| format!("{flag}: {e} in {t:?}"))
        })
        .collect::<Result<_, _>>()?;
    if parts.len() != N {
        return Err(format!("{} needs {} values, got {}", flag, N, parts.len()));
    }
    let mut seen = [false; 16];
    for &t in &parts {
        if (t as usize) >= 16 {
            return Err(format!("{flag}: tile {t} out of range 0..16"));
        }
        if seen[t as usize] {
            return Err(format!("{flag}: duplicate tile {t}"));
        }
        seen[t as usize] = true;
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&parts);
    Ok(out)
}

fn parse_args() -> Result<Args, String> {
    let mut pdb_dir = PathBuf::from("data");
    let mut corners_str: Option<String> = None;
    let mut interior_str: Option<String> = None;
    let mut min_md: u32 = 50;
    let mut min_h: u8 = 62;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--pdb-dir" => {
                i += 1;
                pdb_dir = PathBuf::from(argv.get(i).ok_or("--pdb-dir needs a value")?);
            }
            "--corners" => {
                i += 1;
                corners_str = Some(argv.get(i).ok_or("--corners needs a value")?.clone());
            }
            "--interior" => {
                i += 1;
                interior_str = Some(argv.get(i).ok_or("--interior needs a value")?.clone());
            }
            "--min-md" => {
                i += 1;
                min_md = argv
                    .get(i)
                    .ok_or("--min-md needs a value")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| format!("--min-md: {e}"))?;
            }
            "--min-h" => {
                i += 1;
                min_h = argv
                    .get(i)
                    .ok_or("--min-h needs a value")?
                    .parse()
                    .map_err(|e: std::num::ParseIntError| format!("--min-h: {e}"))?;
            }
            "-h" | "--help" => return Err("help".into()),
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }
    let corners = parse_tile_list::<4>(&corners_str.ok_or("--corners is required")?, "--corners")?;
    let interior = match interior_str {
        Some(s) => Some(parse_tile_list::<4>(&s, "--interior")?),
        None => None,
    };
    if let Some(ref it) = interior {
        for c in &corners {
            if it.contains(c) {
                return Err(format!("corner tile {c} overlaps with interior"));
            }
        }
    }
    Ok(Args {
        pdb_dir,
        corners,
        interior,
        min_md,
        min_h,
    })
}

/// Recursive Heap's algorithm.
fn heap_permute<F: FnMut(&[u8])>(a: &mut [u8], n: usize, cb: &mut F) {
    if n == 1 {
        cb(a);
        return;
    }
    for i in 0..n {
        heap_permute(a, n - 1, cb);
        if n % 2 == 0 {
            a.swap(i, n - 1);
        } else {
            a.swap(0, n - 1);
        }
    }
}

#[inline]
fn check_and_record<H: Heuristic>(
    board: &[u8; N_CELLS],
    min_md: u32,
    min_h: u8,
    h: &H,
    out: &mut Vec<u64>,
) {
    let state = State(*board);
    if !state.is_solvable() {
        return;
    }
    if manhattan(board) < min_md {
        return;
    }
    let h_direct = h.h(&state);
    let h_reflected = h.h(&reflect(&state));
    if h_direct.max(h_reflected) < min_h {
        return;
    }
    out.push(rank(&state));
}

/// Corner-only enumeration. Blank can be at any parity-0 cell.
/// Parallelism: outer over (blank_pos × first-free-tile) chunks.
fn enumerate_corner_only<H: Heuristic + Sync>(
    corners: [u8; 4],
    min_md: u32,
    min_h: u8,
    h: &H,
) -> Vec<u64> {
    let corner_set: u32 = corners.iter().fold(0u32, |acc, &t| acc | (1 << t));
    let blank_in_corner = corners.contains(&0);

    // Valid blank positions: parity-0 cells not at a corner. If 0 is one of the
    // corner tiles, blank is fixed at its corner and we skip blank position
    // iteration entirely.
    let blank_positions: Vec<usize> = if blank_in_corner {
        Vec::new() // blank fixed at corner; only one config in outer loop
    } else {
        (0..N_CELLS)
            .filter(|p| pos_parity(*p) == 0 && !CORNERS.contains(p))
            .collect()
    };

    let free_tiles_all: Vec<u8> = (0..16u8).filter(|t| (corner_set >> t) & 1 == 0).collect();

    if blank_in_corner {
        // Blank is at the corner already; just permute the remaining 12 tiles
        // over the 12 non-corner cells.
        let other_cells: Vec<usize> = (0..N_CELLS).filter(|p| !CORNERS.contains(p)).collect();
        // Parallelize over first-free-tile to give 12 chunks of 11!.
        return (0..free_tiles_all.len())
            .into_par_iter()
            .flat_map(|first_idx| {
                let mut tiles = free_tiles_all.clone();
                tiles.swap(0, first_idx);
                let first_tile = tiles[0];
                let mut rest = tiles[1..].to_vec();
                let n = rest.len();
                let mut out = Vec::new();
                heap_permute(&mut rest, n, &mut |perm| {
                    let mut board = [0u8; N_CELLS];
                    for (i, &p) in CORNERS.iter().enumerate() {
                        board[p] = corners[i];
                    }
                    board[other_cells[0]] = first_tile;
                    for (i, &p) in other_cells[1..].iter().enumerate() {
                        board[p] = perm[i];
                    }
                    check_and_record(&board, min_md, min_h, h, &mut out);
                });
                out
            })
            .collect();
    }

    // Blank not at corner: 6 blank positions × 11! (one tile in free pool is blank).
    let non_zero_tiles: Vec<u8> = free_tiles_all.iter().copied().filter(|&t| t != 0).collect();
    let chunks: Vec<(usize, usize)> = blank_positions
        .iter()
        .flat_map(|&bp| (0..non_zero_tiles.len()).map(move |i| (bp, i)))
        .collect();

    chunks
        .par_iter()
        .flat_map(|&(blank_pos, first_idx)| {
            let other_cells: Vec<usize> = (0..N_CELLS)
                .filter(|p| !CORNERS.contains(p) && *p != blank_pos)
                .collect();
            let mut tiles = non_zero_tiles.clone();
            tiles.swap(0, first_idx);
            let first_tile = tiles[0];
            let mut rest = tiles[1..].to_vec();
            let n = rest.len();
            let mut out = Vec::new();
            heap_permute(&mut rest, n, &mut |perm| {
                let mut board = [0u8; N_CELLS];
                for (i, &p) in CORNERS.iter().enumerate() {
                    board[p] = corners[i];
                }
                board[blank_pos] = 0;
                board[other_cells[0]] = first_tile;
                for (i, &p) in other_cells[1..].iter().enumerate() {
                    board[p] = perm[i];
                }
                check_and_record(&board, min_md, min_h, h, &mut out);
            });
            out
        })
        .collect()
}

/// Corner + interior-multiset enumeration. Smaller search; runs serially.
fn enumerate_corner_interior<H: Heuristic + Sync>(
    corners: [u8; 4],
    interior: [u8; 4],
    min_md: u32,
    min_h: u8,
    h: &H,
) -> Vec<u64> {
    let fixed_tiles: u32 = corners
        .iter()
        .chain(interior.iter())
        .fold(0u32, |acc, &t| acc | (1 << t));
    let frame_noncorner: Vec<usize> = (0..N_CELLS)
        .filter(|p| !CORNERS.contains(p) && !INTERIOR.contains(p))
        .collect();
    debug_assert_eq!(frame_noncorner.len(), 8);

    let blank_in_interior = interior.contains(&0);
    let parity_0_interior: Vec<usize> = INTERIOR
        .iter()
        .copied()
        .filter(|p| pos_parity(*p) == 0)
        .collect();
    let parity_0_frame: Vec<usize> = frame_noncorner
        .iter()
        .copied()
        .filter(|p| pos_parity(*p) == 0)
        .collect();

    let mut out = Vec::new();
    if blank_in_interior {
        let non_zero_interior: Vec<u8> = interior.iter().copied().filter(|&t| t != 0).collect();
        let frame_tiles: Vec<u8> = (0..16u8).filter(|t| (fixed_tiles >> t) & 1 == 0).collect();
        for &blank_pos in &parity_0_interior {
            let other_interior: Vec<usize> = INTERIOR
                .iter()
                .copied()
                .filter(|&p| p != blank_pos)
                .collect();
            let mut int_buf: Vec<u8> = non_zero_interior.clone();
            let n_int = int_buf.len();
            heap_permute(&mut int_buf, n_int, &mut |int_perm| {
                let mut frame_buf = frame_tiles.clone();
                let n_frame = frame_buf.len();
                heap_permute(&mut frame_buf, n_frame, &mut |fperm| {
                    let mut board = [0u8; N_CELLS];
                    for (i, &p) in CORNERS.iter().enumerate() {
                        board[p] = corners[i];
                    }
                    board[blank_pos] = 0;
                    for (i, &p) in other_interior.iter().enumerate() {
                        board[p] = int_perm[i];
                    }
                    for (i, &p) in frame_noncorner.iter().enumerate() {
                        board[p] = fperm[i];
                    }
                    check_and_record(&board, min_md, min_h, h, &mut out);
                });
            });
        }
    } else {
        let frame_tiles_with_blank: Vec<u8> =
            (0..16u8).filter(|t| (fixed_tiles >> t) & 1 == 0).collect();
        let non_blank_frame: Vec<u8> = frame_tiles_with_blank
            .iter()
            .copied()
            .filter(|&t| t != 0)
            .collect();
        for &blank_pos in &parity_0_frame {
            let other_frame: Vec<usize> = frame_noncorner
                .iter()
                .copied()
                .filter(|&p| p != blank_pos)
                .collect();
            let mut int_buf = interior.to_vec();
            let n_int = int_buf.len();
            heap_permute(&mut int_buf, n_int, &mut |int_perm| {
                let mut frame_buf = non_blank_frame.clone();
                let n_frame = frame_buf.len();
                heap_permute(&mut frame_buf, n_frame, &mut |fperm| {
                    let mut board = [0u8; N_CELLS];
                    for (i, &p) in CORNERS.iter().enumerate() {
                        board[p] = corners[i];
                    }
                    for (i, &p) in INTERIOR.iter().enumerate() {
                        board[p] = int_perm[i];
                    }
                    board[blank_pos] = 0;
                    for (i, &p) in other_frame.iter().enumerate() {
                        board[p] = fperm[i];
                    }
                    check_and_record(&board, min_md, min_h, h, &mut out);
                });
            });
        }
    }
    out
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let p7 = ZPatternDb::load_mmap(&args.pdb_dir.join("zpdb15_p7.zbin"))
        .map_err(|e| format!("zpdb15_p7: {e}"))?;
    let p8 = ZPatternDb::load_mmap(&args.pdb_dir.join("zpdb15_p8.zbin"))
        .map_err(|e| format!("zpdb15_p8: {e}"))?;
    let zdbs = [p7, p8];
    let h_zpdb = AdditiveZpdbHeuristic::new(&zdbs);
    WalkingDistanceHeuristic::warm_up();
    let h_wd = WalkingDistanceHeuristic;
    let h_lc = LinearConflictHeuristic;

    struct CombinedH<'a, A, B, C>(&'a A, &'a B, &'a C);
    impl<A: Heuristic, B: Heuristic, C: Heuristic> Heuristic for CombinedH<'_, A, B, C> {
        #[inline]
        fn h(&self, s: &State) -> u8 {
            self.0.h(s).max(self.1.h(s)).max(self.2.h(s))
        }
    }
    let h = CombinedH(&h_zpdb, &h_wd, &h_lc);

    eprintln!(
        "enum_corner15: corners={:?}, interior={:?}, min_md={}, min_h={}",
        args.corners, args.interior, args.min_md, args.min_h
    );

    let mut ranks = if let Some(interior) = args.interior {
        enumerate_corner_interior(args.corners, interior, args.min_md, args.min_h, &h)
    } else {
        enumerate_corner_only(args.corners, args.min_md, args.min_h, &h)
    };
    ranks.sort_unstable();
    ranks.dedup();
    eprintln!("kept {} candidates", ranks.len());

    let stdout = io::stdout();
    let mut w = BufWriter::new(stdout.lock());
    for r in &ranks {
        w.write_all(&r.to_le_bytes()[..6])
            .map_err(|e| format!("stdout: {e}"))?;
    }
    w.flush().map_err(|e| format!("flush: {e}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e == "help" => {
            eprintln!("usage: enum_corner15 --pdb-dir DIR --corners A,B,C,D [--interior W,X,Y,Z] [--min-md N] [--min-h N]");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
