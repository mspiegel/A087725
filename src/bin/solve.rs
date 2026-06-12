//! Read an 8-puzzle position from the command line and print an optimal
//! solution.
//!
//! Usage:
//!     cargo run --release --bin solve -- "8 6 7 / 2 5 4 / 3 _ 1"
//!     cargo run --release --bin solve -- "867254301"
//!
//! The position may use spaces, slashes, dots, or underscores as separators;
//! the blank may be written as `_`, `.`, or `0`. Tile values are `1..=8`.

use puzzle8::io::load;
use puzzle8::search::{idastar, ManhattanHeuristic};
use puzzle8::state::{Move, State};

const TABLE_PATH: &str = "data/dist8.bin";

fn parse_state(input: &str) -> Result<State, String> {
    let mut tiles = Vec::with_capacity(9);
    for c in input.chars() {
        let v = match c {
            '0' | '_' | '.' => 0,
            '1'..='8' => c.to_digit(10).unwrap() as u8,
            ' ' | '/' | '|' | '\t' | '\n' | ',' => continue,
            _ => return Err(format!("unexpected character: {:?}", c)),
        };
        tiles.push(v);
    }
    if tiles.len() != 9 {
        return Err(format!("expected 9 tile values, got {}", tiles.len()));
    }
    let mut seen = [false; 9];
    for &t in &tiles {
        if seen[t as usize] {
            return Err(format!("tile {} appears more than once", t));
        }
        seen[t as usize] = true;
    }
    let mut arr = [0u8; 9];
    arr.copy_from_slice(&tiles);
    let s = State(arr);
    if !s.is_solvable() {
        return Err(format!("state is unsolvable: {:?}", arr));
    }
    Ok(s)
}

fn move_char(m: Move) -> char {
    match m {
        Move::Up => 'U',
        Move::Down => 'D',
        Move::Left => 'L',
        Move::Right => 'R',
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: solve <position>");
        eprintln!("Example: solve \"8 6 7 / 2 5 4 / 3 _ 1\"");
        std::process::exit(2);
    }
    let input = args.join(" ");

    let s = match parse_state(&input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("parse error: {}", e);
            std::process::exit(2);
        }
    };

    // Prefer the persistent table heuristic when available; fall back to
    // Manhattan otherwise.
    let table = load(std::path::Path::new(TABLE_PATH)).ok();
    let solution = if let Some(ref t) = table {
        let h = puzzle8::search::TableHeuristic::new(t);
        idastar(&s, &h).expect("solvable")
    } else {
        idastar(&s, &ManhattanHeuristic).expect("solvable")
    };

    if let Some(t) = &table {
        let expected = t.dist(&s);
        assert_eq!(
            solution.len() as u8,
            expected,
            "solution length does not match table distance"
        );
        eprintln!("optimal length: {}", expected);
    } else {
        eprintln!("optimal length: {} (table not loaded; verifying via IDA* admissibility)", solution.len());
    }

    let s_chars: String = solution.iter().map(|&m| move_char(m)).collect();
    println!("{}", s_chars);

    // Sanity replay.
    let mut cur = s;
    for &m in &solution {
        cur = cur.apply(m);
    }
    debug_assert_eq!(cur, puzzle8::state::GOAL);

    // Bind `table` so it stays alive for the heuristic borrow above.
    drop(table);
}
