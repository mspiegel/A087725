//! Print an optimal solution for the shallowest 8-puzzle state where tile 1 is
//! the lone correctly-placed tile and every optimal solution must move it.
//!
//! Uses the full distance table as a perfect heuristic to guarantee a single
//! optimal path (and to follow neighbour-distances for an alternative choice
//! count at each step, so we can see whether the path choice was forced).

use std::path::Path;

use puzzle8::puzzle8::{io, Move, State, GOAL};

fn render(s: &State) -> String {
    let mut out = String::new();
    for row in 0..3 {
        out.push_str("    ");
        for col in 0..3 {
            let v = s.0[row * 3 + col];
            if v == 0 {
                out.push_str("  _");
            } else {
                out.push_str(&format!(" {v:>2}"));
            }
        }
        out.push('\n');
    }
    out
}

fn move_name(m: Move) -> &'static str {
    match m {
        Move::Up => "Up",
        Move::Down => "Down",
        Move::Left => "Left",
        Move::Right => "Right",
    }
}

fn main() {
    let path = Path::new("data/dist8.bin");
    let table = io::load(path).expect("load data/dist8.bin");

    // The shallowest c=1, tile-1-forced state at distance 13.
    let start = State([1, 0, 5, 6, 2, 3, 4, 7, 8]);
    assert_eq!(table.dist(&start), 13);

    let mut cur = start;
    let mut step = 0;
    println!("step  move    blank-from  tile-moved  alt-optimal-moves");
    println!("----  ------  ----------  ----------  -----------------");
    println!("start dist={}", table.dist(&cur));
    print!("{}", render(&cur));

    while cur != GOAL {
        let d = table.dist(&cur);
        let mut chosen: Option<Move> = None;
        let mut optimal_moves: Vec<Move> = Vec::new();
        for m in cur.legal_moves().iter() {
            let nxt = cur.apply(m);
            if table.dist(&nxt) == d - 1 {
                optimal_moves.push(m);
                if chosen.is_none() {
                    chosen = Some(m);
                }
            }
        }
        let m = chosen.expect("non-goal state has at least one optimal successor");
        let b = cur.blank_pos();
        let nxt = cur.apply(m);
        let moved_tile = cur.0[nxt.blank_pos() as usize];
        step += 1;
        let alts: Vec<&str> = optimal_moves.iter().map(|&x| move_name(x)).collect();
        println!(
            "{:>4}  {:<6}  cell {:>2}     tile {:>2}     [{}]",
            step,
            move_name(m),
            b,
            moved_tile,
            alts.join(", ")
        );
        cur = nxt;
        print!("{}", render(&cur));
        println!("    dist={}", table.dist(&cur));
    }
    println!();
    println!("Solved in {step} moves.");
}
