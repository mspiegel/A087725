//! Print every board in a `.ranks` layer file as a 4×4 grid.
//!
//! Run: `cargo run --release --example show_depth_layer -- data/enum15/depth79.ranks`

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use puzzle8::puzzle15::rank::unrank;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .ok_or("usage: show_depth_layer <path/to/depth*.ranks>")?
        .into();

    let mut bytes = Vec::new();
    BufReader::new(File::open(&path)?).read_to_end(&mut bytes)?;
    if bytes.len() % 6 != 0 {
        return Err(format!(
            "{}: not a u48-packed ranks file ({} bytes)",
            path.display(),
            bytes.len()
        )
        .into());
    }
    let n = bytes.len() / 6;
    println!("{}: {} board(s)\n", path.display(), n);

    for (i, chunk) in bytes.chunks_exact(6).enumerate() {
        let mut le = [0u8; 8];
        le[..6].copy_from_slice(chunk);
        let r = u64::from_le_bytes(le);
        let s = unrank(r);
        println!("#{:>3}  rank {}", i + 1, r);
        for row in 0..4 {
            print!("  ");
            for col in 0..4 {
                let t = s.0[row * 4 + col];
                if t == 0 {
                    print!(" __");
                } else {
                    print!(" {t:>2}");
                }
            }
            println!();
        }
        println!();
    }
    Ok(())
}
