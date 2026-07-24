//! Piece-level parity: Mask iterator AND for_each_piece vs Core (ground
//! truth) on 10MB+ of OWT, for gpt2 and DeepSeek configs.
//!
//! Run: cargo run -p pretokie --release --example piece_diff

use pretokie::{Core, Gpt2, DeepSeek};
use pretokie::{Gpt2Config, DeepSeekConfig};

fn read_owt() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("benches/data/owt_sample.txt");
    let data = std::fs::read(&path).expect("owt_sample.txt missing");
    let cap = std::env::var("PIECE_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(12) * 1_000_000;
    String::from_utf8_lossy(&data[..data.len().min(cap)]).into_owned()
}

macro_rules! check {
    ($name:literal, $cfg:ty, $mask:ident, $text:expr) => {{
        let text: &str = $text;
        // iterator vs Core
        let mut core = Core::<$cfg>::new(text);
        let mut it = $mask::new(text);
        let mut i = 0usize;
        let mut diffs = 0usize;
        loop {
            match (core.next(), it.next()) {
                (Some(a), Some(b)) => { if a != b { diffs += 1; if diffs <= 3 { eprintln!("{} iter piece {i}: core {a:?} mask {b:?}", $name); } } }
                (None, None) => break,
                (a, b) => { diffs += 1; eprintln!("{} iter LEN piece {i}: core {a:?} mask {b:?}", $name); break; }
            }
            i += 1;
        }
        // for_each_piece vs Core
        let mut core2 = Core::<$cfg>::new(text);
        let mut j = 0usize;
        let mut bdiffs = 0usize;
        $mask::new(text).for_each_piece(|b| {
            let a = core2.next();
            if a != Some(b) { bdiffs += 1; if bdiffs <= 3 { eprintln!("{} bulk piece {j}: core {a:?} bulk {b:?}", $name); } }
            j += 1;
        });
        if core2.next().is_some() { bdiffs += 1; eprintln!("{} bulk short", $name); }
        println!("{:12}: {i} pieces, iter diffs {diffs}, bulk diffs {bdiffs}  [{}]",
            $name, if diffs == 0 && bdiffs == 0 { "OK" } else { "FAIL" });
        assert_eq!(diffs, 0, "{} iter mismatch", $name);
        assert_eq!(bdiffs, 0, "{} bulk mismatch", $name);
    }};
}

fn main() {
    let text = read_owt();
    println!("corpus: {:.1} MB", text.len() as f64 / 1e6);
    check!("gpt2", Gpt2Config, Gpt2, &text);
    check!("deepseek", DeepSeekConfig, DeepSeek, &text);
    println!("all piece-diff checks passed");
}
