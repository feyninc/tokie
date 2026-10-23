//! Find where tokie's ids diverge from HF tokenizers, line by line.
//!
//! cargo run --release --features build --example hf_diff -- <tokenizer.json> <corpus.txt>...
//!
//! Prints the first few diverging lines of each corpus, then one summary line
//! per corpus: `<corpus> <differing lines>/<lines>`.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json = &args[1];
    let tok = tokie::Tokenizer::from_json(json).expect("tokie load");
    let hf = tokenizers::Tokenizer::from_file(json).expect("hf load");
    let show_max: usize = std::env::var("HF_DIFF_SHOW").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    for corpus in &args[2..] {
        let text = std::fs::read_to_string(corpus).expect("corpus");
        let (mut diffs, mut lines) = (0usize, 0usize);
        for (n, line) in text.split_inclusive('\n').enumerate() {
            lines += 1;
            let a = tok.encode_ids(line, false);
            let b = hf.encode(line, false).unwrap().get_ids().to_vec();
            if a == b {
                continue;
            }
            diffs += 1;
            if diffs <= show_max {
                let i = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
                let lo = i.saturating_sub(3);
                let show = |ids: &[u32]| {
                    ids[lo.min(ids.len())..(i + 4).min(ids.len())]
                        .iter()
                        .map(|&id| format!("{:?}", hf.id_to_token(id).unwrap_or_default()))
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                println!("  line {n}: first diff at token {i}");
                println!("    tokie: {}", show(&a));
                println!("    hf:    {}", show(&b));
            }
        }
        let name = std::path::Path::new(corpus).file_stem().unwrap().to_string_lossy();
        println!("{name} {diffs}/{lines}");
    }
}
