//! Diff Mask vs Core on a file, printing byte positions of the first
//! mismatch. Debug aid for the mask-scanner differential tests.
//! Run: cargo run -p pretokie --example mask_debug -- <file> <config>

use pretokie::{Core, Mask};


fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let cfg = args.get(2).map(String::as_str).unwrap_or("o200k");

    macro_rules! run {
        ($c:ty) => {{
            let mut core = Core::<$c>::new(&text);
            let mut mask = Mask::<$c>::new(&text);
            let mut pos_core = 0usize;
            let mut pos_mask = 0usize;
            let mut i = 0usize;
            loop {
                let a = core.next();
                let b = mask.next();
                match (a, b) {
                    (Some(a), Some(b)) => {
                        if a != b {
                            println!(
                                "piece {i}: core {:?} @ {} | mask {:?} @ {} (byte%64: core {} mask {})",
                                a, pos_core, b, pos_mask, pos_core % 64, pos_mask % 64
                            );
                            return;
                        }
                        pos_core += a.len();
                        pos_mask += b.len();
                    }
                    (None, None) => {
                        println!("identical: {} pieces", i);
                        return;
                    }
                    (a, b) => {
                        println!("piece {i}: core {:?} mask {:?}", a, b);
                        return;
                    }
                }
                i += 1;
            }
        }};
    }
    match cfg {
        "gpt2" => run!(pretokie::Gpt2Config),
        "cl100k" => run!(pretokie::Cl100kConfig),
        "o200k" => run!(pretokie::O200kConfig),
        "voyage" => run!(pretokie::VoyageConfig),
        "smollm" => run!(pretokie::SmolLMConfig),
        "deepseek" => run!(pretokie::DeepSeekConfig),
        "qwen" => run!(pretokie::QwenConfig),
        _ => panic!("unknown config"),
    }
}
