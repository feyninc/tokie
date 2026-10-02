//! X1 research harness: Backtracking miss path and long-piece cache.
//!
//! cargo run --release --features hf --example x1_miss -- types
//! cargo run --release --features hf --example x1_miss -- equiv [filter]
//! cargo run --release --features hf --example x1_miss -- miss [filter]
//! cargo run --release --features hf --example x1_miss -- bench [filter]
//! cargo run --release --features hf --example x1_miss -- fp [filter]
//! cargo run --release --features hf,build --example x1_miss -- vshf [filter]

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;
use tokie::Tokenizer;

const TOKBENCH: &str = "C:/Users/lain/Desktop/github/tokbench/data";
const FIXTURES: &[&str] = &["english", "chinese", "japanese", "agentic-swe"];

const MODELS: &[(&str, &str)] = &[
    ("gpt2", "tokiers/gpt2"),
    ("cl100k", "tokiers/cl100k"),
    ("o200k", "tokiers/o200k"),
    ("deepseek-v3", "tokiers/DeepSeek-V3"),
    ("qwen3", "tokiers/Qwen3-0.6B"),
    ("llama-3", "tokiers/Llama-3.2-1B"),
    ("bert", "tokiers/bert-base-uncased"),
];

fn load(filter: Option<&str>) -> Vec<(&'static str, Tokenizer)> {
    let mut v = Vec::new();
    for &(name, repo) in MODELS {
        if filter.is_some_and(|f| !f.split(',').any(|f| name.contains(f))) {
            continue;
        }
        match Tokenizer::from_pretrained(repo) {
            Ok(t) => v.push((name, t)),
            Err(e) => eprintln!("  {name} ({repo}): {e}"),
        }
    }
    v
}

fn enwik8(max: usize) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benches/data/enwik8");
    let d = std::fs::read(p).expect("enwik8");
    String::from_utf8_lossy(&d[..max.min(d.len())]).into_owned()
}

/// Text of a tokbench fixture.
fn fixture(name: &str) -> String {
    let p = Path::new(TOKBENCH).join("fixtures").join(format!("{name}.txt"));
    let b = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    String::from_utf8_lossy(&b).into_owned()
}

/// ~10 KiB chunks cut at char boundaries.
fn chunks(text: &str, size: usize) -> Vec<&str> {
    let mut v = Vec::new();
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let mut j = (i + size).min(b.len());
        while j < b.len() && !text.is_char_boundary(j) {
            j += 1;
        }
        v.push(&text[i..j]);
        i = j;
    }
    v
}

fn fnv(ids: &[u32], mut h: u64) -> u64 {
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn types(toks: &[(&str, Tokenizer)]) {
    for (name, t) in toks {
        let e = t.encoder();
        let extra = match e.as_backtracking() {
            Some(b) => format!("rank_merge={}", b.has_rank_merge()),
            None => String::new(),
        };
        println!("{name:12} {:?} {extra}", e.encoder_type());
    }
}

/// Every unique pretoken piece: DAAC walk == rank merge (when available).
fn equiv(toks: &[(&str, Tokenizer)]) {
    let mut corpora = vec![enwik8(4_000_000)];
    for f in FIXTURES {
        corpora.push(fixture(f));
    }
    for (name, t) in toks {
        let Some(enc) = t.encoder().as_backtracking() else { continue };
        if !enc.has_rank_merge() || !enc.has_merge_core() {
            println!("{name}: rank table {} merge core {}", enc.has_rank_merge(), enc.has_merge_core());
            continue;
        }
        let pretok = t.pretokenizer().expect("pretok");
        let mut seen: HashSet<&[u8]> = HashSet::new();
        let (mut n, mut bad, mut bad_core) = (0usize, 0usize, 0usize);
        for c in &corpora {
            for p in pretok.split(c) {
                let b = p.as_bytes();
                if b.is_empty() || !seen.insert(b) {
                    continue;
                }
                let mut a = Vec::new();
                enc.encode_sequential_into(b, &mut a);
                let mut r = Vec::new();
                enc.encode_rank_merge(b, &mut r);
                let mut m = Vec::new();
                enc.encode_merge_core(b, &mut m);
                if a != m {
                    bad_core += 1;
                    if bad_core <= 5 {
                        println!("  {name} CORE DIFF {:?} daac={a:?} core={m:?}", String::from_utf8_lossy(b));
                    }
                }
                n += 1;
                if a != r {
                    bad += 1;
                    if bad <= 5 {
                        println!("  {name} DIFF {:?} daac={a:?} rank={r:?}", String::from_utf8_lossy(b));
                    }
                }
            }
        }
        println!("{name:12} unique pieces {n}, rank mismatches {bad}, core mismatches {bad_core}");
    }
}

/// Per-piece miss-path cost (ns/piece) on unique multi-token pieces,
/// bucketed by length.
fn miss(toks: &[(&str, Tokenizer)]) {
    // X1_CORPUS=chinese,japanese restricts the text to those fixtures.
    let text = match std::env::var("X1_CORPUS") {
        Ok(c) => c.split(',').map(fixture).collect::<String>(),
        Err(_) => enwik8(2_000_000) + &fixture("agentic-swe") + &fixture("chinese"),
    };
    const BUCKETS: &[(usize, usize)] = &[(1, 15), (16, 32), (33, 64), (65, 128), (129, usize::MAX)];
    for (name, t) in toks {
        let Some(enc) = t.encoder().as_backtracking() else { continue };
        let pretok = t.pretokenizer().expect("pretok");
        let mut seen: HashSet<&[u8]> = HashSet::new();
        let mut all: Vec<&[u8]> = Vec::new();
        for p in pretok.split(&text) {
            let b = p.as_bytes();
            if !b.is_empty() && enc.token_cache_get(b).is_none() && seen.insert(b) {
                all.push(b);
            }
        }
        println!("{name} (rank table: {}, merge core: {})", enc.has_rank_merge(), enc.has_merge_core());
        for &(lo, hi) in BUCKETS {
            let pieces: Vec<&[u8]> = all.iter().copied().filter(|p| (lo..=hi).contains(&p.len())).collect();
            if pieces.is_empty() {
                continue;
            }
            let mut out = Vec::with_capacity(1 << 20);
            let mut time = |f: &dyn Fn(&[u8], &mut Vec<u32>)| {
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for p in &pieces {
                        out.clear();
                        f(p, &mut out);
                    }
                    best = best.min(t0.elapsed().as_secs_f64());
                }
                best * 1e9 / pieces.len() as f64
            };
            let dflt0 = time(&|p, o| enc.encode_into(p, None, o));
            let daac = time(&|p, o| enc.encode_backtrack_into(p, o));
            let rank = enc.has_rank_merge().then(|| time(&|p, o| enc.encode_rank_merge(p, o)));
            let core = enc.has_merge_core().then(|| time(&|p, o| enc.encode_merge_core(p, o)));
            let dflt = time(&|p, o| enc.encode_into(p, None, o)).min(dflt0);
            println!(
                "  len {lo:>3}..{:<5} n={:<7} daac {daac:>6.0} ns  rank {:>8}  core {:>8}  encode_into {dflt:>6.0} ns",
                if hi == usize::MAX { "inf".to_string() } else { hi.to_string() },
                pieces.len(),
                rank.map(|r| format!("{r:.0} ns")).unwrap_or("-".into()),
                core.map(|r| format!("{r:.0} ns")).unwrap_or("-".into()),
            );
        }
    }
}

/// MB/s on 10 KiB chunks of each fixture: cold = fresh tokenizer clone per
/// run (first pass), warm = best of repeated passes.
fn bench(toks: &[(&str, Tokenizer)]) {
    let fx: Vec<(&str, String)> = FIXTURES.iter().map(|f| (*f, fixture(f))).collect();
    print!("{:12}", "model");
    for (f, _) in &fx {
        print!(" {:>11} {:>11}", format!("{f:.7}-cold"), format!("{f:.7}-warm"));
    }
    println!();
    for (name, t) in toks {
        print!("{name:12}");
        for (_, text) in &fx {
            let cs = chunks(text, 10 * 1024);
            let nbytes: usize = cs.iter().map(|c| c.len()).sum();
            // cold: a freshly-loaded tokenizer (new cache generation)
            let repo = MODELS.iter().find(|m| m.0 == *name).unwrap().1;
            let fresh = Tokenizer::from_pretrained(repo).unwrap();
            let t0 = Instant::now();
            let mut n = 0usize;
            for c in &cs {
                n += fresh.encode(c, false).ids.len();
            }
            let cold = nbytes as f64 / 1e6 / t0.elapsed().as_secs_f64();
            let mut best = 0f64;
            for _ in 0..5 {
                let t0 = Instant::now();
                for c in &cs {
                    n += t.encode(c, false).ids.len();
                }
                best = best.max(nbytes as f64 / 1e6 / t0.elapsed().as_secs_f64());
            }
            std::hint::black_box(n);
            print!(" {cold:>11.1} {best:>11.1}");
        }
        println!();
    }
}

/// Fingerprints (FNV-1a of ids) over enwik8 1 MB and fixture chunks, single
/// string + batch; compare across builds.
fn fp(toks: &[(&str, Tokenizer)]) {
    let mut corpora = vec![("enwik8".to_string(), enwik8(1_000_000))];
    for f in FIXTURES {
        corpora.push((f.to_string(), fixture(f)));
    }
    for (name, t) in toks {
        let mut line = format!("{name:12}");
        for (cn, text) in &corpora {
            let cs = chunks(text, 10 * 1024);
            let mut h = 0xcbf29ce484222325u64;
            let mut n = 0;
            for c in &cs {
                let ids = t.encode(c, false).ids;
                n += ids.len();
                h = fnv(&ids, h);
            }
            let batch = t.encode_batch(&cs, false);
            let mut hb = 0xcbf29ce484222325u64;
            for e in &batch {
                hb = fnv(&e.ids, hb);
            }
            assert_eq!(h, hb, "{name}/{cn}: batch != sequential");
            line += &format!(" {cn}={h:016x}/{n}");
        }
        println!("{line}");
    }
}

/// Byte share per piece-length bucket, and for each bucket the share of
/// bytes in pieces seen before (an unbounded cache's hit rate).
fn lens(toks: &[(&str, Tokenizer)]) {
    const B: &[(usize, usize)] = &[(1, 15), (16, 64), (65, usize::MAX)];
    for (name, t) in toks {
        let Some(pretok) = t.pretokenizer() else { continue };
        for f in FIXTURES {
            let text = fixture(f);
            let mut seen: HashSet<&[u8]> = HashSet::new();
            let mut tot = [0usize; 3];
            let mut rep = [0usize; 3];
            for p in pretok.split(&text) {
                let b = p.as_bytes();
                let k = B.iter().position(|&(lo, hi)| (lo..=hi).contains(&b.len())).unwrap_or(0);
                tot[k] += b.len();
                if !seen.insert(b) {
                    rep[k] += b.len();
                }
            }
            let all: usize = tot.iter().sum();
            let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
            println!(
                "{name:12} {f:12} bytes%: <=15 {:5.1} (rep {:4.1})  16-64 {:5.1} (rep {:4.1})  >64 {:5.1} (rep {:4.1})",
                pct(tot[0], all), pct(rep[0], tot[0]), pct(tot[1], all), pct(rep[1], tot[1]), pct(tot[2], all), pct(rep[2], tot[2])
            );
        }
    }
}

/// Exactness vs HF tokenizers: every 10 KiB chunk of enwik8 1 MB and the
/// fixtures, id-for-id, for tokie single-string encode and batch encode.
#[cfg(feature = "build")]
fn vshf(toks: &[(&str, Tokenizer)]) {
    let mut corpora = vec![("enwik8".to_string(), enwik8(1_000_000))];
    for f in FIXTURES {
        corpora.push((f.to_string(), fixture(f)));
    }
    let api = hf_hub::api::sync::Api::new().unwrap();
    for (name, t) in toks {
        let repo = MODELS.iter().find(|m| m.0 == *name).unwrap().1;
        let json = api.model(repo.to_string()).get("tokenizer.json").unwrap();
        let hf = tokenizers::Tokenizer::from_file(&json).unwrap();
        let mut line = format!("{name:12}");
        for (cn, text) in &corpora {
            let cs = chunks(text, 10 * 1024);
            let want = hf.encode_batch(cs.clone(), false).unwrap();
            let got = t.encode_batch(&cs, false);
            let (mut bad, mut h, mut n) = (0usize, 0xcbf29ce484222325u64, 0usize);
            for (i, c) in cs.iter().enumerate() {
                let single = t.encode(c, false).ids;
                let w = want[i].get_ids();
                if single != w || got[i].ids != w {
                    bad += 1;
                    if bad <= 2 && std::env::var("X1_SHOW").is_ok() {
                        let k = single.iter().zip(w).position(|(a, b)| a != b).unwrap_or(single.len().min(w.len()));
                        let lo = k.saturating_sub(3);
                        let show = |ids: &[u32]| {
                            ids[lo..(k + 4).min(ids.len())]
                                .iter()
                                .map(|&id| format!("{:?}", t.decode(&[id]).unwrap_or_default()))
                                .collect::<Vec<_>>()
                                .join(" ")
                        };
                        println!("  {name}/{cn} chunk {i}: tokie [{}]\n{:>w$}hf    [{}]", show(&single), "", show(w), w = name.len() + cn.len() + 12);
                    }
                }
                n += single.len();
                h = fnv(&single, h);
            }
            line += &format!(" {cn}={}/{} ({h:016x}, {n} ids)", cs.len() - bad, cs.len());
        }
        println!("{line}");
    }
}

#[cfg(not(feature = "build"))]
fn vshf(_: &[(&str, Tokenizer)]) {
    eprintln!("vshf needs --features hf,build");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("types");
    let toks = load(args.get(2).map(|s| s.as_str()));
    match cmd {
        "types" => types(&toks),
        "equiv" => equiv(&toks),
        "miss" => miss(&toks),
        "bench" => bench(&toks),
        "fp" => fp(&toks),
        "lens" => lens(&toks),
        "vshf" => vshf(&toks),
        _ => eprintln!("unknown cmd"),
    }
}
