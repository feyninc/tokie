//! Every loaded tokenizer matches its spec on a complete set of short strings.
//!
//! The spec is tokenizers v1, specials off and on. When v1 cannot encode the
//! file, the spec for `encode` is gigatoken. `wrap` is then unchecked,
//! because gigatoken does not apply the post-processor template.

use std::path::Path;
use std::time::Instant;

use tokie_bench::hf;
use tokie_bench::rows::{ROWS, model_path};

/// One character from each class a cut branches on.
const CLASS: &[char] = &[
    'a', 'A', 'e', 's', '1', ' ', '\n', '\r', '\t', '\'', ',', '/', '中', 'é', '\u{0301}',
];

/// Length 4 is every string over this smaller set, so a 3-digit group, a slash
/// run, and a newline run all appear.
const EDGE: &[char] = &['a', 'A', '1', ' ', '\n', '/', '\'', '中'];

const CONTRACTIONS: &[&str] = &["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

enum Spec {
    Hf(tk_encode::tokenizer::pipeline::PipelineTokenizer),
    Giga(gigatoken_rs::Tokenizer),
}

#[test]
fn every_tokenizer_matches_the_reference() {
    let corpus = corpus();
    eprintln!("corpus {}", corpus.len());
    let mut checked = Vec::new();
    let mut skipped = Vec::new();
    for row in ROWS {
        let Some(path) = model_path(row) else {
            skipped.push(format!("{}: no tokenizer.json", row.name));
            continue;
        };
        let mut model = match tokie::Tokenizer::open(&path) {
            Ok(model) => model,
            Err(err) => {
                skipped.push(format!("{}: tokie refused the file: {err}", row.name));
                continue;
            }
        };
        let Some(mut spec) = spec_of(row.name, &path) else {
            skipped.push(format!(
                "{}: tokenizers v1 and gigatoken both refused the file",
                row.name
            ));
            continue;
        };
        let mut texts = corpus.clone();
        texts.extend(added_texts(&path));
        texts.sort();
        texts.dedup();
        let started = Instant::now();
        let mut ids = Vec::new();
        let mut reference = Vec::new();
        for text in &texts {
            ids.clear();
            model.encode(text, &mut ids).unwrap_or_else(|err| {
                panic!("{} encode {text:?}: {err}", row.name);
            });
            reference.clear();
            spec_encode(&mut spec, text, false, &mut reference);
            assert_eq!(
                ids, reference,
                "{} encode {text:?}",
                row.name
            );
            if matches!(spec, Spec::Hf(_)) {
                ids.clear();
                model.wrap(text, &mut ids).unwrap_or_else(|err| {
                    panic!("{} wrap {text:?}: {err}", row.name);
                });
                reference.clear();
                spec_encode(&mut spec, text, true, &mut reference);
                assert_eq!(
                    ids, reference,
                    "{} wrap {text:?}",
                    row.name
                );
            }
        }
        let backend = match &spec {
            Spec::Hf(_) => "tokenizers v1",
            Spec::Giga(_) => "gigatoken",
        };
        eprintln!(
            "{}: {} strings via {backend} in {:.1}s",
            row.name,
            texts.len(),
            started.elapsed().as_secs_f64()
        );
        checked.push(row.name);
    }
    assert_eq!(
        checked,
        [
            "gpt2",
            "cl100k",
            "o200k",
            "qwen2",
            "qwen",
            "olmo",
            "nemotron",
            "kimi",
            "deepseek",
            "bert-uncased",
            "t5",
            "albert",
            "llama",
        ]
    );
    assert_eq!(
        skipped,
        [
            "smollm: tokenizers v1 and gigatoken both refused the file".to_string(),
            "bert-cased: no tokenizer.json".to_string(),
        ]
    );
    eprintln!("checked: {}", checked.join(", "));
    if !skipped.is_empty() {
        eprintln!("skipped:\n{}", skipped.join("\n"));
    }
}

fn corpus() -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for len in 0..=3 {
        cur.clear();
        push_len(len, CLASS, &mut cur, &mut out);
    }
    cur.clear();
    push_len(4, EDGE, &mut cur, &mut out);
    let mut side = vec![String::new()];
    for &ch in CLASS {
        side.push(ch.to_string());
    }
    for left in &side {
        for contraction in CONTRACTIONS {
            for right in &side {
                let mut text = String::with_capacity(left.len() + contraction.len() + right.len());
                text.push_str(left);
                text.push_str(contraction);
                text.push_str(right);
                out.push(text);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn push_len(len: usize, alphabet: &[char], cur: &mut Vec<char>, out: &mut Vec<String>) {
    if cur.len() == len {
        out.push(cur.iter().collect());
        return;
    }
    for &ch in alphabet {
        cur.push(ch);
        push_len(len, alphabet, cur, out);
        cur.pop();
    }
}

/// The added token alone, and with one class character on either side.
fn added_texts(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path).expect("tokenizer.json");
    let value: serde_json::Value = serde_json::from_str(&text).expect("tokenizer.json parses");
    let Some(tokens) = value.get("added_tokens").and_then(|token| token.as_array()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for token in tokens {
        let Some(content) = token.get("content").and_then(|content| content.as_str()) else {
            continue;
        };
        if content.is_empty() {
            continue;
        }
        out.push(content.to_string());
        for &ch in CLASS {
            let mut before = String::new();
            before.push(ch);
            before.push_str(content);
            out.push(before);
            let mut after = String::new();
            after.push_str(content);
            after.push(ch);
            out.push(after);
        }
    }
    out
}

fn spec_of(name: &str, path: &Path) -> Option<Spec> {
    if let Ok(tokenizer) = hf::load(path)
        && hf::ids_of(&tokenizer, "a").is_ok()
        && hf::ids_with_specials(&tokenizer, "a").is_ok()
    {
        return Some(Spec::Hf(tokenizer));
    }
    load_giga(name, path).map(Spec::Giga)
}

fn spec_encode(spec: &mut Spec, text: &str, specials: bool, out: &mut Vec<u32>) {
    match spec {
        Spec::Hf(tokenizer) => {
            let ids = if specials {
                hf::ids_with_specials(tokenizer, text)
            } else {
                hf::ids_of(tokenizer, text)
            }
            .unwrap_or_else(|err| panic!("tokenizers v1 on {text:?}: {err}"));
            out.extend_from_slice(&ids);
        }
        Spec::Giga(tokenizer) => {
            assert!(!specials, "gigatoken does not apply the post-processor");
            let mut ids = Vec::new();
            tokenizer.encode_with_added_tokens_flat(text.as_bytes(), &mut ids);
            out.extend_from_slice(&ids);
        }
    }
}

fn load_giga(name: &str, path: &Path) -> Option<gigatoken_rs::Tokenizer> {
    if let Ok(tokenizer) = gigatoken_rs::load_tokenizer::hf::load_hf_bpe(path) {
        return Some(tokenizer);
    }
    let ranks = tokie_bench::rows::tiktoken_path(name)?;
    let text = std::fs::read_to_string(path).expect("tokenizer.json");
    let value: serde_json::Value = serde_json::from_str(&text).expect("tokenizer.json parses");
    let specials = value
        .get("added_tokens")
        .and_then(|token| token.as_array())?
        .iter()
        .filter_map(|token| {
            let content = token.get("content")?.as_str()?.to_string();
            let id = u32::try_from(token.get("id")?.as_u64()?).ok()?;
            if content.is_empty() {
                None
            } else {
                Some((content, id))
            }
        })
        .collect();
    gigatoken_rs::load_tokenizer::tiktoken::load_tiktoken(
        ranks,
        gigatoken_rs::pretokenize::PretokenizerType::Kimi,
        specials,
    )
    .ok()
}
