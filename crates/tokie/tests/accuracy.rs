//! Token-accuracy tests: compare tokie against HuggingFace tokenizers on enwik8.
//!
//! Run with: cargo test -p tokie --test accuracy --features hf -- --ignored
//!
//! Requires network access and benches/data/enwik8 (1MB used).

#![cfg(feature = "build")]

use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;
use tokie::Tokenizer;

/// Load first `max_bytes` of enwik8, returning valid UTF-8.
fn load_enwik8(max_bytes: usize) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benches/data/enwik8");
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    let truncated = &data[..data.len().min(max_bytes)];
    String::from_utf8_lossy(truncated).into_owned()
}

/// Compare tokie (from tokiers/) against HuggingFace on enwik8.
/// Returns (pass, first_diff_index) — None means all tokens match.
fn compare_model(tokiers_repo: &str, hf_model: &str, text: &str) -> (bool, Option<usize>) {
    let tok = Tokenizer::from_pretrained(tokiers_repo)
        .unwrap_or_else(|e| panic!("Failed to load tokie {tokiers_repo}: {e}"));
    let mut hf = HfTokenizer::from_pretrained(hf_model, None)
        .unwrap_or_else(|e| panic!("Failed to load HF {hf_model}: {e}"));
    // Disable any default truncation so we compare the full output
    let _ = hf.with_truncation(None);

    let tokie_ids = tok.encode(text, false).ids;
    let hf_enc = hf.encode(text, false)
        .unwrap_or_else(|e| panic!("HF encode failed for {hf_model}: {e}"));
    let hf_ids = hf_enc.get_ids();

    if tokie_ids.as_slice() == hf_ids {
        (true, None)
    } else {
        let diff = tokie_ids.iter().zip(hf_ids.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(tokie_ids.len().min(hf_ids.len()));
        (false, Some(diff))
    }
}

/// Load up to `max_bytes` of the OpenWebText sample and split into documents.
///
/// Web text exercises unicode the enwik8 XML dump never does (typographic
/// quotes/ellipses before newlines, No-category numerics like ¹ ❶ ½, format
/// chars like U+200B/U+00AD, O'Toole-style contractions) — exactly the
/// patterns where hand-rolled pretokenizers historically diverged from HF.
fn load_owt_docs(max_bytes: usize) -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benches/data/owt_sample.txt");
    let data = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {}: {e}\nDownload it with:\n  curl -sL \
             https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz \
             | gunzip -c | head -c 50000000 > benches/data/owt_sample.txt",
            path.display()
        )
    });
    let truncated = &data[..data.len().min(max_bytes)];
    let text = String::from_utf8_lossy(truncated);
    text.split("<|endoftext|>")
        .filter(|d| !d.is_empty())
        .map(|d| d.to_string())
        .collect()
}

/// Compare tokie against HuggingFace per document, panicking with a readable
/// report (doc index + text fragment around the first divergence).
///
/// Loads tokie from the model's tokenizer.json explicitly: `from_pretrained`
/// would silently prefer a pre-built tokiers/*.tkz when one exists, bypassing
/// the detection path these tests exist to exercise.
fn compare_model_docs(hf_model: &str, docs: &[String]) {
    let api = hf_hub::api::sync::ApiBuilder::new().build().unwrap();
    let json_path = api
        .repo(hf_hub::Repo::model(hf_model.to_string()))
        .get("tokenizer.json")
        .unwrap_or_else(|e| panic!("Failed to download tokenizer.json for {hf_model}: {e}"));
    let tok = Tokenizer::from_json(&json_path)
        .unwrap_or_else(|e| panic!("Failed to load tokie from json {hf_model}: {e}"));
    let mut hf = HfTokenizer::from_pretrained(hf_model, None)
        .unwrap_or_else(|e| panic!("Failed to load HF {hf_model}: {e}"));
    let _ = hf.with_truncation(None);

    let mut failures = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        let tokie_ids = tok.encode(doc, false).ids;
        let hf_enc = hf.encode(doc.as_str(), false)
            .unwrap_or_else(|e| panic!("HF encode failed for {hf_model} doc {i}: {e}"));
        let hf_ids = hf_enc.get_ids();
        if tokie_ids.as_slice() != hf_ids {
            let diff = tokie_ids.iter().zip(hf_ids.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(tokie_ids.len().min(hf_ids.len()));
            let byte = hf_enc.get_offsets().get(diff).map_or(0, |o| o.0);
            let lo = (0..=byte.min(doc.len())).rev().take(40).find(|&p| doc.is_char_boundary(p)).unwrap_or(0);
            let hi = (byte..=doc.len()).take(80).filter(|&p| doc.is_char_boundary(p)).last().unwrap_or(doc.len());
            failures.push(format!(
                "doc {i}: first divergence at token {diff} (byte {byte}): {:?}",
                &doc[lo..hi]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} docs mismatch HF for {hf_model}:\n{}",
        failures.len(),
        docs.len(),
        failures[..failures.len().min(5)].join("\n")
    );
}

// ============================================================================
// WordPiece models (BERT-family)
// ============================================================================

macro_rules! accuracy_test {
    ($name:ident, $tokiers:expr, $hf:expr) => {
        #[test]
        #[ignore] // Requires network + enwik8
        fn $name() {
            let text = load_enwik8(1_000_000);
            let (pass, diff) = compare_model($tokiers, $hf, &text);
            assert!(pass, "Token mismatch at index {:?}", diff);
        }
    };
}

// WordPiece
accuracy_test!(bert_base_uncased,       "tokiers/bert-base-uncased",       "google-bert/bert-base-uncased");
accuracy_test!(all_minilm_l6_v2,        "tokiers/all-MiniLM-L6-v2",       "sentence-transformers/all-MiniLM-L6-v2");
accuracy_test!(all_minilm_l12_v2,       "tokiers/all-MiniLM-L12-v2",      "sentence-transformers/all-MiniLM-L12-v2");
accuracy_test!(all_mpnet_base_v2,       "tokiers/all-mpnet-base-v2",      "sentence-transformers/all-mpnet-base-v2");
accuracy_test!(bge_base_en_v1_5,        "tokiers/bge-base-en-v1.5",       "BAAI/bge-base-en-v1.5");
accuracy_test!(bge_large_en_v1_5,       "tokiers/bge-large-en-v1.5",      "BAAI/bge-large-en-v1.5");
accuracy_test!(bge_small_en_v1_5,       "tokiers/bge-small-en-v1.5",      "BAAI/bge-small-en-v1.5");
accuracy_test!(bge_en_icl,              "tokiers/bge-en-icl",             "BAAI/bge-en-icl");
accuracy_test!(e5_base_v2,              "tokiers/e5-base-v2",             "intfloat/e5-base-v2");
accuracy_test!(e5_large_v2,             "tokiers/e5-large-v2",            "intfloat/e5-large-v2");
accuracy_test!(e5_small_v2,             "tokiers/e5-small-v2",            "intfloat/e5-small-v2");
accuracy_test!(gte_base,                "tokiers/gte-base",               "thenlper/gte-base");
accuracy_test!(gte_large,               "tokiers/gte-large",              "thenlper/gte-large");
accuracy_test!(gte_small,               "tokiers/gte-small",              "thenlper/gte-small");
accuracy_test!(gte_qwen2_7b_instruct,   "tokiers/gte-Qwen2-7B-instruct", "Alibaba-NLP/gte-Qwen2-7B-instruct");
accuracy_test!(ms_marco_minilm_l_4_v2,  "tokiers/ms-marco-MiniLM-L-4-v2","cross-encoder/ms-marco-MiniLM-L-4-v2");
accuracy_test!(ms_marco_minilm_l_6_v2,  "tokiers/ms-marco-MiniLM-L-6-v2","cross-encoder/ms-marco-MiniLM-L-6-v2");
accuracy_test!(mxbai_embed_large_v1,    "tokiers/mxbai-embed-large-v1",   "mixedbread-ai/mxbai-embed-large-v1");
accuracy_test!(mxbai_embed_2d_large_v1, "tokiers/mxbai-embed-2d-large-v1","mixedbread-ai/mxbai-embed-2d-large-v1");
accuracy_test!(mxbai_embed_xsmall_v1,   "tokiers/mxbai-embed-xsmall-v1",  "mixedbread-ai/mxbai-embed-xsmall-v1");
accuracy_test!(deepset_mxbai_embed_de,  "tokiers/deepset-mxbai-embed-de-large-v1", "mixedbread-ai/deepset-mxbai-embed-de-large-v1");
accuracy_test!(nomic_embed_text_v1,     "tokiers/nomic-embed-text-v1",    "nomic-ai/nomic-embed-text-v1");

// BPE (byte-level)
accuracy_test!(gpt2,                    "tokiers/gpt2",                   "openai-community/gpt2");
accuracy_test!(roberta_base,            "tokiers/roberta-base",           "FacebookAI/roberta-base");
accuracy_test!(phi_2,                   "tokiers/phi-2",                  "microsoft/phi-2");
accuracy_test!(phi_3_mini,              "tokiers/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-4k-instruct");
accuracy_test!(modernbert_base,         "tokiers/ModernBERT-base",        "answerdotai/ModernBERT-base");
accuracy_test!(codellama_7b,            "tokiers/CodeLlama-7b-hf",       "tokiers/CodeLlama-7b-hf");
accuracy_test!(llama_3_2_1b,            "tokiers/Llama-3.2-1B",          "tokiers/Llama-3.2-1B");
accuracy_test!(llama_4_scout,           "tokiers/Llama-4-Scout-17B-16E", "tokiers/Llama-4-Scout-17B-16E");
accuracy_test!(mistral_7b,              "tokiers/Mistral-7B-v0.1",       "mistralai/Mistral-7B-v0.1");
accuracy_test!(mistral_nemo,            "tokiers/Mistral-Nemo-Base-2407","mistralai/Mistral-Nemo-Base-2407");
accuracy_test!(mixtral_8x7b,            "tokiers/Mixtral-8x7B-v0.1",    "mistralai/Mixtral-8x7B-v0.1");
accuracy_test!(qwen2_7b,               "tokiers/Qwen2-7B",              "Qwen/Qwen2-7B");
accuracy_test!(qwen3_embed_0_6b,        "tokiers/Qwen3-Embedding-0.6B",  "Qwen/Qwen3-Embedding-0.6B");
accuracy_test!(qwen3_embed_4b,          "tokiers/Qwen3-Embedding-4B",    "Qwen/Qwen3-Embedding-4B");
accuracy_test!(qwen3_embed_8b,          "tokiers/Qwen3-Embedding-8B",    "Qwen/Qwen3-Embedding-8B");

// Jina (BPE)
accuracy_test!(jina_v2_base_en,         "tokiers/jina-embeddings-v2-base-en",   "jinaai/jina-embeddings-v2-base-en");
accuracy_test!(jina_v2_base_code,       "tokiers/jina-embeddings-v2-base-code", "jinaai/jina-embeddings-v2-base-code");
accuracy_test!(jina_v3,                 "tokiers/jina-embeddings-v3",           "jinaai/jina-embeddings-v3");
accuracy_test!(jina_v4,                 "tokiers/jina-embeddings-v4",           "jinaai/jina-embeddings-v4");

// Cohere (BPE)
accuracy_test!(cohere_english_v3,       "tokiers/Cohere-embed-english-v3.0",            "Cohere/Cohere-embed-english-v3.0");
accuracy_test!(cohere_english_light_v3, "tokiers/Cohere-embed-english-light-v3.0",      "Cohere/Cohere-embed-english-light-v3.0");
accuracy_test!(cohere_multi_v3,         "tokiers/Cohere-embed-multilingual-v3.0",       "Cohere/Cohere-embed-multilingual-v3.0");
accuracy_test!(cohere_multi_light_v3,   "tokiers/Cohere-embed-multilingual-light-v3.0", "Cohere/Cohere-embed-multilingual-light-v3.0");

// Voyage (BPE)
accuracy_test!(voyage_3,                "tokiers/voyage-3",                "voyageai/voyage-3");
accuracy_test!(voyage_3_large,          "tokiers/voyage-3-large",          "voyageai/voyage-3-large");
accuracy_test!(voyage_3_lite,           "tokiers/voyage-3-lite",           "voyageai/voyage-3-lite");
accuracy_test!(voyage_3_5,              "tokiers/voyage-3.5",              "voyageai/voyage-3.5");
accuracy_test!(voyage_3_5_lite,         "tokiers/voyage-3.5-lite",         "voyageai/voyage-3.5-lite");
accuracy_test!(voyage_code_2,           "tokiers/voyage-code-2",           "voyageai/voyage-code-2");
accuracy_test!(voyage_code_3,           "tokiers/voyage-code-3",           "voyageai/voyage-code-3");
accuracy_test!(voyage_finance_2,        "tokiers/voyage-finance-2",        "voyageai/voyage-finance-2");
accuracy_test!(voyage_law_2,            "tokiers/voyage-law-2",            "voyageai/voyage-law-2");
accuracy_test!(voyage_multilingual_2,   "tokiers/voyage-multilingual-2",   "voyageai/voyage-multilingual-2");
accuracy_test!(voyage_multimodal_3,     "tokiers/voyage-multimodal-3",     "voyageai/voyage-multimodal-3");

// SentencePiece / Unigram
accuracy_test!(t5_base,                 "tokiers/t5-base",              "google-t5/t5-base");
accuracy_test!(xlm_roberta_base,        "tokiers/xlm-roberta-base",     "FacebookAI/xlm-roberta-base");

// New models
accuracy_test!(deepseek_v3,             "tokiers/DeepSeek-V3",                      "deepseek-ai/DeepSeek-V3");
accuracy_test!(deepseek_r1,             "tokiers/DeepSeek-R1",                      "deepseek-ai/DeepSeek-R1");
accuracy_test!(gemma_2_2b,              "tokiers/gemma-2-2b",                       "tokiers/gemma-2-2b");
accuracy_test!(gemma_3_4b_it,           "tokiers/gemma-3-4b-it",                    "tokiers/gemma-3-4b-it");
accuracy_test!(bge_m3,                  "tokiers/bge-m3",                           "BAAI/bge-m3");
accuracy_test!(snowflake_arctic_v2,     "tokiers/snowflake-arctic-embed-l-v2.0",    "Snowflake/snowflake-arctic-embed-l-v2.0");
accuracy_test!(nv_embed_v2,             "tokiers/NV-Embed-v2",                      "nvidia/NV-Embed-v2");
accuracy_test!(smollm2_135m,            "tokiers/SmolLM2-135M",                     "HuggingFaceTB/SmolLM2-135M");
accuracy_test!(stablelm_2_1_6b,         "tokiers/stablelm-2-1_6b",                  "stabilityai/stablelm-2-1_6b");

// Qwen3 / Qwen3.5
accuracy_test!(qwen3_0_6b,              "tokiers/Qwen3-0.6B",                       "Qwen/Qwen3-0.6B");
accuracy_test!(qwen3_8b,                "tokiers/Qwen3-8B",                         "Qwen/Qwen3-8B");
accuracy_test!(qwen3_coder_30b,         "tokiers/Qwen3-Coder-30B-A3B-Instruct",     "Qwen/Qwen3-Coder-30B-A3B-Instruct");
accuracy_test!(qwen3_5_0_8b,            "tokiers/Qwen3.5-0.8B",                     "Qwen/Qwen3.5-0.8B");
accuracy_test!(qwen3_5_4b,              "tokiers/Qwen3.5-4B",                       "Qwen/Qwen3.5-4B");

// ============================================================================
// Added-token probe accuracy — interleaves every added token with plain text,
// exercising the per-token flags (lstrip/rstrip/normalized/single_word), the
// HF id reassignment, and the per-segment metaspace prepend semantics. These
// held 13 repos out of the v13 regeneration; keep them loud.
// ============================================================================

/// A probe string interleaving the model's added tokens with normal text,
/// mirroring scripts/regen_tokiers_v13.py.
fn added_token_probe(hf: &HfTokenizer) -> String {
    let mut added: Vec<(u32, String)> = hf
        .get_added_tokens_decoder()
        .into_iter()
        .map(|(id, tok)| (id, tok.content))
        .collect();
    added.sort();
    let mut parts = vec!["The quick brown fox".to_string()];
    for (_, content) in added.into_iter().take(40) {
        parts.push(content);
        parts.push("jumps over 123 dogs".to_string());
    }
    parts.join(" ")
}

/// Compare tokie against HF on the added-token probe plus a set of edge-case
/// strings around the first special token.
fn compare_added_token_probe(hf_model: &str) {
    let tok = Tokenizer::from_pretrained(hf_model)
        .unwrap_or_else(|e| panic!("Failed to load tokie {hf_model}: {e}"));
    let mut hf = HfTokenizer::from_pretrained(hf_model, None)
        .unwrap_or_else(|e| panic!("Failed to load HF {hf_model}: {e}"));
    let _ = hf.with_truncation(None);
    hf.with_padding(None);

    let mut cases = vec![added_token_probe(&hf), " leading space".into(), "trailing space ".into()];
    if let Some((_, first_special)) = hf
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, t)| t.special)
        .map(|(id, t)| (id, t.content))
        .min()
    {
        let s = first_special;
        cases.extend([
            format!("Hello {s} world"),
            format!("Hello {s}world"),
            format!("Hello{s} world"),
            format!("{s} lead"),
            format!("tail {s}"),
            format!("a  {s}  b"),
            s,
        ]);
    }
    for text in &cases {
        let tokie_ids = tok.encode(text, false).ids;
        let hf_enc = hf
            .encode(text.as_str(), false)
            .unwrap_or_else(|e| panic!("HF encode failed for {hf_model}: {e}"));
        assert_eq!(
            tokie_ids.as_slice(),
            hf_enc.get_ids(),
            "added-token probe mismatch for {hf_model} on {text:?}"
        );
    }
}

macro_rules! probe_test {
    ($name:ident, $hf:expr) => {
        #[test]
        #[ignore] // Requires network
        fn $name() {
            compare_added_token_probe($hf);
        }
    };
}

// The 13 repos held back from the v13 regeneration:
probe_test!(probe_roberta_base,        "FacebookAI/roberta-base");                       // <mask> lstrip
probe_test!(probe_modernbert_base,     "answerdotai/ModernBERT-base");                   // <mask> lstrip + NFC
probe_test!(probe_jina_v2_base_code,   "jinaai/jina-embeddings-v2-base-code");           // <mask> lstrip
probe_test!(probe_bge_m3,              "BAAI/bge-m3");                                   // <mask> lstrip, SP
probe_test!(probe_snowflake_arctic_v2, "Snowflake/snowflake-arctic-embed-l-v2.0");       // <mask> lstrip, SP
probe_test!(probe_mxbai_de,            "mixedbread-ai/deepset-mxbai-embed-de-large-v1"); // stale ids remapped by HF
probe_test!(probe_mistral_7b,          "mistralai/Mistral-7B-v0.1");                     // Metaspace prepend_scheme=first
probe_test!(probe_phi_3_mini,          "microsoft/Phi-3-mini-4k-instruct");              // rstrip specials
probe_test!(probe_voyage_code_2,       "voyageai/voyage-code-2");                        // normalized specials (▁-fused)
probe_test!(probe_voyage_law_2,        "voyageai/voyage-law-2");                         // normalized specials (▁-fused)
probe_test!(probe_voyage_finance_2,    "voyageai/voyage-finance-2");                     // mixed normalized specials
probe_test!(probe_voyage_multi_2,      "voyageai/voyage-multilingual-2");                // mixed normalized specials
probe_test!(probe_potion_multilingual, "minishlab/potion-multilingual-128M");            // punct-pad normalizer chain

// Regression guards: models whose current behavior must survive the flag work.
probe_test!(probe_tinyllama,           "TinyLlama/TinyLlama-1.1B-Chat-v1.0");            // unconditional ▁ per segment
probe_test!(probe_xlm_roberta,         "FacebookAI/xlm-roberta-base");                   // SP precompiled, no flags
probe_test!(probe_all_mpnet,           "sentence-transformers/all-mpnet-base-v2");       // lstrip no-op under Bert pretok

// ============================================================================
// Web-text (OpenWebText) per-document accuracy — one representative per
// pretokenizer family / algorithm, loaded from the ORIGINAL HF repo so the
// tokenizer.json detection path is exercised (tokiers/*.tkz bypasses it).
// ============================================================================

macro_rules! owt_accuracy_test {
    ($name:ident, $hf:expr) => {
        #[test]
        #[ignore] // Requires network + benches/data/owt_sample.txt
        fn $name() {
            let docs = load_owt_docs(25_000_000);
            compare_model_docs($hf, &docs);
        }
    };
}

owt_accuracy_test!(owt_gpt2,        "openai-community/gpt2");          // GPT-2 pretok
owt_accuracy_test!(owt_qwen3,       "Qwen/Qwen3-0.6B");                // Qwen pretok
owt_accuracy_test!(owt_smollm2,     "HuggingFaceTB/SmolLM2-135M");     // SmolLM pretok
owt_accuracy_test!(owt_deepseek_v3, "deepseek-ai/DeepSeek-V3");        // DeepSeek pretok
owt_accuracy_test!(owt_bert,        "google-bert/bert-base-uncased");  // WordPiece
owt_accuracy_test!(owt_xlm_roberta,  "FacebookAI/xlm-roberta-base");  // SP-Unigram (precompiled charsmap)
owt_accuracy_test!(owt_mistral_7b,  "mistralai/Mistral-7B-v0.1");      // SP-BPE

// ============================================================================
// tiktoken models (CL100K, O200K) — compared against tiktoken-rs
// ============================================================================

/// Compare tokie against tiktoken-rs.
fn compare_tiktoken(tokiers_repo: &str, tiktoken_model: &str, text: &str) -> (bool, Option<usize>) {
    let tok = Tokenizer::from_pretrained(tokiers_repo)
        .unwrap_or_else(|e| panic!("Failed to load tokie {tokiers_repo}: {e}"));
    let bpe = tiktoken_rs::get_bpe_from_model(tiktoken_model)
        .unwrap_or_else(|e| panic!("Failed to load tiktoken {tiktoken_model}: {e}"));

    let tokie_ids = tok.encode(text, false).ids;
    let tiktoken_ids = bpe.encode_with_special_tokens(text);

    if tokie_ids.as_slice() == tiktoken_ids.as_slice() {
        (true, None)
    } else {
        let diff = tokie_ids.iter().zip(tiktoken_ids.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(tokie_ids.len().min(tiktoken_ids.len()));
        (false, Some(diff))
    }
}

macro_rules! tiktoken_accuracy_test {
    ($name:ident, $tokiers:expr, $tiktoken_model:expr) => {
        #[test]
        #[ignore]
        fn $name() {
            let text = load_enwik8(1_000_000);
            let (pass, diff) = compare_tiktoken($tokiers, $tiktoken_model, &text);
            assert!(pass, "Token mismatch at index {:?}", diff);
        }
    };
}

tiktoken_accuracy_test!(tiktoken_cl100k, "tokiers/cl100k", "gpt-4");
tiktoken_accuracy_test!(tiktoken_o200k,  "tokiers/o200k",  "gpt-4o");

/// SentencePiece added-token splitting: HF normalizes each segment around
/// a special token independently, so the Prepend("▁") + Replace(" "→"▁")
/// sequence emits ▁ tokens on both sides of the special ("Hello </s>
/// world" -> ▁Hello ▁ </s> ▁ ▁world). tokie used to skip the prepend on
/// space-leading segments and drop those ▁ tokens.
#[test]
#[ignore] // Requires network
fn sp_added_token_segment_prepend() {
    let hf_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
    let tok = Tokenizer::from_pretrained(hf_model)
        .unwrap_or_else(|e| panic!("Failed to load tokie {hf_model}: {e}"));
    let mut hf = HfTokenizer::from_pretrained(hf_model, None)
        .unwrap_or_else(|e| panic!("Failed to load HF {hf_model}: {e}"));
    let _ = hf.with_truncation(None);
    let cases = [
        "Hello </s> world",
        "Hello </s>world",
        "Hello</s> world",
        "</s>",
        " </s>",
        "multi </s> tokens </s> here",
        " spaces lead",
        "Hello  </s>  world",
    ];
    for text in cases {
        let tokie_ids = tok.encode(text, false).ids;
        let hf_ids = hf.encode(text, false).unwrap();
        assert_eq!(
            tokie_ids.as_slice(),
            hf_ids.get_ids(),
            "mismatch on {text:?}"
        );
    }
}
