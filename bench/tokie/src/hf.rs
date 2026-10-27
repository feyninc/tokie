//! Token ids from Hugging Face tokenizers v1.
//!
//! A `tokenizer.json` from the hub is version 1.0. Tokenizers v1 reads the
//! canonical 2.0 form, so the file is upgraded before it is encoded.

use tk_encode::tokenizer::pipeline::PipelineTokenizer;
use tk_encode::tokenizer::pipeline::encode_options::EncodeOptions;

pub fn load(path: &std::path::Path) -> Result<PipelineTokenizer, String> {
    let json = tk_convert::canonicalize_file(path).map_err(|err| err.to_string())?;
    tk_serialize::from_json(&json).map_err(|err| err.to_string())
}

pub fn reference_ids(path: &std::path::Path, text: &str) -> Vec<u32> {
    let tokenizer = load(path).unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
    ids_of(&tokenizer, text).unwrap_or_else(|err| panic!("tokenizers v1 encode failed: {err}"))
}

pub fn ids_of(tokenizer: &PipelineTokenizer, text: &str) -> Result<Vec<u32>, String> {
    ids_with(tokenizer, text, &EncodeOptions::no_specials())
}

pub fn ids_with_specials(tokenizer: &PipelineTokenizer, text: &str) -> Result<Vec<u32>, String> {
    ids_with(tokenizer, text, &EncodeOptions::default())
}

pub fn text_of(tokenizer: &PipelineTokenizer, ids: &[u32], skip_specials: bool) -> Result<String, String> {
    tokenizer.decode(ids, skip_specials).map_err(|err| err.to_string())
}

fn ids_with(
    tokenizer: &PipelineTokenizer,
    text: &str,
    options: &EncodeOptions,
) -> Result<Vec<u32>, String> {
    let encoded = tokenizer
        .encode(text, options)
        .wait()
        .map_err(|err| err.to_string())?;
    Ok(encoded
        .into_iter()
        .next()
        .expect("one input")
        .ids()
        .iter()
        .map(|token| token.id())
        .collect())
}
