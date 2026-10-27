use tokie_bench::hf;
use tokie_bench::rows::{SAMPLES, ROWS};

#[test]
fn every_cut_has_a_row() {
    for cut in tokie::cut::Cut::ALL {
        assert!(
            ROWS.iter().any(|row| row.cut == cut),
            "{cut:?} has no oracle row"
        );
    }
}

#[test]
fn a_present_json_matches_tokenizers_v1() {
    let mut checked = 0usize;
    for row in ROWS {
            let Some(path) = tokie_bench::rows::model_path(row) else {
            continue;
        };
        let mut model = tokie::Tokenizer::open(&path).unwrap_or_else(|err| {
            panic!("{}: {err}", row.name);
        });
        let reference = match hf::load(&path) {
            Ok(tokenizer) => tokenizer,
            Err(err) => {
                eprintln!("{}: tokenizers v1 refused the file: {err}", row.name);
                continue;
            }
        };
        let mut refused = false;
        for sample in SAMPLES {
            let mut ids = Vec::new();
            model.encode(sample, &mut ids).unwrap_or_else(|err| {
                panic!("{} on {sample:?}: {err}", row.name);
            });
            let hf = match hf::ids_of(&reference, sample) {
                Ok(ids) => ids,
                Err(err) => {
                    eprintln!("{}: tokenizers v1 refused to encode: {err}", row.name);
                    refused = true;
                    break;
                }
            };
            assert_eq!(ids, hf, "{} on {sample:?}", row.name);
            assert_decode(&model, &reference, &ids, row.name, sample);
            ids.clear();
            model.wrap(sample, &mut ids).unwrap_or_else(|err| {
                panic!("{} specials on {sample:?}: {err}", row.name);
            });
            let hf = hf::ids_with_specials(&reference, sample).unwrap_or_else(|err| {
                panic!("{} specials on {sample:?}: tokenizers v1: {err}", row.name);
            });
            assert_eq!(ids, hf, "{} specials on {sample:?}", row.name);
            assert_decode(&model, &reference, &ids, row.name, sample);
        }
        if refused {
            continue;
        }
        checked += 1;
    }
    assert!(checked > 0, "no tokenizer.json was available to check");
}

#[test]
fn a_present_json_matches_gigatoken() {
    let mut checked = 0usize;
    for row in ROWS {
        let Some(path) = tokie_bench::rows::model_path(row) else {
            continue;
        };
        let mut giga = match gigatoken_rs::load_tokenizer::hf::load_hf_bpe(&path) {
            Ok(tokenizer) => tokenizer,
            Err(_) => {
                let Some(ranks) = tokie_bench::rows::tiktoken_path(row.name) else {
                    eprintln!("{}: gigatoken has no loader for this file", row.name);
                    continue;
                };
                let text = std::fs::read_to_string(&path).expect("kimi json");
                let value: serde_json::Value =
                    serde_json::from_str(&text).expect("kimi json parses");
                let specials = value
                    .get("added_tokens")
                    .and_then(|token| token.as_array())
                    .expect("kimi added tokens")
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
                .unwrap_or_else(|err| panic!("{}: tiktoken load: {err}", row.name))
            }
        };
        let mut model = tokie::Tokenizer::open(&path).unwrap_or_else(|err| {
            panic!("{}: {err}", row.name);
        });
        for sample in SAMPLES {
            let mut ids = Vec::new();
            model.encode(sample, &mut ids).unwrap_or_else(|err| {
                panic!("{} on {sample:?}: {err}", row.name);
            });
            let mut giga_ids = Vec::new();
            giga.encode_with_added_tokens_flat(sample.as_bytes(), &mut giga_ids);
            assert_eq!(ids, giga_ids, "{} on {sample:?}", row.name);
        }
        checked += 1;
    }
    assert!(checked > 0, "no tokenizer.json was available to check against gigatoken");
}

fn assert_decode(
    model: &tokie::Tokenizer,
    reference: &tk_encode::tokenizer::pipeline::PipelineTokenizer,
    ids: &[u32],
    row: &str,
    sample: &str,
) {
    let mut text = String::new();
    model
        .decode(ids, &mut text, tokie::Specials::Keep)
        .unwrap_or_else(|err| panic!("{row} decode {sample:?}: {err}"));
    let hf = hf::text_of(reference, ids, false).unwrap_or_else(|err| {
        panic!("{row} decode {sample:?}: tokenizers v1: {err}")
    });
    assert_eq!(text, hf, "{row} decode {sample:?}");
    text.clear();
    model
        .decode(ids, &mut text, tokie::Specials::Skip)
        .unwrap_or_else(|err| panic!("{row} decode skip {sample:?}: {err}"));
    let hf = hf::text_of(reference, ids, true).unwrap_or_else(|err| {
        panic!("{row} decode skip {sample:?}: tokenizers v1: {err}")
    });
    assert_eq!(text, hf, "{row} decode skip {sample:?}");
}

#[test]
fn batch_matches_encode() {
    let Some(row) = ROWS.iter().find(|row| tokie_bench::rows::model_path(row).is_some()) else {
        return;
    };
    let path = tokie_bench::rows::model_path(row).expect("row has a file");
    let mut tokenizer = tokie::Tokenizer::open(&path).unwrap_or_else(|err| panic!("{}: {err}", row.name));
    let mut alone = tokenizer.fork();
    tokenizer.threads(2);
    let mut ids = Vec::new();
    let mut ends = Vec::new();
    tokenizer
        .batch(SAMPLES, &mut ids, &mut ends)
        .unwrap_or_else(|err| panic!("{} batch: {err}", row.name));
    let mut expected = Vec::new();
    let mut expected_ends = Vec::new();
    for sample in SAMPLES {
        alone
            .encode(sample, &mut expected)
            .unwrap_or_else(|err| panic!("{} encode {sample:?}: {err}", row.name));
        expected_ends.push(expected.len());
    }
    assert_eq!(ids, expected, "{} batch ids", row.name);
    assert_eq!(ends, expected_ends, "{} batch ends", row.name);
}
