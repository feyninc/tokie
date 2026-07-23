use std::sync::{Arc, RwLock};

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::create_exception;
use pyo3::types::PyBytes;

create_exception!(tokie, TokieError, pyo3::exceptions::PyException);

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    TokieError::new_err(e.to_string())
}

/// Result of encoding text, with token IDs, attention mask, and type IDs.
///
/// `tokens` and `special_tokens_mask` are derived lazily from `ids` on access:
/// building per-token strings for every encoding is pure overhead for the
/// common ids-only consumers, and it used to happen under the GIL, serially,
/// for every batch element.
#[pyclass(name = "Encoding")]
#[derive(Clone)]
struct PyEncoding {
    #[pyo3(get)]
    ids: Vec<u32>,
    attention_mask_inner: Vec<u8>,
    type_ids_inner: Vec<u8>,
    offsets_inner: Vec<(usize, usize)>,
    tok: Arc<RwLock<tokie_core::Tokenizer>>,
}

#[pymethods]
impl PyEncoding {
    #[getter]
    fn tokens(&self) -> Vec<String> {
        let tokenizer = self.tok.read().unwrap();
        self.ids.iter().map(|&id| {
            tokenizer.id_to_token(id)
                .map(|s| s.into_owned())
                .unwrap_or_default()
        }).collect()
    }

    #[getter]
    fn special_tokens_mask(&self) -> Vec<u32> {
        let tokenizer = self.tok.read().unwrap();
        self.ids.iter().map(|&id| {
            if tokenizer.post_processor().is_special_token(id) { 1 } else { 0 }
        }).collect()
    }

    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        // Empty inner + non-empty ids means the fast ids-only path was taken:
        // no padding was configured, so the mask is all ones by construction
        if self.attention_mask_inner.len() != self.ids.len() {
            return vec![1u32; self.ids.len()];
        }
        self.attention_mask_inner.iter().map(|&x| x as u32).collect()
    }

    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        if self.type_ids_inner.len() != self.ids.len() {
            return vec![0u32; self.ids.len()];
        }
        self.type_ids_inner.iter().map(|&x| x as u32).collect()
    }

    #[getter]
    fn offsets(&self) -> Vec<(usize, usize)> {
        self.offsets_inner.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Encoding(ids=[...{}], attention_mask=[...{}], type_ids=[...{}])",
            self.ids.len(),
            self.attention_mask_inner.len(),
            self.type_ids_inner.len(),
        )
    }

    fn __len__(&self) -> usize {
        self.ids.len()
    }
}

impl PyEncoding {
    /// Create from a core Encoding. O(1) beyond moving the buffers — token
    /// strings and special-token masks are derived on attribute access.
    fn from_encoding(enc: tokie_core::Encoding, tok: Arc<RwLock<tokie_core::Tokenizer>>) -> Self {
        Self {
            ids: enc.ids,
            attention_mask_inner: enc.attention_mask,
            type_ids_inner: enc.type_ids,
            offsets_inner: enc.offsets,
            tok,
        }
    }

}

/// Fast, correct tokenizer. Supports BPE, WordPiece, and Unigram.
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: Arc<RwLock<tokie_core::Tokenizer>>,
}

impl PyTokenizer {
    fn read(&self) -> std::sync::RwLockReadGuard<'_, tokie_core::Tokenizer> {
        self.inner.read().unwrap()
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, tokie_core::Tokenizer> {
        self.inner.write().unwrap()
    }
}

#[pymethods]
impl PyTokenizer {
    /// Load a tokenizer from a HuggingFace tokenizer.json file.
    #[staticmethod]
    fn from_json(path: &str) -> PyResult<Self> {
        let inner = tokie_core::Tokenizer::from_json(path).map_err(to_py_err)?;
        Ok(Self { inner: Arc::new(RwLock::new(inner)) })
    }

    /// Load a tokenizer from a .tkz binary file.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = tokie_core::Tokenizer::from_file(path).map_err(to_py_err)?;
        Ok(Self { inner: Arc::new(RwLock::new(inner)) })
    }

    /// Download and load a tokenizer from the HuggingFace Hub.
    /// Tries .tkz first, then falls back to tokenizer.json.
    #[cfg(feature = "hf")]
    #[staticmethod]
    fn from_pretrained(py: Python<'_>, repo_id: &str) -> PyResult<Self> {
        let repo = repo_id.to_string();
        let inner = py.allow_threads(|| {
            tokie_core::Tokenizer::from_pretrained(&repo).map_err(to_py_err)
        })?;
        Ok(Self { inner: Arc::new(RwLock::new(inner)) })
    }

    /// Call the tokenizer: encode text or a text pair.
    /// Usage: tokenizer("text") or tokenizer("text_a", "text_b")
    #[pyo3(signature = (text, text_pair=None, add_special_tokens=true))]
    fn __call__(
        &self,
        py: Python<'_>,
        text: &str,
        text_pair: Option<&str>,
        add_special_tokens: bool,
    ) -> PyEncoding {
        match text_pair {
            Some(pair) => {
                let a = text.to_string();
                let b = pair.to_string();
                let inner = self.read();
                let enc = py.allow_threads(|| inner.encode_pair(&a, &b, add_special_tokens));
                PyEncoding::from_encoding(enc, self.inner.clone())
            }
            None => {
                let text = text.to_string();
                let inner = self.read();
                let enc = py.allow_threads(|| inner.encode(&text, add_special_tokens));
                PyEncoding::from_encoding(enc, self.inner.clone())
            }
        }
    }

    /// Encode text into an Encoding (ids, attention_mask, type_ids).
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn encode(&self, py: Python<'_>, text: &str, add_special_tokens: bool) -> PyEncoding {
        let text = text.to_string();
        let inner = self.read();
        if inner.padding().is_none() {
            // Fast path: bare ids; masks are synthesized lazily on access
            let ids = py.allow_threads(|| inner.encode_ids(&text, add_special_tokens));
            return PyEncoding {
                ids,
                attention_mask_inner: Vec::new(),
                type_ids_inner: Vec::new(),
                offsets_inner: Vec::new(),
                tok: self.inner.clone(),
            };
        }
        let enc = py.allow_threads(|| inner.encode(&text, add_special_tokens));
        PyEncoding::from_encoding(enc, self.inner.clone())
    }

    /// Encode a pair of texts (e.g. for cross-encoder models).
    #[pyo3(signature = (text_a, text_b, add_special_tokens=true))]
    fn encode_pair(
        &self,
        py: Python<'_>,
        text_a: &str,
        text_b: &str,
        add_special_tokens: bool,
    ) -> PyEncoding {
        let a = text_a.to_string();
        let b = text_b.to_string();
        let inner = self.read();
        let enc = py.allow_threads(|| inner.encode_pair(&a, &b, add_special_tokens));
        PyEncoding::from_encoding(enc, self.inner.clone())
    }

    /// Encode text into an Encoding with byte offsets into the (normalized) input.
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn encode_with_offsets(&self, py: Python<'_>, text: &str, add_special_tokens: bool) -> PyEncoding {
        let text = text.to_string();
        let inner = self.read();
        let enc = py.allow_threads(|| inner.encode_with_offsets(&text, add_special_tokens));
        PyEncoding::from_encoding(enc, self.inner.clone())
    }

    /// Encode raw bytes into token IDs.
    fn encode_bytes(&self, data: &[u8]) -> Vec<u32> {
        self.read().encode_bytes(data)
    }

    /// Decode token IDs back to a string. Returns None if not valid UTF-8.
    fn decode(&self, tokens: Vec<u32>) -> Option<String> {
        self.read().decode(&tokens)
    }

    /// Decode token IDs back to raw bytes.
    fn decode_bytes<'py>(&self, py: Python<'py>, tokens: Vec<u32>) -> Bound<'py, PyBytes> {
        let bytes = self.read().decode_bytes(&tokens);
        PyBytes::new(py, &bytes)
    }

    /// Convert a token ID to its string representation.
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.read().id_to_token(id).map(|s| s.into_owned())
    }

    /// Convert a token string to its ID.
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.read().token_to_id(token)
    }

    /// Get the full vocabulary as a dict mapping token strings to IDs.
    fn get_vocab(&self) -> HashMap<String, u32> {
        self.read().get_vocab()
    }

    /// Decode multiple token ID sequences in parallel.
    fn decode_batch(&self, py: Python<'_>, sequences: Vec<Vec<u32>>) -> Vec<Option<String>> {
        let inner = self.read();
        py.allow_threads(|| {
            sequences.iter().map(|tokens| inner.decode(tokens)).collect()
        })
    }

    /// Encode multiple texts in parallel, returning a list of Encoding objects.
    #[pyo3(signature = (texts, add_special_tokens=true))]
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>, add_special_tokens: bool) -> Vec<PyEncoding> {
        let inner = self.read();
        let encodings = py.allow_threads(|| {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            inner.encode_batch(&text_refs, add_special_tokens)
        });
        encodings.into_iter()
            .map(|enc| PyEncoding::from_encoding(enc, self.inner.clone()))
            .collect()
    }

    /// Encode multiple texts in parallel into one contiguous buffer.
    ///
    /// Returns `(ids, lengths)` as numpy arrays: `ids` is a uint32 array of
    /// every document's token ids concatenated in order; `lengths` is a
    /// uint64 array of per-document id counts (`np.cumsum(lengths)` gives
    /// document end offsets). No per-document Python objects are
    /// materialized; the Rust buffers are handed to numpy without copying.
    /// Padding is not applied; truncation and special tokens are.
    #[pyo3(signature = (texts, add_special_tokens=true))]
    fn encode_batch_flat<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> (Bound<'py, numpy::PyArray1<u32>>, Bound<'py, numpy::PyArray1<u64>>) {
        use numpy::IntoPyArray;
        let inner = self.read();
        let (ids, lens) = py.allow_threads(|| {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            inner.encode_batch_flat(&text_refs, add_special_tokens)
        });
        (ids.into_pyarray(py), lens.into_pyarray(py))
    }

    /// Count the number of tokens in the text.
    fn count_tokens(&self, py: Python<'_>, text: &str) -> usize {
        let text = text.to_string();
        let inner = self.read();
        py.allow_threads(|| inner.count_tokens(&text))
    }

    /// Count tokens for multiple texts in parallel.
    fn count_tokens_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<usize> {
        let inner = self.read();
        py.allow_threads(|| {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            inner.count_tokens_batch(&text_refs)
        })
    }

    /// Save the tokenizer to a .tkz binary file.
    fn save(&self, path: &str) -> PyResult<()> {
        self.read().to_file(path).map_err(to_py_err)
    }

    /// The vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.read().vocab_size()
    }

    /// The pad token ID, if set.
    #[getter]
    fn pad_token_id(&self) -> Option<u32> {
        self.read().pad_token_id()
    }

    /// Enable padding for encode_batch.
    #[pyo3(signature = (*, direction="right", pad_id=0, pad_type_id=0, length=None, pad_to_multiple_of=None))]
    fn enable_padding(
        &self,
        direction: &str,
        pad_id: u32,
        pad_type_id: u8,
        length: Option<usize>,
        pad_to_multiple_of: Option<usize>,
    ) -> PyResult<()> {
        let direction = match direction {
            "right" => tokie_core::PaddingDirection::Right,
            "left" => tokie_core::PaddingDirection::Left,
            _ => return Err(TokieError::new_err("direction must be 'left' or 'right'")),
        };
        let strategy = match length {
            Some(n) => tokie_core::PaddingStrategy::Fixed(n),
            None => tokie_core::PaddingStrategy::BatchLongest,
        };
        let params = tokie_core::PaddingParams {
            strategy,
            direction,
            pad_to_multiple_of,
            pad_id,
            pad_type_id,
        };
        self.write().enable_padding(params);
        Ok(())
    }

    /// Enable truncation.
    #[pyo3(signature = (max_length, *, stride=0, strategy="longest_first", direction="right"))]
    fn enable_truncation(
        &self,
        max_length: usize,
        stride: usize,
        strategy: &str,
        direction: &str,
    ) -> PyResult<()> {
        let strategy = match strategy {
            "longest_first" => tokie_core::TruncationStrategy::LongestFirst,
            "only_first" => tokie_core::TruncationStrategy::OnlyFirst,
            "only_second" => tokie_core::TruncationStrategy::OnlySecond,
            _ => return Err(TokieError::new_err("strategy must be 'longest_first', 'only_first', or 'only_second'")),
        };
        let direction = match direction {
            "right" => tokie_core::TruncationDirection::Right,
            "left" => tokie_core::TruncationDirection::Left,
            _ => return Err(TokieError::new_err("direction must be 'left' or 'right'")),
        };
        let params = tokie_core::TruncationParams {
            max_length,
            strategy,
            direction,
            stride,
        };
        self.write().enable_truncation(params);
        Ok(())
    }

    /// Get current padding configuration, or None if disabled.
    #[getter]
    fn padding(&self) -> Option<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            self.read().padding().map(|p| {
                let mut d = HashMap::new();
                match p.strategy {
                    tokie_core::PaddingStrategy::Fixed(n) =>
                        d.insert("length".to_string(), n.into_pyobject(py).unwrap().into_any().unbind()),
                    tokie_core::PaddingStrategy::BatchLongest =>
                        d.insert("length".to_string(), py.None()),
                };
                d.insert("pad_to_multiple_of".to_string(),
                    p.pad_to_multiple_of.map(|n| n.into_pyobject(py).unwrap().into_any().unbind())
                        .unwrap_or_else(|| py.None()));
                d.insert("pad_id".to_string(), p.pad_id.into_pyobject(py).unwrap().into_any().unbind());
                d.insert("pad_type_id".to_string(), p.pad_type_id.into_pyobject(py).unwrap().into_any().unbind());
                let dir_str = match p.direction {
                    tokie_core::PaddingDirection::Right => "right",
                    tokie_core::PaddingDirection::Left => "left",
                };
                d.insert("direction".to_string(), dir_str.into_pyobject(py).unwrap().into_any().unbind());
                d
            })
        })
    }

    /// Get current truncation configuration, or None if disabled.
    #[getter]
    fn truncation(&self) -> Option<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            self.read().truncation().map(|t| {
                let mut d = HashMap::new();
                d.insert("max_length".to_string(), t.max_length.into_pyobject(py).unwrap().into_any().unbind());
                d.insert("stride".to_string(), t.stride.into_pyobject(py).unwrap().into_any().unbind());
                let strat = match t.strategy {
                    tokie_core::TruncationStrategy::LongestFirst => "longest_first",
                    tokie_core::TruncationStrategy::OnlyFirst => "only_first",
                    tokie_core::TruncationStrategy::OnlySecond => "only_second",
                };
                d.insert("strategy".to_string(), strat.into_pyobject(py).unwrap().into_any().unbind());
                let dir_str = match t.direction {
                    tokie_core::TruncationDirection::Right => "right",
                    tokie_core::TruncationDirection::Left => "left",
                };
                d.insert("direction".to_string(), dir_str.into_pyobject(py).unwrap().into_any().unbind());
                d
            })
        })
    }

    /// Disable padding.
    fn no_padding(&self) {
        self.write().no_padding();
    }

    /// Disable truncation.
    fn no_truncation(&self) {
        self.write().no_truncation();
    }

    /// Number of special tokens added for a single sequence or pair.
    #[pyo3(signature = (is_pair=false))]
    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        self.read().num_special_tokens_to_add(is_pair)
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.read().vocab_size())
    }
}

#[pymodule]
fn tokie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyEncoding>()?;
    m.add("TokieError", m.py().get_type::<TokieError>())?;
    Ok(())
}
