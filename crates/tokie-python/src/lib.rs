//! Python bindings for the tokie 0.2 tokenizer.
//!
//! `encode` and `batch` borrow the Python strings, release the GIL, and return
//! NumPy arrays that own the id buffer. The Rust call is `Tokenizer::encode`
//! and `Tokenizer::batch`. No per-id Python objects are built.

use std::sync::{Arc, Mutex};

use numpy::Element;
use numpy::IntoPyArray;
use numpy::PyArray1;
use numpy::PyArrayDescrMethods;
use numpy::PyReadonlyArray1;
use numpy::npyffi::{self, PY_ARRAY_API};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyString, PyStringMethods};
use tokie_core::{Plus, Specials, Tokenizer};

create_exception!(tokie, TokieError, pyo3::exceptions::PyException);

fn to_py_err(err: impl std::fmt::Display) -> PyErr {
    TokieError::new_err(err.to_string())
}

/// Copy the ids out of a NumPy array while the GIL is held.
///
/// The typed argument already rejects non-`uint32` dtypes, non-native byte
/// order, and ranks other than one. This check keeps the call strict about
/// layout: a non-empty array must also be C-contiguous and aligned so that one
/// bulk copy can stand in for element-by-element access. Empty arrays pass as
/// a zero-length slice, matching `decode`'s accepted empty input.
fn copy_ids(ids: &PyReadonlyArray1<'_, u32>) -> PyResult<Vec<u32>> {
    ids.as_slice()
        .map(<[u32]>::to_vec)
        .map_err(|_| PyValueError::new_err("ids must be a C-contiguous and aligned array"))
}

/// One tokenizer. Encode and batch take the lock and run without the GIL.
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: Mutex<Tokenizer>,
    /// Id storage from the last array the caller dropped. The next encode
    /// writes into it, so the pages stay resident.
    ids: Arc<Mutex<Vec<u32>>>,
}

fn fresh(inner: Tokenizer) -> PyTokenizer {
    PyTokenizer {
        inner: Mutex::new(inner),
        ids: Arc::new(Mutex::new(Vec::new())),
    }
}

#[pymethods]
impl PyTokenizer {
    /// Read a `.tokie` file or a `tokenizer.json`.
    #[staticmethod]
    fn open(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path = path.to_string();
        let inner = py.detach(|| Tokenizer::open(&path).map_err(to_py_err))?;
        Ok(fresh(inner))
    }

    /// Parse the text of a `tokenizer.json`.
    #[staticmethod]
    fn json(py: Python<'_>, text: &str) -> PyResult<Self> {
        let text = text.to_string();
        let inner = py.detach(|| Tokenizer::json(&text).map_err(to_py_err))?;
        Ok(fresh(inner))
    }

    /// Download the repo's `.tokie` file, or `tokenizer.json` when the repo has none.
    #[staticmethod]
    fn from_pretrained(py: Python<'_>, repo: &str) -> PyResult<Self> {
        let repo = repo.to_string();
        let inner = py.detach(|| Tokenizer::pretrained(&repo).map_err(to_py_err))?;
        Ok(fresh(inner))
    }

    /// Read a `.tokie` file or a `tokenizer.json` from `path`. Alias for `open`;
    /// `path` is a filename, not the json text.
    #[staticmethod]
    fn from_json(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path = path.to_string();
        let inner = py.detach(|| Tokenizer::from_json(&path).map_err(to_py_err))?;
        Ok(fresh(inner))
    }

    /// Read a `.tokie` file or a `tokenizer.json` from `path`. Alias for `open`;
    /// `path` is a filename, not the json text.
    #[staticmethod]
    fn from_file(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path = path.to_string();
        let inner = py.detach(|| Tokenizer::from_file(&path).map_err(to_py_err))?;
        Ok(fresh(inner))
    }

    /// Append-style encode. Specials that occur in `text` are emitted.
    /// The single-sequence template is not applied. Returns a `uint32` array.
    fn encode<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyArray1<u32>>> {
        // `text` points at the Python string's UTF-8 cache. The object stays
        // alive for this call, and the cache is immutable, so encode can run
        // without copying the text.
        let ids = py.detach(|| {
            let mut tok = self.inner.lock().unwrap();
            let mut ids = checkout(&self.ids, text.len() / 4 + 16);
            tok.encode(text, &mut ids).map_err(|err| err.to_string())?;
            Ok::<Vec<u32>, String>(ids)
        })
        .map_err(to_py_err)?;
        ids_array(py, &self.ids, ids)
    }

    /// `encode`, then the template. `pair` uses the pair template.
    /// `truncate` cuts to `max_len`. `pad` extends to `max_len`.
    /// Returns a `uint32` array.
    #[pyo3(signature = (text, pair=None, max_len=None, truncate=false, pad=false))]
    fn encode_plus<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        pair: Option<&str>,
        max_len: Option<usize>,
        truncate: bool,
        pad: bool,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let ids = py.detach(|| {
            let mut tok = self.inner.lock().unwrap();
            let mut ids = checkout(&self.ids, text.len() / 4 + 16);
            if pair.is_none() && !truncate && !pad {
                tok.encode_plus(text, &mut ids).map_err(|err| err.to_string())?;
            } else {
                let mut plus = Plus::new(text);
                if let Some(pair) = pair {
                    plus = plus.pair(pair);
                }
                if truncate {
                    let n = max_len.ok_or("truncate needs a length")?;
                    plus = plus.truncate(n);
                }
                if pad {
                    let n = max_len.ok_or("pad needs a length")?;
                    plus = plus.pad(n);
                }
                tok.encode_plus(plus, &mut ids).map_err(|err| err.to_string())?;
            }
            Ok::<Vec<u32>, String>(ids)
        })
        .map_err(to_py_err)?;
        ids_array(py, &self.ids, ids)
    }

    /// How many ids `encode` would return for `text`. The template stays off.
    fn count_tokens(&self, py: Python<'_>, text: &str) -> PyResult<usize> {
        py.detach(|| {
            let mut tok = self.inner.lock().unwrap();
            tok.count_tokens(text).map_err(|err| err.to_string())
        })
        .map_err(to_py_err)
    }

    /// One length per text, from `batch`. The template stays off.
    fn count_tokens_batch(&self, py: Python<'_>, texts: Vec<Bound<'_, PyString>>) -> PyResult<Vec<usize>> {
        let borrowed = texts
            .iter()
            .map(|text| text.to_str())
            .collect::<PyResult<Vec<&str>>>()?;
        py.detach(|| {
            let mut tok = self.inner.lock().unwrap();
            tok.count_tokens_batch(&borrowed)
                .map_err(|err| err.to_string())
        })
        .map_err(to_py_err)
    }

    /// Text for `ids`. `ids` is a 1-D `uint32` NumPy array that is
    /// C-contiguous, aligned, and native-endian. Empty and read-only arrays are
    /// accepted. A wrong dtype, non-native byte order, or other rank raises
    /// `TypeError`; a non-empty non-contiguous or misaligned array raises
    /// `ValueError`. Do not mutate `ids` while this call runs. `skip` omits
    /// specials.
    #[pyo3(signature = (ids, skip=false))]
    fn decode(
        &self,
        py: Python<'_>,
        ids: PyReadonlyArray1<'_, u32>,
        skip: bool,
    ) -> PyResult<String> {
        let specials = if skip { Specials::Skip } else { Specials::Keep };
        let id_vec = copy_ids(&ids)?;
        // Drop the readonly borrow guard before `py.detach` releases the GIL.
        // The core call reads `id_vec`, which owns its ids, so no NumPy memory
        // crosses the GIL release.
        drop(ids);
        py.detach(|| {
            let tok = self.inner.lock().unwrap();
            let mut text = String::new();
            tok.decode(&id_vec, &mut text, specials).map_err(|err| err.to_string())?;
            Ok::<String, String>(text)
        })
        .map_err(to_py_err)
    }

    /// Raw decoded bytes for `ids`. `skip` omits specials. Invalid UTF-8 stays
    /// as bytes. `ids` is a 1-D `uint32` NumPy array that is C-contiguous,
    /// aligned, and native-endian. Empty and read-only arrays are accepted. A
    /// wrong dtype, non-native byte order, or other rank raises `TypeError`; a
    /// non-empty non-contiguous or misaligned array raises `ValueError`. Do not
    /// mutate `ids` while this call runs.
    #[pyo3(signature = (ids, skip=false))]
    fn decode_bytes<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray1<'_, u32>,
        skip: bool,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let specials = if skip { Specials::Skip } else { Specials::Keep };
        let id_vec = copy_ids(&ids)?;
        // Same rule as `decode`: only the owned copy reaches the core call.
        drop(ids);
        let bytes = py
            .detach(|| {
                self.inner
                    .lock()
                    .unwrap()
                    .decode_bytes(&id_vec, specials)
                    .map_err(|err| err.to_string())
            })
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Encode each text with `encode`. Returns `(ids, ends)` as `uint32` and
    /// `uint64` arrays. `ends[i]` is the id count after text `i`.
    fn batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<Bound<'py, PyString>>,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u64>>)> {
        let borrowed = texts
            .iter()
            .map(|text| text.to_str())
            .collect::<PyResult<Vec<&str>>>()?;
        let (ids, ends) = py.detach(|| {
            let mut tok = self.inner.lock().unwrap();
            let bytes = borrowed.iter().map(|text| text.len()).sum::<usize>();
            let mut ids = checkout(&self.ids, bytes / 4 + 16);
            let mut ends = Vec::with_capacity(borrowed.len());
            tok.batch(&borrowed, &mut ids, &mut ends)
                .map_err(|err| err.to_string())?;
            let ends: Vec<u64> = ends.into_iter().map(|n| n as u64).collect();
            Ok::<(Vec<u32>, Vec<u64>), String>((ids, ends))
        })
        .map_err(to_py_err)?;
        Ok((ids_array(py, &self.ids, ids)?, ends.into_pyarray(py)))
    }

    /// How many workers `batch` uses.
    fn threads(&self, n: usize) {
        self.inner.lock().unwrap().threads(n);
    }

    /// The vocab string for `id`.
    fn token(&self, id: u32) -> Option<String> {
        self.inner.lock().unwrap().token(id).map(str::to_string)
    }

    /// The id for a vocab string.
    fn id(&self, token: &str) -> Option<u32> {
        self.inner.lock().unwrap().id(token)
    }

    /// The largest id plus one, counting empty slots.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.lock().unwrap().vocab_size()
    }

    /// The pad id, when the vocab spells it `[PAD]` or `<pad>`.
    #[getter]
    fn pad_token_id(&self) -> Option<u32> {
        self.inner.lock().unwrap().pad_token_id()
    }

    /// The vocabulary with its raw tokenizer.json spelling.
    fn get_vocab<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let vocab = py.detach(|| self.inner.lock().unwrap().get_vocab());
        let dict = PyDict::new(py);
        for (token, id) in vocab {
            dict.set_item(token, id)?;
        }
        Ok(dict)
    }

    /// Ids the template adds for one text, or for a pair.
    #[pyo3(signature = (is_pair=false))]
    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        self.inner.lock().unwrap().num_special_tokens_to_add(is_pair)
    }

    /// Write this tokenizer as a `.tokie` file.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let path = path.to_string();
        py.detach(|| {
            self.inner.lock().unwrap().save(&path).map_err(to_py_err)
        })
    }

    fn __repr__(&self) -> &'static str {
        "Tokenizer()"
    }
}

/// Owns an id buffer for as long as the NumPy array that views it is alive.
/// Dropping the array returns the buffer to the tokenizer.
#[pyclass]
struct IdOwner {
    pool: Arc<Mutex<Vec<u32>>>,
    buf: Vec<u32>,
}

impl Drop for IdOwner {
    fn drop(&mut self) {
        let mut slot = self.pool.lock().unwrap();
        self.buf.clear();
        if self.buf.capacity() > slot.capacity() {
            *slot = std::mem::take(&mut self.buf);
        }
    }
}

fn checkout(pool: &Mutex<Vec<u32>>, need: usize) -> Vec<u32> {
    let mut slot = pool.lock().unwrap();
    let mut ids = std::mem::take(&mut *slot);
    drop(slot);
    ids.clear();
    if ids.capacity() < need {
        ids.reserve(need);
    }
    ids
}

fn ids_array<'py>(
    py: Python<'py>,
    pool: &Arc<Mutex<Vec<u32>>>,
    ids: Vec<u32>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let len = ids.len();
    let owner = Bound::new(
        py,
        IdOwner {
            pool: Arc::clone(pool),
            buf: ids,
        },
    )?;
    let data = owner.borrow().buf.as_ptr();
    let mut dims = [len as npyffi::npy_intp];
    let strides = [std::mem::size_of::<u32>() as npyffi::npy_intp];
    unsafe {
        let ptr = PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            npyffi::get_type_object(py, npyffi::NpyTypes::PyArray_Type),
            u32::get_dtype(py).into_dtype_ptr(),
            1,
            dims.as_mut_ptr(),
            strides.as_ptr() as *mut npyffi::npy_intp,
            data as *mut std::ffi::c_void,
            npyffi::NPY_ARRAY_WRITEABLE,
            std::ptr::null_mut(),
        );
        if ptr.is_null() {
            return Err(PyErr::fetch(py));
        }
        let owner_ptr = owner.into_ptr();
        if PY_ARRAY_API.PyArray_SetBaseObject(py, ptr.cast(), owner_ptr) < 0 {
            pyo3::ffi::Py_DECREF(owner_ptr);
            pyo3::ffi::Py_DECREF(ptr);
            return Err(PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, ptr).cast_into_unchecked())
    }
}

#[pymodule]
fn tokie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add("TokieError", m.py().get_type::<TokieError>())?;
    Ok(())
}
