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
use numpy::npyffi::{self, PY_ARRAY_API};
use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyStringMethods};
use tokie_core::{Specials, Tokenizer};

create_exception!(tokie, TokieError, pyo3::exceptions::PyException);

fn to_py_err(err: impl std::fmt::Display) -> PyErr {
    TokieError::new_err(err.to_string())
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

    /// `encode`, then the single-sequence template.
    fn wrap<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let ids = py.detach(|| {
            let mut tok = self.inner.lock().unwrap();
            let mut ids = checkout(&self.ids, text.len() / 4 + 16);
            tok.wrap(text, &mut ids).map_err(|err| err.to_string())?;
            Ok::<Vec<u32>, String>(ids)
        })
        .map_err(to_py_err)?;
        ids_array(py, &self.ids, ids)
    }

    /// Text for `ids`. `skip` omits specials.
    #[pyo3(signature = (ids, skip=false))]
    fn decode(&self, py: Python<'_>, ids: Vec<u32>, skip: bool) -> PyResult<String> {
        let specials = if skip { Specials::Skip } else { Specials::Keep };
        py.detach(|| {
            let tok = self.inner.lock().unwrap();
            let mut text = String::new();
            tok.decode(&ids, &mut text, specials).map_err(|err| err.to_string())?;
            Ok::<String, String>(text)
        })
        .map_err(to_py_err)
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
