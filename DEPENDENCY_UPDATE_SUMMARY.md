# Dependency Update Summary

This document summarizes the changes made to update Rust crate dependencies.

## Updated Dependencies

### Main Package (`Cargo.toml`)

| Dependency | Old Version | New Version |
|------------|-------------|-------------|
| ndarray | 0.16.1 | 0.17.1 |
| ndarray-rand | 0.15 | 0.16.0 |
| rand | 0.8 | 0.9 |

### Python Bindings (`biosphere-py/Cargo.toml`)

| Dependency | Old Version | New Version |
|------------|-------------|-------------|
| ndarray | 0.16 | 0.17.1 |
| numpy | 0.26 | 0.27.0 |
| pyo3 | 0.26 | 0.27.1 |

## Code Changes Required

### 1. rand 0.9 API Changes

**Issue:** `Uniform::new()` now returns `Result<Uniform<T>, Error>` instead of `Uniform<T>`

**Fix:** Added `.unwrap()` calls to `Uniform::new()` invocations
- `src/utils.rs`: 2 occurrences
- `src/tree/decision_tree_node.rs`: 2 occurrences

**Issue:** `rng.gen()` renamed to `rng.random()` and `rng.gen_range()` renamed to `rng.random_range()`

**Fix:** Updated method calls
- `src/forest.rs`: 3 occurrences of `gen()` → `random()`
- `src/utils.rs`: 1 occurrence of `gen_range()` → `random_range()`
- `src/tree/decision_tree_node.rs`: 1 occurrence of `gen_range()` → `random_range()`

### 2. pyo3 0.27 API Changes

**Issue:** `FromPyObject` trait signature changed to require two lifetimes

**Before:**
```rust
impl<'source> FromPyObject<'source> for PyMaxFeatures {
    fn extract_bound(ob: &pyo3::Bound<'source, PyAny>) -> PyResult<Self> {
        // ...
    }
}
```

**After:**
```rust
impl<'py> FromPyObject<'_, 'py> for PyMaxFeatures {
    type Error = PyErr;
    
    fn extract(ob: pyo3::Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        // ...
    }
}
```

**Changes:**
- Added `type Error = PyErr;`
- Changed method from `extract_bound` to `extract`
- Changed parameter from `&pyo3::Bound<'source, PyAny>` to `pyo3::Borrowed<'_, 'py, PyAny>`
- Changed lifetime parameters

**Files affected:**
- `biosphere-py/src/utils.rs`

### 3. numpy 0.27 API Changes

**Issue:** `ToPyArray` trait renamed to `IntoPyArray`, method `to_pyarray()` renamed to `into_pyarray()`

**Fix:** Updated imports and method calls
- `biosphere-py/src/decision_tree.rs`: Import and 1 method call
- `biosphere-py/src/random_forest.rs`: Import and 2 method calls

## Vendored Dependencies

Two dependencies required patching to support ndarray 0.17:

### ndarray-csv
- **Repository:** https://github.com/paulkernfeld/ndarray-csv
- **Change:** Updated ndarray constraint from `">= 0.7, < 0.17"` to `">= 0.7, < 0.18"`
- **Location:** `vendor/ndarray-csv/`

### rust-numpy
- **Repository:** https://github.com/PyO3/rust-numpy
- **Change:** Updated ndarray constraint from `">= 0.15, < 0.17"` to `">= 0.15, < 0.18"`
- **Location:** `vendor/rust-numpy/`

Both patches are minimal version constraint updates that allow the libraries to work with ndarray 0.17. The upstream projects should accept these changes once ndarray 0.17 becomes more widely adopted.

## Testing

All tests pass successfully:
- Main package: 29 tests passed
- Python bindings: Build successful (debug and release modes)
- Security scan: No vulnerabilities found

## Notes for Future Maintainers

1. Monitor the upstream repositories for official releases supporting ndarray 0.17:
   - https://github.com/paulkernfeld/ndarray-csv
   - https://github.com/PyO3/rust-numpy

2. Once official releases are available, remove the vendored versions and update to use the published crates.

3. The vendored dependencies should be considered temporary workarounds.
