//! GPU-accelerated inference for trained random forests.
//!
//! Enabled with the `gpu` feature flag. Uses [wgpu] compute shaders (Vulkan,
//! Metal, DX12, WebGPU) with no platform-specific code.
//!
//! # Quick start — single forest
//!
//! ```rust,no_run
//! use biosphere::{FlatForest, RandomForest, RandomForestParameters};
//! use biosphere::gpu::GpuForest;
//! use ndarray::Array2;
//!
//! // 1. Train a forest as usual.
//! let X: Array2<f64> = /* ... your training data ... */
//! # Array2::zeros((1, 1));
//! let y = /* ... your labels ... */
//! # ndarray::Array1::zeros(1);
//! let mut forest = RandomForest::new(RandomForestParameters::default());
//! forest.fit(&X.view(), &y.view());
//!
//! // 2. Convert to a flat representation (works on CPU too).
//! let n_features = X.ncols();
//! let flat = FlatForest::from_forest(&forest, n_features);
//!
//! // 3. Upload to the GPU, reserving capacity for up to 1024 samples per call.
//! let gpu_forest = GpuForest::from_flat_forest(&flat, 1024).unwrap();
//!
//! // 4. Run inference. Features must be f32 (cast from f64 if needed).
//! let test_X: Array2<f64> = /* ... your test data ... */
//! # Array2::zeros((1, 1));
//! let test_X_f32 = test_X.mapv(|v| v as f32);
//! let predictions: ndarray::Array1<f32> = gpu_forest.predict(&test_X_f32.view()).unwrap();
//! ```
//!
//! # Multiple forests — shared device
//!
//! When serving several forests, initialise a [`GpuContext`] once and pass it
//! to each [`GpuForest::with_context`] call. This reuses the GPU device and
//! compiled compute pipelines, avoiding redundant initialisation overhead.
//!
//! ```rust,no_run
//! use biosphere::{FlatForest, RandomForest, RandomForestParameters};
//! use biosphere::gpu::{GpuContext, GpuForest};
//! use ndarray::Array2;
//!
//! # let make_flat = |seed| {
//! #     let mut f = RandomForest::new(RandomForestParameters::default().with_seed(seed));
//! #     f.fit(&Array2::<f64>::zeros((2, 1)).view(), &ndarray::Array1::zeros(2).view());
//! #     FlatForest::from_forest(&f, 1)
//! # };
//! # let flat_a = make_flat(1);
//! # let flat_b = make_flat(2);
//! // Initialise device + compile shaders once.
//! let ctx = GpuContext::new().unwrap();
//!
//! // Each forest gets its own node/meta buffers and inference buffers,
//! // but shares the device, queue, and pipelines.
//! let forest_a = GpuForest::with_context(ctx.clone(), &flat_a, 1024);
//! let forest_b = GpuForest::with_context(ctx.clone(), &flat_b, 2048);
//! ```
//!
//! # Data flow
//!
//! ```text
//! RandomForest  ──from_forest──►  FlatForest (f32 nodes, CPU)
//!                                      │
//!                         from_flat_forest / with_context
//!                                      │
//!                                      ▼
//!                               GpuForest (f32, GPU)
//!                                      │
//!                                  predict()
//!                                      │
//!                        ┌────────────────────────┐
//!                        │ traverse.wgsl          │
//!                        │  one thread per        │
//!                        │  (sample, tree) pair   │
//!                        └──────────┬─────────────┘
//!                                   │ per-tree predictions
//!                        ┌──────────▼─────────────┐
//!                        │ reduce.wgsl             │
//!                        │  mean across trees      │
//!                        └──────────┬─────────────┘
//!                                   │
//!                           Array1<f32> output
//! ```
//!
//! # Pipelining — overlapping GPU work across batches
//!
//! [`GpuForest::predict`] submits work and blocks until results are ready.
//! For higher throughput, use [`GpuForest::predict_submit`] to start GPU work
//! immediately and [`PredictHandle::collect`] to retrieve results later.
//! Submitting from two forked handles before collecting either lets the GPU
//! run both jobs concurrently.
//!
//! ```rust,no_run
//! use biosphere::{FlatForest, RandomForest, RandomForestParameters};
//! use biosphere::gpu::GpuForest;
//! use ndarray::Array2;
//!
//! # let flat = FlatForest::from_forest(
//! #     &{ let mut f = RandomForest::new(RandomForestParameters::default());
//! #        f.fit(&Array2::<f64>::zeros((2,2)).view(), &ndarray::Array1::zeros(2).view()); f },
//! #     2);
//! // Two handles sharing compiled pipelines and uploaded node data.
//! let forest_a = GpuForest::from_flat_forest(&flat, 1024).unwrap();
//! let forest_b = forest_a.fork(1024);
//!
//! let batch_a: Array2<f32> = /* ... first batch ... */
//! # Array2::zeros((1, 2));
//! let batch_b: Array2<f32> = /* ... second batch ... */
//! # Array2::zeros((1, 2));
//!
//! // Submit both without waiting — the GPU can work on them concurrently.
//! let handle_a = forest_a.predict_submit(&batch_a.view()).unwrap().unwrap();
//! let handle_b = forest_b.predict_submit(&batch_b.view()).unwrap().unwrap();
//!
//! // Collect in any order; each blocks only until its own submission is done.
//! let preds_a = handle_a.collect().unwrap();
//! let preds_b = handle_b.collect().unwrap();
//! ```
//!
//! Each [`GpuForest`] instance enforces a single-outstanding-handle constraint:
//! calling [`predict_submit`] again before collecting the previous handle panics.
//! Use [`GpuForest::fork`] to get independent handles.
//!
//! [`predict_submit`]: GpuForest::predict_submit
//!
//! # Precision
//!
//! [`FlatForest`] stores thresholds and leaf values as `f32`. CPU inference
//! casts feature values to `f32` before comparison so that split decisions are
//! identical to GPU inference. Results may differ slightly from
//! [`RandomForest::predict`] (which uses `f64` throughout); the difference is
//! within normal f32 rounding error (~1 × 10⁻⁷ relative).
//!
//! Because both [`FlatForest::predict`] and [`GpuForest::predict`] use `f32`
//! comparisons, their predictions agree very closely. The only remaining
//! difference is that `FlatForest` accumulates leaf values in `f64` while the
//! GPU shader accumulates in `f32`; this contributes at most ~n_trees × 10⁻⁷
//! additional error.
//!
//! [`FlatForest`]: crate::FlatForest
//! [`RandomForest::predict`]: crate::RandomForest::predict

mod pipeline;
pub use pipeline::{GpuContext, GpuForest, GpuInferenceError, GpuInitError, PredictHandle};
