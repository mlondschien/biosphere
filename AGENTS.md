Biosphere is a Rust library for fast and simple Random Forests.

# Project Overview

The repository consists of:
- **Rust core library** (`src/`): Core random forest and decision tree implementation
- **Python bindings** (`biosphere-py/`): PyO3-based Python package built with Maturin

## Development Commands

Run tests with all features:

```bash
cargo test --all-features
```

*Note:** Add/update tests when APIs are added or changed.

## Core Components

1. **DecisionTree** (`src/tree/decision_tree.rs`)
   - Individual decision tree implementation
   - Uses sorted samples for efficient splitting
   - Configured via `DecisionTreeParameters`

2. **RandomForest** (`src/forest.rs`)
   - Ensemble of decision trees
   - Parallel training using rayon threadpool
   - OOB (out-of-bag) prediction support
   - Configured via `RandomForestParameters`

3. **Tree Node Structure** (`src/tree/decision_tree_node.rs`)
   - Recursive tree node implementation
   - Handles splitting logic and prediction

4. **Utilities** (`src/utils.rs`)
   - `argsort`: Sorting indices computation
   - `sample_weights`: Bootstrap sample generation
   - `sorted_samples`: Pre-sorting optimization for training

## Important Implementation Notes

**Array Naming Convention:**
The codebase uses capital `X` for feature arrays (following scikit-learn convention) and allows this via `#![allow(non_snake_case)]` in lib.rs.

**Random State:**
- Seeds are u64 values
- Each tree in a forest gets a derived seed for reproducibility
- Uses `StdRng` from rand crate

**MaxFeatures Options:**
The `MaxFeatures` enum supports multiple feature sampling strategies:
- `None`: Use all features
- `Value(usize)`: Use specific number of features
- `Fraction(f64)`: Use fraction of features
- `Sqrt`: Use sqrt(n_features)

## GPU Feature (`--features gpu`)

**What was built:**
- `src/flat_forest.rs` — `FlatForest` / `FlatNode` / `ForestMeta`: flat tree array for CPU and GPU inference. Nodes in BFS visit order with **explicit child indices** (`left: i32`, `right: i32`; -1 for leaves). `max_tree_size` is the actual max node count across trees (NOT `2^(max_depth+1) - 1`). `FlatForest.meta: ForestMeta` holds the four `u32` dimensions; `FlatForest` impls `Deref<Target = ForestMeta>` for ergonomic access.
- `src/gpu/pipeline.rs` — `GpuForest`: uploads `FlatForest` nodes **directly** (already `f32`; zero conversion), compiles WGSL pipelines with device-queried workgroup size, runs batched inference via `predict(&ArrayView2<f32>) -> Array1<f32>`.
- `src/gpu/shaders/traverse.wgsl` — 2D dispatch `(ceil(n_samples/wg_size), n_trees, 1)`, one thread per (sample, tree). Uses `u32(node.left)` / `u32(node.right)` for traversal; `max_depth` is the GPU loop safety bound.
- `src/gpu/shaders/reduce.wgsl` — averages per-tree predictions per sample.
- Tests: `tests/flat_forest.rs` (CPU, no feature flag), `tests/gpu_inference.rs` (requires GPU, `--features gpu`).

**FlatNode layout (16 bytes, `#[repr(C)]`):**
```
left: i32 | right: i32 | feature_index: u32 | value: f32
```
`value` is dual-purpose: split threshold for internal nodes, leaf prediction for leaves. `left < 0` is the sole discriminant — no separate `is_leaf` field needed. 16 bytes = exactly 4 nodes per 64-byte cache line. Under `gpu` feature, `FlatNode` and `ForestMeta` derive `bytemuck::Pod + Zeroable`, so nodes upload as `bytemuck::cast_slice::<FlatNode, u8>(&flat.nodes)` with no conversion loop.

**Precision model:**
- `FlatNode` stores `value: f32`. `FlatForest::predict` takes `&ArrayView2<f32>` directly — no per-split cast. `GpuForest::predict` also takes `&ArrayView2<f32>`. Both CPU and GPU traversal are therefore identical.
- `FlatForest::predict` results differ from `RandomForest::predict` (f64) by ~1e-5 due to f32 quantisation of leaf values; use that tolerance in tests.
- `FlatForest::predict` vs `GpuForest::predict` differ by <1e-5 (only f64 vs f32 accumulation of f32 leaf values).
- Callers convert f64 training data once with `X.mapv(|v| v as f32)` before calling either predict.

**FlatForest::predict_one loop bound:**
Uses `0..=self.meta.max_depth` (inclusive), not `max_tree_size`. This matches the GPU shader and gives the compiler a tight, small bound (e.g. 9 iterations for depth-8 trees), enabling better optimisation. `max_depth` is computed by `node_depth()` during `from_forest` — it is the actual maximum depth across all trees, so `max_depth + 1` iterations always suffice to reach any leaf.

**GpuForest API (wgpu 28):**
- `GpuForest::from_flat_forest(flat: &FlatForest, max_samples: usize) -> GpuForest` — creates GPU buffers sized for up to `max_samples` rows. Pre-allocates 4 buffers at init; no per-call VRAM allocation.
- `GpuForest::fork(&self, max_samples: usize) -> GpuForest` — clones the `Arc<GpuForestShared>` (device, queue, pipelines, static node/meta buffers) and allocates fresh per-instance buffers. Use this to get per-thread handles without recompiling shaders or re-uploading node data. Inherits `collect_timeout` from the parent.
- `GpuForest::is_uma() -> bool` — returns `true` when the adapter supports `MAPPABLE_PRIMARY_BUFFERS` (Apple Silicon, Vulkan/DX12 UMA). Useful for logging/telemetry.
- `GpuForest::with_collect_timeout(Duration) -> GpuForest` — builder method to set the maximum wait time for `PredictHandle::collect`. Default: 10 seconds. Returns `self` for chaining after `from_flat_forest`.
- `GpuForest::predict(&ArrayView2<f32>) -> Array1<f32>` — calls `predict_submit` then `collect`. Uses `X.as_standard_layout()` internally to get a contiguous slice (zero-copy if already C-order).
- `GpuForest::predict_submit(&ArrayView2<f32>) -> Option<PredictHandle<'_>>` — submits GPU work immediately; returns `None` for empty input. `PredictHandle::collect() -> Array1<f32>` blocks until done (up to `collect_timeout`). Use submit+collect to overlap GPU work across multiple forests. **Panics if called while a previous `PredictHandle` is still outstanding** (busy flag via `AtomicBool`).
- Sub-range binding: pass `wgpu::BufferBinding { buffer, offset: 0, size: Some(BufferSize::new(feature_bytes)) }` — this is essential for `arrayLength()` to reflect the actual batch size, not buffer capacity.
- wgpu 28 poll API: `queue.submit()` returns a `SubmissionIndex`; pass it to `device.poll(PollType::Wait { submission_index: Some(idx), timeout: Some(duration) })` for per-submission blocking wait with timeout.
- Always call `staging_buffer.unmap()` after reading from a pre-allocated (reused) staging buffer, so the next `encoder.copy_buffer_to_buffer` + map cycle works correctly.

**UMA (Unified Memory Architecture) optimisation:**
- Detection: `adapter.features().contains(MAPPABLE_PRIMARY_BUFFERS)`, requested in `DeviceDescriptor::required_features`. Stored as `GpuForestShared::uma`; forked handles inherit it.
- Feature upload on UMA: `STORAGE | MAP_WRITE` buffer, mapped directly via `map_async(Write)` + `poll(Wait { submission_index: None, timeout: Some(1s) })` + `copy_from_slice`. No staging blit. The 1-second timeout catches wgpu validation errors that would otherwise hang.
- Output readback on UMA: `STORAGE | MAP_READ` buffer mapped directly via `map_async(Read)`; no `copy_buffer_to_buffer` + staging buffer needed.
- On discrete GPUs: feature upload uses `queue.write_buffer` (STORAGE | COPY_DST); output copied to a `COPY_DST | MAP_READ` staging buffer before readback.
- `staging_buffer: Option<wgpu::Buffer>` is `None` on UMA, `Some(...)` on discrete.

**Busy flag (double-submit guard):**
- `GpuForest::busy: AtomicBool` — set to `true` via `swap(true, Acquire)` at the start of `predict_submit`, reset to `false` via `store(false, Release)` at the end of `PredictHandle::collect`. Panics with "outstanding PredictHandle" if `predict_submit` is called while a handle is still live.

**wgpu 28 pipeline overridable constants (workgroup size):**
- WGSL: `override wg_size: u32 = 64u;` at top of shader, then `@compute @workgroup_size(wg_size, 1, 1)`. Override expressions are valid in `@workgroup_size`.
- Rust: `PipelineCompilationOptions { constants: &[("wg_size", value_as_f64)], ..Default::default() }`. The type is `&[(&str, f64)]` — **not** `HashMap<String, f64>` (that was an older wgpu API).
- Workgroup size heuristic: `device.limits().max_compute_invocations_per_workgroup.min(device.limits().max_compute_workgroup_size_x).min(256)`. Clamp both per-group and per-X-dimension limits; cap at 256 (larger groups rarely help memory-bandwidth-bound kernels). Store result in `GpuForestShared::workgroup_size` and use in `n_samples.div_ceil(shared.workgroup_size)` dispatch.

**Thread safety and TLS on Metal/wgpu:**
- wgpu/Metal GPU resources held in `thread_local!` cause `"cannot access a Thread Local Storage value during or after destruction: AccessError"` when rayon worker threads tear down, because Metal's OS autorelease pool is destroyed before Rust TLS destructors run.
- Fix: explicitly drop GPU resources in rayon's `exit_handler`, which runs while the thread is still alive (before TLS destructors). Example: `.exit_handler(|_| { GPU_FORESTS.with(|gf| { gf.borrow_mut().take(); }); })`
- This is cleaner than pre-allocating a `Vec<GpuForest>` indexed by thread, because lazy init and cleanup both stay local to the thread.

**Why not BFS positional encoding (`2i+1`/`2i+2`):**
Real-world RF trees trained without a `max_depth` constraint are deep and sparse. A positional BFS layout requires `2^(depth+1) - 1` slots per tree — depth ~44 means petabytes of allocation. Explicit child indices allocate only actual nodes.

## Known pre-existing issue

The debug assertion in `decision_tree_node.rs:180` fires for random training sets > ~300 samples — keep test training sizes ≤ 200 samples.

# Memory

**Keep this file updated.** After every session where you discover something stable and non-obvious about this project, add it below. Remove or correct entries that turn out to be wrong or outdated. Do not duplicate entries already in CLAUDE.md.

# Interactivity guidelines

When you are asked to implement something, always ask for clarifications if needed.
If you are unsure about the requirements, ask for more details.
If you think there is a better way to implement something, suggest it and explain your reasoning, but don't implement it immediately without approval.
