# Biosphere

Simple, fast random forests.

Random forests with a runtime of `O(n d log(n) + n_estimators d n max_depth)` instead of `O(n_estimators mtry n log(n) max_depth)`.

`biosphere` is available as a rust crate and as a Python package.

## GPU Inference

Enable the `gpu` feature for wgpu-based GPU inference:

```toml
biosphere = { version = "0.5", features = ["gpu"] }
```

Convert a trained `RandomForest` to a flat representation and run inference on the GPU:

```rust
use biosphere::{RandomForest, FlatForest};
use biosphere::gpu::GpuForest;

// train as usual
let mut forest = RandomForest::default();
forest.fit(&X.view(), &y.view());

// convert and upload to GPU
let flat = FlatForest::from(&forest);
let gpu = GpuForest::from_flat_forest(&flat, /* max_samples */ 1024);

// predict — returns Vec<f32>, one value per sample
let predictions = gpu.predict(&features_flat, n_samples);
```

For multithreaded use, call `gpu.fork()` to get a per-thread handle that shares compiled shaders and node data without re-uploading anything.

For overlapping CPU and GPU work, use `predict_submit` / `PredictHandle::collect` separately.

`FlatForest` also works on CPU without the `gpu` feature — the cache-friendly f32 node layout gives a modest speedup over `RandomForest::predict`, at the cost of f32 precision (~1e-5 difference in leaf values):

```rust
let flat = FlatForest::from(&forest);
let predictions = flat.predict(&X.view()); // Vec<f32>
```

## Serialize / deserialize a `DecisionTree`

Enable the `serde` feature and choose a serde format (here: `postcard`):

```toml
# Cargo.toml
biosphere = { version = "0.4.2", features = ["serde"] }
postcard = { version = "1", features = ["use-std"] }
```

```rust
use biosphere::DecisionTree;

let X = ndarray::array![[0.0], [1.0], [2.0], [3.0]];
let y = ndarray::array![0.0, 0.0, 1.0, 1.0];

let mut tree = DecisionTree::default();
tree.fit(&X.view(), &y.view());

// serialize and deserialize the tree
let bytes = postcard::to_stdvec(&tree).unwrap();
// deserialize the tree from bytes
let restored: DecisionTree = postcard::from_bytes(&bytes).unwrap();

assert_eq!(tree.predict(&X.view()), restored.predict(&X.view()));
```

In this repo you can run: `cargo run --example decision_tree_serde --features serde`.
