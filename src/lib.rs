//! Fast, simple random forests in Rust.
//!
//! Biosphere trains and runs decision-tree ensembles for regression or binary
//! classification on tabular data. It is designed to be easy to use with minimal
//! hyperparameter tuning: adjust tree count and seed, everything else has sensible
//! defaults.
//!
//! # Quickstart
//!
//! ```rust
//! use biosphere::{RandomForest, RandomForestParameters};
//! use ndarray::array;
//!
//! let X = array![[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]];
//! let y = array![0.0, 1.0, 0.0, 1.0];
//!
//! let mut forest = RandomForest::new(RandomForestParameters::default());
//! forest.fit(&X.view(), &y.view());
//!
//! let predictions = forest.predict(&X.view());
//! ```
//!
//! # GPU inference
//!
//! Enable the `gpu` cargo feature to convert a trained forest to a [`FlatForest`]
//! and upload it to a [`gpu::GpuForest`] for batched inference on Metal, Vulkan,
//! or DX12. See the [`gpu`] module for a complete example.

// Allow capital X for arrays.
#![allow(non_snake_case)]
pub use flat_forest::{FlatForest, FlatNode, ForestMeta};
pub use forest::RandomForest;
pub use forest::RandomForestParameters;
pub use tree::{DecisionTree, DecisionTreeParameters, MaxFeatures};
mod flat_forest;
mod forest;
mod tree;
pub mod utils; // This is public for benchmarking only.

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(test)]
mod testing;
