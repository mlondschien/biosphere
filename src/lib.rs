// Allow capital X for arrays.
#![allow(non_snake_case)]
pub use forest::RandomForest;
pub use forest::RandomForestParameters;
pub use tree::DecisionTree;
pub use tree::DecisionTreeParameters;
mod forest;
mod quick_sort;
mod tree;
pub mod utils; // This is public for benchmarking only.

#[cfg(test)]
mod testing;
