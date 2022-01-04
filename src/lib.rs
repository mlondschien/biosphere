// Allow capital X for arrays.
#![allow(non_snake_case)]
pub use forest::RandomForest;
pub use tree::DecisionTree;
mod forest;
mod quick_sort;
mod tree;
pub mod utils;

#[cfg(test)]
mod testing;
