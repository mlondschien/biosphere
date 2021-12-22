// Allow capital X for arrays.
#![allow(non_snake_case)]
pub use tree::DecisionTree;
mod tree;
mod utils;

mod forest;
#[cfg(test)]
mod testing;
