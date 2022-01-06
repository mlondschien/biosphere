// Allow capital X for arrays.
#![allow(non_snake_case)]
pub use forest::RandomForest;
pub use forest::RandomForestParameters;
mod forest;
mod quick_sort;
pub mod tree;
pub mod utils;

#[cfg(test)]
mod testing;
