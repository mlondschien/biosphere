use crate::forest::RandomForest;
use crate::tree::DecisionTreeNode;
use ndarray::{Array1, ArrayView2};
use std::collections::VecDeque;

/// Forest metadata: dimensions shared by CPU and GPU inference.
///
/// NOTE: field order and types are significant for binary serde (e.g. postcard)
/// and for direct GPU upload via bytemuck. Changing them requires a migration step
/// for any data serialised with an earlier layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(
    any(feature = "gpu", feature = "serde"),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct ForestMeta {
    pub n_trees: u32,
    pub n_features: u32,
    /// Nodes per tree (all trees are padded to this size).
    pub max_tree_size: u32,
    /// Maximum depth across all trees (used as a GPU traversal bound).
    pub max_depth: u32,
}

/// A single node in a flat tree stored in BFS visit order.
///
/// Internal nodes use `feature_index`, `value` (as threshold), `left`, and `right`.
/// Leaf nodes use `value` (as the leaf prediction); `left` and `right` are -1.
///
/// `value` is a dual-purpose field: it holds the split threshold for internal
/// nodes and the leaf prediction for leaf nodes. The two interpretations never
/// overlap — `left < 0` unambiguously identifies leaves. This alias shrinks the
/// struct from 20 → 16 bytes, fitting exactly 4 nodes per 64-byte cache line.
///
/// Child indices are explicit offsets into the per-tree node slice, so trees
/// of any shape can be stored without exponential padding.
///
/// NOTE: field order and types are significant for binary serde (e.g. postcard)
/// and for direct GPU upload via bytemuck. Changing them requires a migration step
/// for any data serialised with an earlier layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(
    any(feature = "gpu", feature = "serde"),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct FlatNode {
    /// Index of the left child within the tree's node slice, or -1 for leaves.
    pub left: i32,
    /// Index of the right child within the tree's node slice, or -1 for leaves.
    pub right: i32,
    pub feature_index: u32,
    /// Split threshold for internal nodes; leaf prediction for leaf nodes.
    pub value: f32,
}

impl FlatNode {
    fn dummy_leaf() -> Self {
        FlatNode {
            left: -1,
            right: -1,
            feature_index: 0,
            value: 0.0,
        }
    }
}

/// A [`RandomForest`] converted to a flat, contiguous array for fast inference.
///
/// This is the bridge between a trained forest and GPU execution. Convert once
/// with [`FlatForest::from_forest`], then either call [`FlatForest::predict`]
/// for CPU inference or pass it to [`GpuForest::from_flat_forest`] to run on
/// the GPU.
///
/// ```rust
/// use biosphere::{FlatForest, RandomForest, RandomForestParameters};
/// use ndarray::array;
///
/// let X = array![[0.0, 1.0], [1.0, 0.0]];
/// let y = array![0.0, 1.0];
/// let mut forest = RandomForest::new(RandomForestParameters::default());
/// forest.fit(&X.view(), &y.view());
///
/// let flat = FlatForest::from_forest(&forest, X.ncols());
/// let X_f32 = X.mapv(|v| v as f32);
/// let predictions = flat.predict(&X_f32.view()); // comparable to forest.predict()
/// ```
///
/// Internally, each tree is stored in BFS order with explicit child indices so
/// deep, sparse trees (common in practice) don't require exponential padding.
///
/// All thresholds and leaf values are stored as `f32`. Features must be passed
/// as `f32` so that split decisions are identical to GPU inference. Results are
/// accumulated in `f64` and returned as `f64` for API consistency with
/// [`RandomForest::predict`].
///
/// [`GpuForest::from_flat_forest`]: crate::gpu::GpuForest::from_flat_forest
/// [`RandomForest`]: crate::RandomForest
#[derive(Clone)]
pub struct FlatForest {
    /// Flat node array: length = `meta.n_trees * meta.max_tree_size`.
    pub(crate) nodes: Vec<FlatNode>,
    /// Number of real (non-padding) nodes in each tree; computed once in
    /// `from_forest` alongside `max_tree_size` and reused by serde.
    #[allow(unused, reason = "serde needs this for round-trip validation")]
    pub(crate) tree_sizes: Vec<u32>,
    pub meta: ForestMeta,
}

impl std::ops::Deref for FlatForest {
    type Target = ForestMeta;
    fn deref(&self) -> &ForestMeta {
        &self.meta
    }
}

impl FlatForest {
    /// Build a `FlatForest` from a trained `RandomForest`.
    ///
    /// `n_features` must match the number of features the forest was trained on.
    /// Passing an incorrect value will produce silently wrong predictions,
    /// because `n_features` is used as the GPU shader's feature-indexing stride.
    pub fn from_forest(forest: &RandomForest, n_features: usize) -> Self {
        assert!(n_features > 0, "n_features must be > 0");
        let trees = forest.trees();
        let n_trees = trees.len();
        assert!(n_trees > 0, "Forest must have at least one tree");

        let max_depth = trees
            .iter()
            .map(|t| node_depth(t.root()))
            .max()
            .unwrap_or(0);

        let tree_sizes: Vec<u32> = trees.iter().map(|t| count_nodes(t.root()) as u32).collect();

        let max_tree_size = tree_sizes.iter().copied().max().unwrap_or(1) as usize;

        let mut nodes = vec![FlatNode::dummy_leaf(); n_trees * max_tree_size];

        for (tree_idx, tree) in trees.iter().enumerate() {
            let offset = tree_idx * max_tree_size;
            fill_bfs(tree.root(), &mut nodes[offset..offset + max_tree_size]);
        }

        FlatForest {
            nodes,
            tree_sizes,
            meta: ForestMeta {
                n_trees: n_trees as u32,
                n_features: n_features as u32,
                max_tree_size: max_tree_size as u32,
                max_depth: max_depth as u32,
            },
        }
    }

    /// Run inference on a batch of samples.
    ///
    /// Features must be `f32` to match the precision of stored thresholds and
    /// GPU inference. The per-tree leaf values (f32) are accumulated in f64 and
    /// the final mean is returned as `Array1<f64>`.
    ///
    /// `X` may be in any memory layout (row-major, column-major, or
    /// non-contiguous). The method converts to C-order internally; if `X` is
    /// already row-major this is a zero-copy borrow.
    ///
    /// The loop order is outer=tree, inner=sample so that each tree's node array
    /// stays warm in L3 cache while all samples are processed against it, rather
    /// than re-loading every tree's nodes for every sample.
    pub fn predict(&self, X: &ArrayView2<f32>) -> Array1<f64> {
        let X_c = X.as_standard_layout();
        let n_samples = X_c.nrows();
        let mut output = Array1::<f64>::zeros(n_samples);

        for tree_idx in 0..self.meta.n_trees as usize {
            for sample in 0..n_samples {
                let row = X_c.row(sample);
                let features = row
                    .as_slice()
                    .expect("standard layout is always contiguous");
                output[sample] += self.predict_one(tree_idx, features);
            }
        }

        output / self.meta.n_trees as f64
    }

    fn predict_one(&self, tree_idx: usize, features: &[f32]) -> f64 {
        let offset = tree_idx * self.meta.max_tree_size as usize;
        let mut idx = 0usize;
        for _ in 0..=self.meta.max_depth as usize {
            let node = &self.nodes[offset + idx];
            if node.left < 0 {
                return node.value as f64;
            }
            if features[node.feature_index as usize] < node.value {
                idx = node.left as usize;
            } else {
                idx = node.right as usize;
            }
        }
        // max_depth is computed from node_depth() during from_forest, so max_depth + 1
        // iterations always reach a leaf for a correctly built tree. Reaching here
        // indicates a bug in from_forest or fill_bfs.
        unreachable!(
            "predict_one: traversal exceeded max_depth={}",
            self.meta.max_depth
        );
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::{FlatForest, FlatNode, ForestMeta};
    use serde::de::{Error, Visitor};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::fmt;

    /// Newtype that serializes/deserializes `Vec<FlatNode>` as raw bytes.
    struct CompactNodes(Vec<FlatNode>);

    impl Serialize for CompactNodes {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            s.serialize_bytes(bytemuck::cast_slice(&self.0))
        }
    }

    impl<'de> Deserialize<'de> for CompactNodes {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = CompactNodes;
                fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, "raw bytes encoding a sequence of FlatNodes")
                }
                fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                    let node_size = std::mem::size_of::<FlatNode>();
                    if !v.len().is_multiple_of(node_size) {
                        return Err(E::custom(format!(
                            "expected a multiple of {node_size} bytes (FlatNode size), got {}",
                            v.len()
                        )));
                    }
                    let n = v.len() / node_size;
                    let mut nodes = vec![bytemuck::Zeroable::zeroed(); n];
                    bytemuck::cast_slice_mut::<FlatNode, u8>(&mut nodes).copy_from_slice(v);
                    Ok(CompactNodes(nodes))
                }
                fn visit_byte_buf<E: Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                    self.visit_bytes(&v)
                }
            }
            d.deserialize_bytes(V)
        }
    }

    /// Wire format: metadata + per-tree actual sizes + compact node bytes.
    /// Padding (dummy leaves) is not stored; it is reconstructed on deserialize.
    #[derive(Serialize, Deserialize)]
    struct Wire {
        meta: ForestMeta,
        /// Number of real nodes in each tree (prefix of the padded slot array).
        tree_sizes: Vec<u32>,
        /// Compact node data: only actual nodes, no padding, raw bytes.
        nodes: CompactNodes,
    }

    impl Serialize for FlatForest {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let max_tree_size = self.meta.max_tree_size as usize;

            let total: usize = self.tree_sizes.iter().map(|&s| s as usize).sum();
            let mut compact = Vec::with_capacity(total);
            for (i, &size) in self.tree_sizes.iter().enumerate() {
                let offset = i * max_tree_size;
                compact.extend_from_slice(&self.nodes[offset..offset + size as usize]);
            }

            Wire {
                meta: self.meta,
                tree_sizes: self.tree_sizes.clone(),
                nodes: CompactNodes(compact),
            }
            .serialize(s)
        }
    }

    impl<'de> Deserialize<'de> for FlatForest {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let Wire {
                meta,
                tree_sizes,
                nodes: CompactNodes(compact),
            } = Wire::deserialize(d)?;

            let n_trees = meta.n_trees as usize;
            let max_tree_size = meta.max_tree_size as usize;

            if tree_sizes.len() != n_trees {
                return Err(D::Error::custom(format!(
                    "expected {n_trees} tree_sizes entries, got {}",
                    tree_sizes.len()
                )));
            }
            let expected_total: usize = tree_sizes.iter().map(|&s| s as usize).sum();
            if compact.len() != expected_total {
                return Err(D::Error::custom(format!(
                    "expected {expected_total} compact nodes, got {}",
                    compact.len()
                )));
            }

            let mut nodes = vec![FlatNode::dummy_leaf(); n_trees * max_tree_size];
            let mut src = 0;
            for (i, &size) in tree_sizes.iter().enumerate() {
                let size = size as usize;
                if size > max_tree_size {
                    return Err(D::Error::custom(format!(
                        "tree {i}: size {size} exceeds max_tree_size {max_tree_size}"
                    )));
                }
                let dst = i * max_tree_size;
                nodes[dst..dst + size].copy_from_slice(&compact[src..src + size]);
                src += size;
            }

            Ok(FlatForest {
                nodes,
                tree_sizes,
                meta,
            })
        }
    }
}

/// Recursively compute the depth of a node (0 = leaf, 1 = one split level, …).
fn node_depth(node: &DecisionTreeNode) -> usize {
    if node.feature_index.is_none() {
        0
    } else {
        1 + node_depth(
            node.left_child
                .as_ref()
                .expect("internal node (feature_index is Some) must have a left child"),
        )
        .max(node_depth(node.right_child.as_ref().expect(
            "internal node (feature_index is Some) must have a right child",
        )))
    }
}

/// Count the total number of nodes in the subtree rooted at `node`.
fn count_nodes(node: &DecisionTreeNode) -> usize {
    if node.feature_index.is_none() {
        1
    } else {
        1 + count_nodes(
            node.left_child
                .as_ref()
                .expect("internal node (feature_index is Some) must have a left child"),
        ) + count_nodes(
            node.right_child
                .as_ref()
                .expect("internal node (feature_index is Some) must have a right child"),
        )
    }
}

/// Fill `nodes` with the tree rooted at `root` in BFS order, storing explicit
/// child indices in each internal node.
fn fill_bfs(root: &DecisionTreeNode, nodes: &mut [FlatNode]) {
    let mut queue: VecDeque<(&DecisionTreeNode, usize)> = VecDeque::new();
    queue.push_back((root, 0));
    let mut next_slot = 1usize;

    while let Some((node, idx)) = queue.pop_front() {
        if let Some(feature_index) = node.feature_index {
            let left_idx = next_slot;
            next_slot += 1;
            let right_idx = next_slot;
            next_slot += 1;

            nodes[idx] = FlatNode {
                left: left_idx as i32,
                right: right_idx as i32,
                feature_index: feature_index as u32,
                value: node
                    .feature_value
                    .expect("internal node (feature_index is Some) must have a feature_value")
                    as f32,
            };

            queue.push_back((
                node.left_child
                    .as_ref()
                    .expect("internal node (feature_index is Some) must have a left child"),
                left_idx,
            ));
            queue.push_back((
                node.right_child
                    .as_ref()
                    .expect("internal node (feature_index is Some) must have a right child"),
                right_idx,
            ));
        } else {
            nodes[idx] = FlatNode {
                left: -1,
                right: -1,
                feature_index: 0,
                value: node
                    .label
                    .expect("leaf node (feature_index is None) must have a label")
                    as f32,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RandomForest;
    use crate::forest::RandomForestParameters;
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::mem::size_of;

    fn make_data(
        n_samples: usize,
        n_features: usize,
        seed: u64,
    ) -> (Array2<f64>, ndarray::Array1<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let X = Array2::random_using(
            (n_samples, n_features),
            Uniform::new(0.0, 1.0).unwrap(),
            &mut rng,
        );
        let y = X.column(0).to_owned();
        (X, y)
    }

    /// nodes.len() must always equal n_trees * max_tree_size (test #16).
    #[test]
    fn nodes_length_invariant() {
        let (X, y) = make_data(120, 6, 16);
        let params = RandomForestParameters::default()
            .with_n_estimators(15)
            .with_seed(16);
        let mut forest = RandomForest::new(params);
        forest.fit(&X.view(), &y.view());

        let flat = FlatForest::from_forest(&forest, 6);
        assert_eq!(
            flat.nodes.len(),
            flat.n_trees as usize * flat.max_tree_size as usize,
            "nodes.len() should equal n_trees * max_tree_size"
        );
    }

    /// Every slot beyond a tree's real BFS footprint must be a dummy leaf (test #15).
    #[test]
    fn padding_nodes_are_leaves() {
        let (X, y) = make_data(150, 8, 15);
        let params = RandomForestParameters::default()
            .with_n_estimators(10)
            .with_seed(15);
        let mut forest = RandomForest::new(params);
        forest.fit(&X.view(), &y.view());

        let flat = FlatForest::from_forest(&forest, 8);
        let n_trees = flat.n_trees as usize;
        let max_tree_size = flat.max_tree_size as usize;

        for tree_idx in 0..n_trees {
            let offset = tree_idx * max_tree_size;
            let tree_nodes = &flat.nodes[offset..offset + max_tree_size];
            let real_count = count_bfs_nodes(tree_nodes);
            for (i, node) in tree_nodes
                .iter()
                .enumerate()
                .take(max_tree_size)
                .skip(real_count)
            {
                assert!(
                    node.left < 0,
                    "tree {tree_idx}, slot {i} (beyond real size {real_count}): expected dummy leaf, got left={}",
                    node.left
                );
            }
        }
    }

    /// `max_tree_size` must equal the actual maximum node count, not the BFS positional
    /// formula `2^(max_depth+1) - 1` (test #3). For unconstrained trees deep enough that
    /// the positional layout would differ, the two values must diverge.
    #[test]
    fn max_tree_size_is_actual_count_not_bfs_formula() {
        let (X, y) = make_data(150, 6, 42);
        let params = RandomForestParameters::default()
            .with_n_estimators(10)
            .with_seed(42);
        let mut forest = RandomForest::new(params);
        forest.fit(&X.view(), &y.view());

        let flat = FlatForest::from_forest(&forest, 6);

        // Independent recomputation of max node count using the private helper.
        let expected_max = forest
            .trees()
            .iter()
            .map(|t| count_nodes(t.root()))
            .max()
            .unwrap();
        assert_eq!(
            flat.max_tree_size as usize, expected_max,
            "max_tree_size should equal the actual max node count across trees"
        );

        // For trees with max_depth >= 4, the positional BFS layout would require
        // at least 2^5 - 1 = 31 nodes, whereas typical trees with the same depth
        // but sparse structure will have far fewer. Assert we use the compact count.
        let bfs_formula = (1usize << (flat.max_depth as usize + 1)).saturating_sub(1);
        if flat.max_depth >= 4 {
            assert!(
                (flat.max_tree_size as usize) < bfs_formula,
                "max_tree_size ({}) should be less than BFS formula ({}) for depth-{} trees",
                flat.max_tree_size,
                bfs_formula,
                flat.max_depth
            );
        }
    }

    /// Every padding node beyond a tree's real BFS footprint must have `value == 0.0` (test #5).
    /// This verifies that `FlatNode::dummy_leaf()` initialises `value` to zero, so padding
    /// slots cannot accidentally contribute non-zero predictions if the loop bound is wrong.
    #[test]
    fn dummy_leaf_value_is_zero() {
        let (X, y) = make_data(150, 8, 50);
        let params = RandomForestParameters::default()
            .with_n_estimators(10)
            .with_seed(50);
        let mut forest = RandomForest::new(params);
        forest.fit(&X.view(), &y.view());

        let flat = FlatForest::from_forest(&forest, 8);
        let max_tree_size = flat.max_tree_size as usize;

        for tree_idx in 0..flat.n_trees as usize {
            let offset = tree_idx * max_tree_size;
            let tree_nodes = &flat.nodes[offset..offset + max_tree_size];
            let real_count = count_bfs_nodes(tree_nodes);
            for (i, node) in tree_nodes
                .iter()
                .enumerate()
                .take(max_tree_size)
                .skip(real_count)
            {
                assert_eq!(
                    node.value, 0.0,
                    "tree {tree_idx}, padding slot {i}: expected value=0.0"
                );
            }
        }
    }

    /// Serializing then deserializing a FlatForest produces the same predictions.
    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip_predictions() {
        let (X, y) = make_data(150, 6, 99);
        let params = RandomForestParameters::default()
            .with_n_estimators(10)
            .with_seed(99);
        let mut forest = RandomForest::new(params);
        forest.fit(&X.view(), &y.view());
        let flat = FlatForest::from_forest(&forest, 6);

        let bytes = postcard::to_stdvec(&flat).unwrap();
        let restored: FlatForest = postcard::from_bytes(&bytes).unwrap();

        let X_f32 = X.mapv(|v| v as f32);
        let original = flat.predict(&X_f32.view());
        let from_serde = restored.predict(&X_f32.view());
        assert_eq!(
            original, from_serde,
            "predictions must match after round-trip"
        );
    }

    /// Compact serialization is smaller than the full padded node array.
    #[cfg(feature = "serde")]
    #[test]
    fn serde_compact_smaller_than_padded() {
        let (X, y) = make_data(150, 8, 77);
        let params = RandomForestParameters::default()
            .with_n_estimators(20)
            .with_seed(77);
        let mut forest = RandomForest::new(params);
        forest.fit(&X.view(), &y.view());
        let flat = FlatForest::from_forest(&forest, 8);

        let compact_bytes = postcard::to_stdvec(&flat).unwrap();
        let padded_bytes =
            flat.meta.n_trees as usize * flat.meta.max_tree_size as usize * size_of::<FlatNode>();
        assert!(
            compact_bytes.len() < padded_bytes,
            "compact serde ({} bytes) should be smaller than raw padded layout ({} bytes)",
            compact_bytes.len(),
            padded_bytes
        );
    }

    /// Count reachable nodes from root via BFS using explicit child indices.
    fn count_bfs_nodes(nodes: &[FlatNode]) -> usize {
        use std::collections::VecDeque;
        let mut queue = VecDeque::new();
        queue.push_back(0usize);
        let mut count = 0;
        while let Some(idx) = queue.pop_front() {
            if idx >= nodes.len() {
                break;
            }
            count += 1;
            let node = &nodes[idx];
            if node.left >= 0 {
                queue.push_back(node.left as usize);
                queue.push_back(node.right as usize);
            }
        }
        count
    }
}
