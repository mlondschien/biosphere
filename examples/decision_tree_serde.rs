#![allow(non_snake_case)]

#[cfg(not(feature = "serde"))]
fn main() {
    eprintln!(
        "This example requires the `serde` feature.\n\nRun:\n  cargo run --example decision_tree_serde --features serde"
    );
}

#[cfg(feature = "serde")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use biosphere::DecisionTree;

    let X = ndarray::array![[0.0], [1.0], [2.0], [3.0]];
    let y = ndarray::array![0.0, 0.0, 1.0, 1.0];

    let mut tree = DecisionTree::default();
    tree.fit(&X.view(), &y.view());

    let bytes = postcard::to_stdvec(&tree)?;
    let restored: DecisionTree = postcard::from_bytes(&bytes)?;

    assert_eq!(tree.predict(&X.view()), restored.predict(&X.view()));
    println!("Serialized {} bytes", bytes.len());
    Ok(())
}
