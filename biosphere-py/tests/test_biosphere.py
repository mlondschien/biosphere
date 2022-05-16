from pathlib import Path

import numpy as np
import pytest

from biosphere import DecisionTree, RandomForest

_IRIS_FILE = "iris.csv"
_IRIS_PATH = Path(__file__).resolve().parents[2] / "testdata" / _IRIS_FILE


def test_forest():
    data = np.loadtxt(_IRIS_PATH, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3, 4))
    X = data[:, 0:4]
    y = data[:, 4]

    random_forest = RandomForest()
    oob_predictions = random_forest.fit_predict_oob(X, y)
    predictions = random_forest.predict(X)

    oob_mse = np.mean((oob_predictions - y) ** 2)
    mse = np.mean((predictions - y) ** 2)

    assert oob_mse < 0.05
    assert mse < oob_mse / 2


def test_tree():
    data = np.loadtxt(_IRIS_PATH, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3, 4))
    X = data[:, 0:4]
    y = data[:, 4]

    decision_tree = DecisionTree()
    decision_tree.fit(X, y)
    predictions = decision_tree.predict(X)

    mse = np.mean((predictions - y) ** 2)

    assert mse < 0.05


# TODO: Better test checking that supplying parameters had correct effect.
@pytest.mark.parametrize("max_features", [0.2, 3, "sqrt", None])
def test_max_features(max_features):
    _ = RandomForest(max_features=max_features)
    _ = DecisionTree(max_features=max_features)
