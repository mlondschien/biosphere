import numpy as np
from pathlib import Path
from biosphere import predict_oob

_IRIS_FILE = "iris.csv"
_IRIS_PATH = Path(__file__).resolve().parents[2] / "testdata" / _IRIS_FILE

def test_predict_oob():
    data = np.loadtxt(_IRIS_PATH, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3, 4))
    X = data[:, 0:4]
    y = data[:, 4]
    result = predict_oob(X, y)
    mse = np.mean((result - y) ** 2)
    assert mse < 0.05