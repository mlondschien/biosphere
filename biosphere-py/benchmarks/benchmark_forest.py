import inspect

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from biosphere import RandomForest

from .common import Benchmark


class ScikitLearnForest(Benchmark):
    def __init__(self, n_jobs=2):
        self.n_jobs = n_jobs  # GitHub CI has two nodes.

    name = "scikit-learn forest"
    param_names = ["n", "n_estimators", "mtry"]
    # params=([1000, 10000, 100000], [100, 400], [4, 12], [2])
    params = ([10000], [100], [4, 12])

    def _setup_model(self, params):
        _, n_estimators, mtry = params
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=8,
            max_features=mtry,
            n_jobs=self.n_jobs,
            oob_score=True,
        )

    def _time_fit_predict_oob(self):
        self.oob_score = True
        self.model.fit(self.X_train, self.y_train)
        return (1 - self.model.oob_score_) * np.mean(
            (self.y_train - self.y_train.mean()) ** 2
        )


class BiosphereForest(Benchmark):
    def __init__(self, n_jobs=2):
        self.n_jobs = n_jobs  # GitHub CI has two nodes.

    name = "biosphere forest"
    param_names = ["n", "n_estimators", "mtry"]
    params = ([1000, 10000, 100000], [100], [4, 12])

    def _setup_model(self, params):
        _, n_estimators, mtry = params
        kwargs = {"n_trees": n_estimators, "max_depth": 8, "mtry": mtry}

        if "n_jobs" in inspect.getargspec(RandomForest.__init__):
            kwargs["n_jobs"] = self.n_jobs

        self.model = RandomForest(**kwargs)

    def _time_fit_predict_oob(self):
        predictions = self.model.fit_predict_oob(self.X_train, self.y_train)
        return mean_squared_error(self.y_train, predictions)
