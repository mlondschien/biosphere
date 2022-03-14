import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from biosphere import RandomForest

from .common import Benchmark


class ScikitLearnForest(Benchmark):
    def __init__(self, n_jobs=2):
        self.n_jobs = n_jobs  # GitHub CI has two nodes.

    name = "scikit-learn forest"
    param_names = ["n", "n_estimators", "max_features"]
    # params=([1000, 10000, 100000], [100, 400], [4, 12], [2])
    params = ([10000], [100], [4, 12])

    def _setup_model(self, params):
        _, n_estimators, max_features = params
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=8,
            max_features=max_features,
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
    param_names = ["n", "n_estimators", "max_features"]
    params = ([1000, 10000, 100000], [100], [4, 12])

    def _setup_model(self, params):
        _, n_estimators, max_features = params

        # inspect.getargspec(RandomForest.__init__) does not work
        try:
            self.model = RandomForest(
                max_depth=8,
                n_jobs=self.n_jobs,
                max_features=max_features,
                n_estimators=n_estimators,
            )
        # For biosphere<0.3.0, max_features was called mtry and n_trees was called
        # n_estimators.
        except TypeError:
            try:
                self.model = RandomForest(
                    max_depth=8,
                    n_jobs=self.n_jobs,
                    mtry=max_features,
                    n_trees=n_estimators,
                )
            # For biosphere<0.2.0, no parallelization
            except TypeError:
                self.model = RandomForest(max_depth=8, mtry=max_features)

    def _time_fit_predict_oob(self):
        predictions = self.model.fit_predict_oob(self.X_train, self.y_train)
        return mean_squared_error(self.y_train, predictions)
