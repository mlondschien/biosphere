import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from biosphere import RandomForest

from .common import Benchmark


class ScikitLearnForest(Benchmark):
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
            n_jobs=2,
            oob_score=True,
        )

    def _time_fit_predict_oob(self):
        self.oob_score = True
        self.model.fit(self.X_train, self.y_train)
        return (1 - self.model.oob_score_) * np.mean(
            (self.y_train - self.y_train.mean()) ** 2
        )


class BiosphereForest(Benchmark):
    name = "biosphere forest"
    param_names = ["n", "cat", "n_estimators", "mtry"]
    params = ([1000, 10000, 100000], [True, False], [100], [4, 12])
    # params=([10000], [100, 400], [12], [1])

    def _setup_model(self, params):
        _, _, n_estimators, mtry = params
        self.model = RandomForest(
            n_trees=n_estimators, max_depth=8, mtry=mtry, n_jobs=2,
        )

    def _time_fit_predict_oob(self):
        predictions = self.model.fit_predict_oob(self.X_train, self.y_train)
        return mean_squared_error(self.y_train, predictions)
