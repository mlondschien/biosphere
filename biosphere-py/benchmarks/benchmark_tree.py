from sklearn.tree import DecisionTreeRegressor

from biosphere import DecisionTree

from .common import Benchmark


class BiosphereTree(Benchmark):
    name = "biosphere tree"
    param_names = ["n", "max_features"]
    params = ([1000, 10000, 100000, 1000000], [4, 12])

    def _setup_model(self, params):
        _, max_features = params

        # inspect.getargspec(DecisionTree.__init__) does not work
        try:
            self.model = DecisionTree(max_depth=8, max_features=max_features,)
        # For biosphere<0.3.0, max_features was called mtry
        except TypeError:
            self.model = DecisionTree(max_depth=8, mtry=max_features)


class ScikitLearnTree(Benchmark):
    name = "scikit-learn tree"
    param_names = ["n", "max_fetures"]
    params = ([1000, 10000, 100000, 1000000], [4, 12])

    def _setup_model(self, params):
        _, max_features = params
        self.model = DecisionTreeRegressor(max_depth=8, max_features=max_features)
