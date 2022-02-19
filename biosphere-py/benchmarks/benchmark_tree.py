from sklearn.tree import DecisionTreeRegressor

from biosphere import DecisionTree

from .common import Benchmark


class BiosphereTree(Benchmark):
    name = "biosphere tree"
    param_names = ["n", "mtry"]
    params = ([1000, 10000, 100000, 1000000], [4, 12])

    def _setup_model(self, params):
        _, mtry = params
        self.model = DecisionTree(max_depth=8, mtry=mtry)


class ScikitLearnTree(Benchmark):
    name = "scikit-learn tree"
    param_names = ["n", "mtry"]
    params = ([1000, 10000, 100000, 1000000], [4, 12])

    def _setup_model(self, params):
        _, mtry = params
        self.model = DecisionTreeRegressor(max_depth=8, max_features=mtry,)
