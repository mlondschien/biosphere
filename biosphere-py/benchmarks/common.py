import timeit
from abc import ABC, abstractmethod
from typing import Any

from sklearn.metrics import mean_squared_error

from .load import load_nyc_taxi


class Benchmark(ABC):
    # Measure wall time instead of CPU usage
    timer = timeit.default_timer

    name = None

    params: Any
    param_names: Any

    def _time_predict(self, *args):
        self.model.predict(self.X_test)

    def time_fit(self, *args):
        self.model.fit(self.X_train, self.y_train)

    def setup(self, *params):
        self._setup_data(params)
        self._setup_model(params)

    def _setup_data(self, params):
        idx = self.param_names.index("n")
        n = params[idx]
        self.X_train, self.X_test, self.y_train, self.y_test = load_nyc_taxi(n, n)

    @abstractmethod  # Required for asv to skip.
    def _setup_model(self, params):
        pass

    def score(self):
        return mean_squared_error(self.model.predict(self.X_test), self.y_test,)
