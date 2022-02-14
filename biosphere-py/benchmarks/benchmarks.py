from biosphere import RandomForest
from sklearn.ensemble import RandomForestRegressor
from load import load_nyc_taxi
from sklearn.metrics import mean_squared_error
import pandas as pd

import time


class Benchmark:
    name = None

    def setup(self, n, d, mtry, n_estimators, max_depth, n_jobs):
        self.setup_data(n)
        self.setup_model(n_estimators, mtry, max_depth, n_jobs)

    def setup_data(self, n):
        self.X_train, self.X_test, self.y_train, self.y_test = load_nyc_taxi(n, n)

    def setup_model(self, n_estimators, mtry, max_depth, n_jobs):
        raise NotImplementedError()

    def benchmark(self):
        return self.model.fit(self.X_train, self.y_train)

    def score(self):
        return mean_squared_error(
            self.model.predict(self.X_test),
            self.y_test,
        )


class ScikitLearn(Benchmark):
    name = "scikit-learn"

    def setup_model(self, n_estimators, mtry, max_depth, n_jobs):
        if n_jobs is None:
            n_jobs = -1

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=mtry,
            n_jobs=n_jobs,
        )

    def benchmark(self):
        return self.model.fit(self.X_train, self.y_train)


class Biosphere(Benchmark):
    name = "biosphere"

    def setup_model(self, n_estimators, mtry, max_depth, n_jobs):
        self.model = RandomForest(
            n_trees=n_estimators,
            max_depth=max_depth,
            mtry=mtry,
            n_jobs=n_jobs,
        )


benchmark_parameters = [
    (1000, None, 12, 400, 8, 4),
    (2000, None, 12, 400, 8, 4),
    (4000, None, 12, 400, 8, 4),
    (8000, None, 12, 400, 8, 4),
    (16000, None, 12, 400, 8, 4),
    (32000, None, 12, 400, 8, 4),
    (64000, None, 12, 400, 8, 4),
    (128000, None, 12, 400, 8, 4),
    (256000, None, 12, 400, 8, 4),
    (512000, None, 12, 400, 8, 4),
    (1024000, None, 12, 400, 8, 4),
    (2048000, None, 12, 400, 8, 4),
    (4096000, None, 12, 400, 8, 4),
]

models = [ScikitLearn, Biosphere]

n_samples = 1

if __name__ == "__main__":
    results = pd.DataFrame(
        columns=[
            "model",
            "n",
            "d",
            "mtry",
            "n_estimators",
            "max_depth",
            "n_jobs",
            "time",
            "score",
        ]
    )

    for parameters in benchmark_parameters:
        for model in models:
            m = model()
            m.setup(*parameters)

            for _ in range(n_samples):
                tic = time.perf_counter_ns()
                m.benchmark()
                toc = time.perf_counter_ns()
                score = m.score()

                results.loc[len(results)] = [m.name, *parameters, toc - tic, score]
            
            print(f"{model.name} {parameters} score={score:.4f} time={results.tail(n_samples)['time'].mean()/1e9:.4f}")

    results.to_csv("results.csv")