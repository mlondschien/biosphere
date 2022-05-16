# Benchmarks for the README
import time

import pandas as pd

from benchmarks.benchmark_forest import BiosphereForest, ScikitLearnForest

benchmark_parameters = [
    (1000, 400, 12),
    (2000, 400, 12),
    (4000, 400, 12),
    (8000, 400, 12),
    (16000, 400, 12),
    (32000, 400, 12),
    (64000, 400, 12),
    (128000, 400, 12),
    (256000, 400, 12),
    (512000, 400, 12),
    (1024000, 400, 12),
    (2048000, 400, 12),
]

models = [ScikitLearnForest, BiosphereForest]

n_samples = 10

if __name__ == "__main__":
    results = pd.DataFrame(
        columns=["model", "n", "mtry", "n_estimators", "time", "score", "oob_score"]
    )

    for parameters in benchmark_parameters:
        for model in models:
            m = model(n_jobs=4)
            m.setup(*parameters)

            for _ in range(n_samples):
                tic = time.perf_counter_ns()
                oob_score = m._time_fit_predict_oob()
                toc = time.perf_counter_ns()

                score = m.score()
                results.loc[len(results)] = [
                    m.name,
                    *parameters,
                    toc - tic,
                    score,
                    oob_score,
                ]

            print(
                f"{model.name} {parameters} "
                f"time={results.tail(n_samples)['time'].min()/1e9:.4f}"
                f" score={score:.4f} oob_score={oob_score:.4f}"
            )

    results.to_csv("results.csv")
