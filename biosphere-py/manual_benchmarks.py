from .benchmarks.benchmark_forest import ScikitLearnForest, BiosphereForest
import time
import pandas as pd

benchmark_parameters = [
    (1000, 400, 12, 4),
    (2000, 400, 12, 4),
    (4000, 400, 12, 4),
    (8000, 400, 12, 4),
    (16000, 400, 12, 4),
    (32000, 400, 12, 4),
    (64000, 400, 12, 4),
    (128000, 400, 12, 4),
    (256000, 400, 12, 4),
    (512000, 400, 12, 4),
    (1024000, 400, 12, 4),
    (2048000, 400, 12, 4),
]

models = [ScikitLearnForest, BiosphereForest]

n_samples = 10

if __name__ == "__main__":
    results = pd.DataFrame(
        columns=["model", "n", "mtry", "n_estimators", "n_jobs", "time", "score", "oob_score"]
    )

    for parameters in benchmark_parameters:
        for model in models:
            m = model()
            m.setup(parameters)

            for _ in range(n_samples):
                tic = time.perf_counter_ns()
                oob_score = m._time_fit_predict_oob()
                toc = time.perf_counter_ns()

                score = m.score()
                results.loc[len(results)] = [m.name, *parameters, toc - tic, score, oob_score]
            
            print(f"{model.name} {parameters} time={results.tail(n_samples)['time'].mean()/1e9:.4f} score={score:.4f} oob_score={oob_score:.4f}")

    results.to_csv("results.csv")