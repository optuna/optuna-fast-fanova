import itertools
import math
import time

import optuna
from optuna.importance import FanovaImportanceEvaluator

from optuna_fast_fanova import FanovaImportanceEvaluator as FastFanovaImportanceEvaluator

optuna.logging.set_verbosity(optuna.logging.ERROR)


def run_optimize(storage, trials, params):
    study_name = f"study-trial{trials}-params{params}"
    study = optuna.create_study(storage=storage, study_name=study_name)

    def objective(trial: optuna.Trial):
        val = 0
        for i in range(params):
            xi = trial.suggest_float(str(i), -4, 4)
            val += (xi - 2) ** 2
        return val

    study.optimize(objective, n_trials=trials)
    return study


def print_markdown_table(results):
    print("| n_trials | n_params | n_trees | fANOVA (Optuna) | fast-fanova |")
    print("| -------- | -------- | ------- | --------------- | ----------- |")

    for n_trials, n_params, n_trees, s1, s2 in results:
        print(f"| {n_trials} | {n_params} | {n_trees} | {s1:.3f}s | {s2:.3f}s (-{s1-s2/s1*100:.1f}%) |")


def is_importance_close(a, b):
    assert len(a) == len(b)
    for k1, k2 in zip(a, b):
        assert k1 == k2
        assert math.isclose(a[k1], b[k2])


def main():
    storage = "sqlite:///benchmark-fanova.db"
    results = []
    for n_trials, n_params in itertools.product([1000, 100], [32, 8, 2]):
        study = run_optimize(storage, n_trials, n_params)
        for n_trees in [32, 64]:
            start = time.time()
            importances_before = optuna.importance.get_param_importances(
                study, evaluator=FanovaImportanceEvaluator(n_trees=n_trees, seed=0)
            )
            elapsed_before = time.time() - start

            start = time.time()
            importances_after = optuna.importance.get_param_importances(
                study, evaluator=FastFanovaImportanceEvaluator(n_trees=n_trees, seed=0)
            )
            elapsed_after = time.time() - start

            print(f"Before: n_trees={n_trees} elapsed={elapsed_before:.3f}\t{dict(importances_before)}")
            print(f"After:  n_trees={n_trees} elapsed={elapsed_after:.3f}\t{importances_after}")
            is_importance_close(importances_before, importances_after)

            results.append((n_trials, n_params, n_trees, elapsed_before, elapsed_after))
    print_markdown_table(results)


if __name__ == '__main__':
    main()
