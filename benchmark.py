import os
import time

import optuna
from optuna.importance import FanovaImportanceEvaluator

from fast_fanova import FastFanovaImportanceEvaluator


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_float("y", -1, 1)
    z = trial.suggest_int("z", -10, 10)
    foo = trial.suggest_categorical("foo", ["foo", "bar"])
    return x**2 + y + z + len(foo)


def create_study(filepath):
    db_url = f"sqlite:///{filepath}"
    if os.path.exists(filepath):
        return optuna.load_study(storage=db_url, study_name="foo")

    study = optuna.create_study(storage=db_url, study_name="foo")
    study.optimize(objective, n_trials=1000)
    return study


if __name__ == "__main__":
    study = create_study("profile-fanova.db")
    print("n_trials", study._storage.get_n_trials(study._study_id))

    for n_trees in [1, 4, 8, 16, 32, 64]:
        for (name, evaluator_cls) in [
            # ("MeanDecreaseImpurity", MeanDecreaseImpurityImportanceEvaluator),
            ("Fanova (Optuna)     ", FanovaImportanceEvaluator),
            ("Fanova (FastFanova) ", FastFanovaImportanceEvaluator),
        ]:
            start = time.time()
            importances = optuna.importance.get_param_importances(
                study, evaluator=evaluator_cls(n_trees=n_trees, seed=0)
            )
            elapsed = time.time() - start
            print(f"{name}\tn_trees={n_trees}\t{elapsed:.3f}s\t{dict(importances)}")
