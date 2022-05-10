"""
$ python dump_cprofile_stats.py
$ gprof2dot -f pstats ./fanova-profile.stats | dot -Tpng -o profile.png
"""
import cProfile
import os

import optuna

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


def main():
    study = create_study("profile-fanova.db")

    profiler = cProfile.Profile()
    evaluator = FastFanovaImportanceEvaluator(n_trees=16, seed=0)
    profiler.runcall(optuna.importance.get_param_importances, study, evaluator=evaluator)
    profiler.dump_stats("./fanova-profile.stats")


if __name__ == "__main__":
    main()
