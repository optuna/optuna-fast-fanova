import optuna

from optuna_fast_fanova import FanovaImportanceEvaluator


# from optuna.importance import FanovaImportanceEvaluator

optuna.logging.set_verbosity(optuna.logging.ERROR)


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -100, 100)
    z = trial.suggest_categorical("z", ["foo", "bar", "baz"])
    return x**2 + y + len(z)


def main():
    study = optuna.create_study(study_name="foo")
    study.optimize(objective, n_trials=20)

    evaluator = FanovaImportanceEvaluator(n_trees=4, max_depth=16, seed=0)
    importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
    print(importance)


if __name__ == "__main__":
    main()
