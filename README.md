## optuna-fast-fanova

optuna-fast-fanova provides Cython-accelerated version of [FanovaImportanceEvaluator](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html#optuna.importance.FanovaImportanceEvaluator).

| n_trials | n_params | n_trees      | fANOVA (Optuna) | fast-fanova     |
|----------|----------|--------------|-----------------|-----------------|
| 1000     | 32       | 64 (default) | 71.431s         | 2.963s (-95.9%) |
| 1000     | 8        | 64 (default) | 92.307s         | 2.315s (-97.5%) |
| 1000     | 2        | 64 (default) | 52.295s         | 1.297s (-97.5%) |
| 1000     | 32       | 32           | 36.069s         | 1.526s (-95.8%) |
| 1000     | 8        | 32           | 46.340s         | 1.174s (-97.5%) |
| 1000     | 2        | 32           | 26.278s         | 0.677s (-97.4%) |
| 100      | 32       | 64 (default) | 1.668s          | 0.306s (-81.6%) |
| 100      | 8        | 64 (default) | 1.652s          | 0.138s (-91.7%) |
| 100      | 2        | 64 (default) | 1.242s          | 0.095s (-92.4%) |
| 100      | 32       | 32           | 0.889s          | 0.161s (-81.9%) |
| 100      | 8        | 32           | 0.830s          | 0.075s (-90.9%) |
| 100      | 2        | 32           | 0.631s          | 0.048s (-92.3%) |

[The benchmark script](./tools/benchmark.py) was run on my laptop (Macbook M1 Pro) so the times should not be taken precisely.

### Installation

Supported Python versions are 3.7 or later.

```
$ pip install optuna-fast-fanova
```

### Usage

Usage is like this:

```python
import optuna
from optuna_fast_fanova import FanovaImportanceEvaluator


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    return x ** 2 + y


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=1000)

    importance = optuna.importance.get_param_importances(
        study, evaluator=FanovaImportanceEvaluator()
    )
    print(importance)
```

You can use optuna-fast-fanova in only two steps.

1. Add an import statement: `from optuna_fast_fanova import FanovaImportanceEvaluator`.
2. Pass a `FanovaImportanceEvaluator()` object to an `evaluator` argument of `get_param_importances()` function.

## How to cite fANOVA

This is a derived work of https://github.com/automl/fanova.
For how to cite the original work, please refer to https://automl.github.io/fanova/cite.html.
