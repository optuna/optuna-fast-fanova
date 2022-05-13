## optuna-fast-fanova

optuna-fast-fanova provides Cython-accelerated version of [FanovaImportanceEvaluator](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html#optuna.importance.FanovaImportanceEvaluator).

| n_trials | n_params | n_trees      | fANOVA (Optuna) | fast-fanova     |
|----------|----------|--------------|-----------------|-----------------|
| 1000     | 32       | 64 (default) | 72.326s         | 3.191s (-95.6%) |
| 1000     | 8        | 64 (default) | 94.002s         | 2.493s (-97.3%) |
| 1000     | 2        | 64 (default) | 52.941s         | 1.472s (-97.2%) |
| 1000     | 32       | 32           | 36.452s         | 1.641s (-95.5%) |
| 1000     | 8        | 32           | 46.983s         | 1.258s (-97.3%) |
| 1000     | 2        | 32           | 26.843s         | 0.770s (-97.1%) |
| 100      | 32       | 64 (default) | 1.721s          | 0.329s (-80.9%) |
| 100      | 8        | 64 (default) | 1.673s          | 0.156s (-90.7%) |
| 100      | 2        | 64 (default) | 1.255s          | 0.114s (-90.9%) |
| 100      | 32       | 32           | 0.899s          | 0.170s (-81.1%) |
| 100      | 8        | 32           | 0.836s          | 0.081s (-90.3%) |
| 100      | 2        | 32           | 0.644s          | 0.059s (-90.9%) |

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
