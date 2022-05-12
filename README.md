## optuna-fast-fanova

optuna-fast-fanova provides Cython-accelerated version of [FanovaImportanceEvaluator](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html#optuna.importance.FanovaImportanceEvaluator).

| n_trials | n_params | n_trees      | fANOVA (Optuna) | fast-fanova     |
|----------|----------|--------------|-----------------|-----------------|
| 1000     | 32       | 64 (default) | 86.476s         | 8.325s (-90.4%) |
| 1000     | 8        | 64 (default) | 112.277s        | 7.397s (-93.4%) |
| 1000     | 2        | 64 (default) | 64.795s         | 4.503s (-93.1%) |
| 100      | 32       | 64 (default) | 2.406s          | 0.769s (-68.0%) |
| 100      | 8        | 64 (default) | 2.091s          | 0.578s (-72.3%) |
| 100      | 2        | 64 (default) | 1.695s          | 0.381s (-77.5%) |
| 1000     | 32       | 32           | 44.318s         | 4.805s (-89.2%) |
| 1000     | 8        | 32           | 56.309s         | 3.718s (-93.4%) |
| 1000     | 2        | 32           | 32.406s         | 2.654s (-91.8%) |
| 100      | 32       | 32           | 1.191s          | 0.676s (-43.2%) |
| 100      | 8        | 32           | 1.114s          | 0.292s (-73.8%) |
| 100      | 2        | 32           | 0.815s          | 0.237s (-70.9%) |

[The benchmark script](./tools/benchmark.py) was run on my laptop so the times should not be taken precisely.

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
