## optuna-fast-fanova

optuna-fast-fanova provides Cython-accelerated version of [FanovaImportanceEvaluator](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html#optuna.importance.FanovaImportanceEvaluator).


| n_trials | n_params | n_trees      | fANOVA (Optuna) | fast-fanova     |
|----------|----------|--------------|-----------------|-----------------|
| 1000     | 32       | 64 (default) | 71.044s         | 6.198s (-91.3%) |
| 1000     | 8        | 64 (default) | 92.360s         | 5.836s (-93.7%) |
| 1000     | 2        | 64 (default) | 51.937s         | 2.978s (-94.3%) |
| 1000     | 32       | 32           | 35.852s         | 3.147s (-91.2%) |
| 1000     | 8        | 32           | 46.271s         | 2.938s (-93.7%) |
| 1000     | 2        | 32           | 26.196s         | 1.497s (-94.3%) |
| 100      | 32       | 64 (default) | 1.653s          | 0.411s (-75.2%) |
| 100      | 8        | 64 (default) | 1.646s          | 0.241s (-85.4%) |
| 100      | 2        | 64 (default) | 1.232s          | 0.177s (-85.6%) |
| 100      | 32       | 32           | 0.878s          | 0.216s (-75.4%) |
| 100      | 8        | 32           | 0.816s          | 0.130s (-84.0%) |
| 100      | 2        | 32           | 0.627s          | 0.091s (-85.4%) |

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
