## optuna-fast-fanova

optuna-fast-fanova provides Cython-accelerated version of [FanovaImportanceEvaluator](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html#optuna.importance.FanovaImportanceEvaluator).

### Installation

Supported Python versions are 3.7 or later.

```
$ pip install optuna-fast-fanova
```

### Usage

Usage is like this:

```python
import optuna
from fast_fanova import FanovaImportanceEvaluator


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    return x**2 + y


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=1000)

    importance = optuna.importance.get_param_importances(
        study, evaluator=FanovaImportanceEvaluator()
    )
    print(importance)
```

You can use optuna-fast-fanova in only two steps.

1. Add an import statement: `from fast_fanova import FanovaImportanceEvaluator`.
2. Pass a `FanovaImportanceEvaluator()` object to an `evaluator` argument of `get_param_importances()` function.

| trials / params | fANOVA (Optuna) | fast-fanova | 
|-----------------|-----------------|-------------|
| 1000 / 4        | 60.592s         | 4.690s      |
| 1000 / 4        | 30.515s         | 2.393s      |
| 1000 / 4        | 15.297s         | 1.225s      |

[The benchmark script](./benchmark.py) was run on my laptop (M1 Pro Macbook) so the times should not be taken precisely.
This library seems to be about more than 10-times faster than Optuna's FanovaImportanceExecutor (with Optuna v3.0.0b0).


## How to cite fANOVA

This is a derived work of https://github.com/automl/fanova.
For how to cite the original work, please refer to https://automl.github.io/fanova/cite.html.
