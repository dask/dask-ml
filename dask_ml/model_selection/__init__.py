"""Utilities for hyperparameter optimization.

These estimators will operate in parallel. Their scalability depends
on the underlying estimators being used.
"""
from ._search import GridSearchCV, RandomizedSearchCV, compute_n_splits, check_cv
from ._split import ShuffleSplit, KFold, train_test_split


__all__ = [
    "GridSearchCV",
    "RandomizedSearchCV",
    "ShuffleSplit",
    "KFold",
    "train_test_split",
    "compute_n_splits",
    "check_cv",
]


try:
    from ._incremental import IncrementalSearchCV
    from ._hyperband import HyperbandSearchCV
    from ._successive_halving import SuccessiveHalvingSearchCV

    __all__.extend(
        ["IncrementalSearchCV", "HyperbandSearchCV", "SuccessiveHalvingSearchCV"]
    )
except ImportError:
    pass
