"""Utilities for hyperparameter optimization.

These estimators will operate in parallel. Their scalability depends
on the underlying estimators being used.
"""
from warnings import warn

from ._search import GridSearchCV, RandomizedSearchCV, compute_n_splits, check_cv
from ._split import ShuffleSplit, KFold, train_test_split
try:
    import distributed
except ImportError:
    warn(
        "The Dask Distributed library has not been found. The IncrementalSearchCV"
        "class and it's children depend on it. For installation instructions, "
        "see http://docs.dask.org/en/latest/install.html", ImportWarning
    )


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
    from ._incremental import IncrementalSearchCV  # noqa: F401

    __all__.extend(["IncrementalSearchCV"])
except ImportError as e:
    pass
