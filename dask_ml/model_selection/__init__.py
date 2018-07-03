"""Utilities for hyperparameter optimization.

These estimators will operate in parallel. Their scalability depends
on the underlying estimators being used.
"""
from ._search import (
    GridSearchCV, RandomizedSearchCV,
    compute_n_splits, check_cv
)
from ._split import ShuffleSplit, train_test_split
from ._validate import cross_validate, cross_val_score, cross_val_predict


__all__ = [
    'GridSearchCV',
    'RandomizedSearchCV',
    'ShuffleSplit',
    'train_test_split',
    'compute_n_splits',
    'check_cv',
    'cross_validate',
    'cross_val_score',
    'cross_val_predict',
]
