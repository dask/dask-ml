"""The ``dask_ml.linear_model`` module implements linear models for
classification and regression.
"""
from .glm import LinearRegression, LogisticRegression, PoissonRegression

__all__ = [
    "LogisticRegression",
    "LinearRegression",
    "PoissonRegression",
]
