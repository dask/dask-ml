"""The ``dask_ml.linear_model`` module implements linear models for
classification and regression.
"""
from .stochastic_gradient import PartialSGDClassifier, PartialSGDRegressor
from .perceptron import PartialPerceptron
from .passive_aggressive import (
    PartialPassiveAggressiveClassifier,
    PartialPassiveAggressiveRegressor,
)
from .glm import LogisticRegression, LinearRegression, PoissonRegression

__all__ = [
    "PartialPassiveAggressiveClassifier",
    "PartialPassiveAggressiveRegressor",
    "PartialPerceptron",
    "PartialSGDClassifier",
    "PartialSGDRegressor",
    "LogisticRegression",
    "LinearRegression",
    "PoissonRegression",
]
