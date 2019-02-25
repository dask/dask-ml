"""The ``dask_ml.linear_model`` module implements linear models for
classification and regression.
"""
from .glm import LinearRegression, LogisticRegression, PoissonRegression
from .passive_aggressive import (
    PartialPassiveAggressiveClassifier,
    PartialPassiveAggressiveRegressor,
)
from .perceptron import PartialPerceptron
from .stochastic_gradient import PartialSGDClassifier, PartialSGDRegressor

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
