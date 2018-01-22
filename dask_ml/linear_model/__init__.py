"""The ``dask_ml.linear_model`` module implements linear models for
classification and regression.
"""
from .stochastic_gradient import PartialSGDClassifier, PartialSGDRegressor  # noqa
from .perceptron import PartialPerceptron  # noqa
from .passive_aggressive import PartialPassiveAggressiveClassifier, PartialPassiveAggressiveRegressor # noqa
from .glm import LogisticRegression, LinearRegression, PoissonRegression  # noqa
