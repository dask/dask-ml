"""The ``daskml.linear_model`` module implements linear models for
classification and regression.
"""
from .stochastic_gradient import PartialSGDClassifier, PartialSGDRegressor  # noqa
from .perceptron import PartialPerceptron  # noqa
from .passive_aggressive import PartialPassiveAggressiveClassifier, PartialPassiveAggressiveRegressor # noqa

from dask_glm.estimators import LogisticRegression, LinearRegression, PoissonRegression  # noqa
