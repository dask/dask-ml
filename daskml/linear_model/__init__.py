"""The ``daskml.linear_model`` module implements linear models for
classification and regression.
"""
from .stochastic_gradient import BigSGDClassifier, BigSGDRegressor  # noqa
from .perceptron import BigPerceptron  # noqa
from .passive_aggressive import BigPassiveAggressiveClassifier, BigPassiveAggressiveRegressor # noqa

from dask_glm.estimators import LogisticRegression, LinearRegression, PoissonRegression  # noqa
