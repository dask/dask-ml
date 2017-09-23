from sklearn import linear_model as _lm

from daskml.base import _BigPartialFitMixin


class BigPassiveAggressiveClassifier(_BigPartialFitMixin,
                                     _lm.PassiveAggressiveClassifier):
    _init_kwargs = _fit_kwargs = ['classes']


class BigPassiveAggressiveRegressor(_BigPartialFitMixin,
                                    _lm.PassiveAggressiveRegressor):
    pass
