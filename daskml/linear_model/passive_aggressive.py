from sklearn import linear_model as _lm

from daskml.base import _BigPartialFitMixin, _copy_partial_doc


@_copy_partial_doc
class BigPassiveAggressiveClassifier(_BigPartialFitMixin,
                                     _lm.PassiveAggressiveClassifier):
    _init_kwargs = _fit_kwargs = ['classes']


@_copy_partial_doc
class BigPassiveAggressiveRegressor(_BigPartialFitMixin,
                                    _lm.PassiveAggressiveRegressor):
    pass
