from sklearn import linear_model as _lm

from .._partial import _BigPartialFitMixin, _copy_partial_doc


@_copy_partial_doc
class PartialPassiveAggressiveClassifier(
    _BigPartialFitMixin, _lm.PassiveAggressiveClassifier
):
    _init_kwargs = _fit_kwargs = ["classes"]


@_copy_partial_doc
class PartialPassiveAggressiveRegressor(
    _BigPartialFitMixin, _lm.PassiveAggressiveRegressor
):
    pass
