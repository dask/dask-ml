import sklearn.linear_model

from .._partial import _BigPartialFitMixin, _copy_partial_doc


@_copy_partial_doc
class PartialSGDClassifier(_BigPartialFitMixin, sklearn.linear_model.SGDClassifier):

    _init_kwargs = ["classes"]
    _fit_kwargs = ["classes"]


@_copy_partial_doc
class PartialSGDRegressor(_BigPartialFitMixin, sklearn.linear_model.SGDRegressor):
    pass


__all__ = ["PartialSGDClassifier", "PartialSGDRegressor"]
