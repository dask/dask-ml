from sklearn.linear_model import stochastic_gradient as _sg

from .._partial import _BigPartialFitMixin, _copy_partial_doc


@_copy_partial_doc
class PartialSGDClassifier(_BigPartialFitMixin, _sg.SGDClassifier):

    _init_kwargs = ["classes"]
    _fit_kwargs = ["classes"]


@_copy_partial_doc
class PartialSGDRegressor(_BigPartialFitMixin, _sg.SGDRegressor):
    pass
