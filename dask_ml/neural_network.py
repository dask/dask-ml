from sklearn import neural_network as _nn

from ._partial import _BigPartialFitMixin, _copy_partial_doc


@_copy_partial_doc
class ParitalMLPClassifier(_BigPartialFitMixin, _nn.MLPClassifier):
    _init_kwargs = _fit_kwargs = ["classes"]


@_copy_partial_doc
class ParitalMLPRegressor(_BigPartialFitMixin, _nn.MLPRegressor):
    pass
