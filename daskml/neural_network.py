from sklearn import neural_network as _nn

from daskml.base import _BigPartialFitMixin


class BigMLPClassifier(_BigPartialFitMixin, _nn.MLPClassifier):
    _init_kwargs = _fit_kwargs = ['classes']


class BigMLPRegressor(_BigPartialFitMixin, _nn.MLPRegressor):
    pass
