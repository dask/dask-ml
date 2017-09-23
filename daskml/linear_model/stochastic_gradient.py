from sklearn.linear_model import stochastic_gradient as _sg

from daskml.base import _BigPartialFitMixin


class BigSGDClassifier(_BigPartialFitMixin, _sg.SGDClassifier):

    _init_kwargs = ['classes']
    _fit_kwargs = ['classes']


class BigSGDRegressor(_BigPartialFitMixin, _sg.SGDRegressor):
    pass
