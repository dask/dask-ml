from sklearn.linear_model import stochastic_gradient as _sg

from daskml.base import _BigPartialFitMixin, _copy_partial_doc


@_copy_partial_doc
class BigSGDClassifier(_BigPartialFitMixin, _sg.SGDClassifier):

    _init_kwargs = ['classes']
    _fit_kwargs = ['classes']


@_copy_partial_doc
class BigSGDRegressor(_BigPartialFitMixin, _sg.SGDRegressor):
    pass
