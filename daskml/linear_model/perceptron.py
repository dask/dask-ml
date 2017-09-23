from sklearn.linear_model import Perceptron as _Perceptron

from daskml.base import _BigPartialFitMixin


class BigPerceptron(_BigPartialFitMixin, _Perceptron):
    _init_kwargs = ['classes']
    _fit_kwargs = ['classes']
