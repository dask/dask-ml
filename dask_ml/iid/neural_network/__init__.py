import sklearn.neural_network

from ...base import _make_estimator, _find_estimators

__all__ = []
_models = _find_estimators(sklearn.neural_network)

for _model in _models:
    _name = _model.__name__
    globals()[_name] = _make_estimator(_model)
    __all__.append(_name)
