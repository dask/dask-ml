import functools

from sklearn.model_selection import cross_validate as _cross_validate
from sklearn.model_selection import cross_val_score as _cross_val_score
from sklearn.model_selection import cross_val_predict as _cross_val_predict
from dask.base import is_dask_collection

from dask_ml import joblib


def _joblibify(f):
    """
    Allows execution of functions using joblib('dask') as backend.

    :param f: A function with signature ``f(something, X, y=None, **kwargs)``
    :return: A function that runs on dask
    """
    @functools.wraps(f)
    def wrap(*args, **kwargs):
        X = args[1]
        y = kwargs.get("y")

        if not y and len(args) == 3:
            y = args[2]

        if not y:  # unsupervised
            scatter = [X]
        else:
            scatter = [X, y]

        if is_dask_collection(X) or is_dask_collection(y):
            message = ('Dask collections are not supported '
                       'by {} yet. '.format(f.__name__))
            message += ('You can explicily compute them (if you data fit '
                        'in memory) through ``X, y = dask.compute(X, y)``')
            raise NotImplementedError(message)

        with joblib.parallel_backend("dask", scatter=scatter):
            return f(*args, **kwargs)

    return wrap


cross_validate = _joblibify(_cross_validate)
cross_val_score = _joblibify(_cross_val_score)
cross_val_predict = _joblibify(_cross_val_predict)
