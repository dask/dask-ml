from __future__ import absolute_import, print_function, division

import numpy as np
from dask.base import tokenize
from dask.delayed import delayed
from sklearn.base import clone, is_classifier

from .wrapped import Wrapped
from .utils import unpack_arguments, unpack_as_lists_of_keys, check_X_y


_unique_chunk = delayed(np.unique, pure=True)


@delayed(pure=True)
def _unique_merge(x):
    return np.unique(np.concatenate(x))


def _partial_fit(est, X, y, classes, kwargs):
    # XXX: this mutates est!
    if classes is None:
        return est.partial_fit(X, y, **kwargs)
    return est.partial_fit(X, y, classes=classes, **kwargs)


class Chained(Wrapped):
    def __init__(self, est, dask=None, name=None):
        super(Chained, self).__init__(est, dask=dask, name=name)
        if not hasattr(est, 'partial_fit'):
            raise ValueError("estimator must support `partial_fit`")

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y)
        name = 'partial_fit-' + tokenize(self, X, y, **kwargs)
        x_parts, y_parts, dsk = unpack_as_lists_of_keys(X, y)

        # Extract classes if applicable
        if is_classifier(self):
            classes = kwargs.pop('classes', None)
            if classes is None:
                classes = _unique_merge([_unique_chunk(i) for i in y_parts])
            classes, dsk2 = unpack_arguments(classes)
            dsk.update(dsk2)
        else:
            classes = None

        # Clone so that this estimator isn't mutated
        sk_est = clone(self._est)

        dsk[(name, 0)] = (_partial_fit, sk_est, x_parts[0], y_parts[0],
                          classes, kwargs)

        for i, (x, y) in enumerate(zip(x_parts[1:], y_parts[1:]), 1):
            dsk[(name, i)] = (_partial_fit, (name, i - 1), x, y, None, kwargs)
        return Chained(clone(sk_est), dsk, (name, len(x_parts) - 1))

    def _fit_transform(self, X, y, **kwargs):
        fit = self.fit(X, y, **kwargs)
        Xt = fit.transform(X)
        return fit, Xt
