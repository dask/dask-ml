from __future__ import absolute_import, print_function, division

import numpy as np
from dask.base import tokenize
from sklearn.base import clone, is_classifier

from .wrapped import Wrapped, _fit
from .utils import unpack_as_lists_of_keys, check_X_y


def merge_estimators(estimators):
    """Merge several estimators together by averaging.

    Performs a simple average of `coef_` and `intercept_` across all
    estimators. For classifiers, the classes are also unioned.

    Parameters
    ----------
    estimators : list
        A list of estimators to merge

    Returns
    -------
    merged : estimator
        The merged estimator.
    """
    if len(estimators) == 1:
        return estimators[0]
    o = clone(estimators[0])
    if is_classifier(o):
        classes = np.unique(np.concatenate([m.classes_ for m in estimators]))
        o.classes_ = classes
        n_classes = len(classes)
        if all(m.classes_.size == n_classes for m in estimators):
            o.coef_ = np.mean([m.coef_ for m in estimators], axis=0)
            o.intercept_ = np.mean([m.intercept_ for m in estimators], axis=0)
        else:
            # Not all estimators got all classes. Multiclass problems result in
            # a row per class. Here we average the coefficients for each class,
            # using zero if that class wasn't fit.
            n_features = estimators[0].coef_.shape[1]
            coef = np.zeros((n_classes, n_features), dtype='f8')
            intercept = np.zeros((n_classes, 1), dtype='f8')
            for m in estimators:
                if len(m.classes_) == 2:
                    i = [m.classes_[1]]
                else:
                    i = np.in1d(classes, m.classes_)
                coef[i] += m.coef_
                intercept[i] += m.intercept_
            o.coef_ = coef / len(estimators)
            o.intercept_ = intercept / len(estimators)
    else:
        o.coef_ = np.mean([m.coef_ for m in estimators], axis=0)
        o.intercept_ = np.mean([m.intercept_ for m in estimators], axis=0)
    return o


class Averaged(Wrapped):
    def fit(self, X, y, **kwargs):
        # Remove classes, for signature compatibility with Chained
        kwargs.pop('classes', None)
        X, y = check_X_y(X, y)
        name = 'fit-' + tokenize(self, X, y, kwargs)
        x_parts, y_parts, dsk = unpack_as_lists_of_keys(X, y)
        keys = [(name, i) for i in range(len(x_parts))]
        for k, x, y in zip(keys, x_parts, y_parts):
            dsk[k] = (_fit, self._name, x, y, kwargs)
        dsk[name] = (merge_estimators, keys)
        dsk.update(self.dask)
        return Averaged(clone(self._est), dask=dsk, name=name)

    def _fit_transform(self, X, y, **kwargs):
        fit = self.fit(X, y, **kwargs)
        Xt = fit.transform(X)
        return fit, Xt
