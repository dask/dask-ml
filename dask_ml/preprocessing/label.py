from __future__ import division

from operator import getitem

import dask.array as da
import dask.dataframe as dd
import numpy as np
from sklearn.preprocessing import label as sklabel
from sklearn.utils.validation import check_is_fitted


class LabelEncoder(sklabel.LabelEncoder):

    __doc__ = sklabel.LabelEncoder.__doc__

    def _check_array(self, y):
        if isinstance(y, dd.Series):
            y = da.asarray(y)
        return y

    def fit(self, y):
        y = self._check_array(y)

        if isinstance(y, da.Array):
            classes_ = da.unique(y)
            classes_ = classes_.compute()
        else:
            classes_ = np.unique(y)

        self.classes_ = classes_

        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = self._check_array(y)

        if isinstance(y, da.Array):
            return da.map_blocks(np.searchsorted, self.classes_, y,
                                 dtype=self.classes_.dtype)
        else:
            return np.searchsorted(self.classes_, y)

    def inverse_transform(self, y):
        check_is_fitted(self, 'classes_')
        y = self._check_array(y)

        if isinstance(y, da.Array):
            return da.map_blocks(getitem, self.classes_, y,
                                 dtype=self.classes_.dtype)
        else:
            y = np.asarray(y)
            return self.classes_[y]
