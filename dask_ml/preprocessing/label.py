from __future__ import division

from operator import getitem

import dask.array as da
import dask.dataframe as dd
import pandas as pd
import numpy as np

from sklearn.preprocessing import label as sklabel
from sklearn.utils.validation import check_is_fitted


class LabelEncoder(sklabel.LabelEncoder):
    """Encode labels with value between 0 and n_classes-1.

    .. note::

       This differs from the scikit-learn version for Categorical data.
       When passed a categorical `y`, this implementation will use the
       categorical information for the label encoding and transformation.
       You will receive different answers when

       1. Your categories are not monotonically increasing
       2. You have unobserved categories

       Specify ``use_categorical=False`` to recover the scikit-learn behavior.

    Parameters
    ----------
    use_categorical : bool, default True
        Whether to use the categorical dtype information when `y` is a
        dask or pandas Series with a categorical dtype.

    Attributes
    ----------
    classes_ : array of shape (n_class,)
        Holds the label for each class.
    dtype_ : Optional CategoricalDtype
        For Categorical `y`, the dtype is stored here.

    Examples
    --------
    `LabelEncoder` can be used to normalize labels.

    >>> from dask_ml import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6]) #doctest: +ELLIPSIS
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"]) #doctest: +ELLIPSIS
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']

    When using Dask, we strongly recommend using a Categorical dask Series if
    possible. This avoids a (potentially expensive) scan of the values and
    enables a faster `transform` algorithm.

    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> data = dd.from_pandas(pd.Series(['a', 'a', 'b'], dtype='category'),
    ...                       npartitions=2)
    >>> le.fit_transform(data)
    dask.array<values, shape=(nan,), dtype=int8, chunksize=(nan,)>
    >>> le.fit_transform(data).compute()
    array([0, 0, 1], dtype=int8)
    """

    def __init__(self, use_categorical=True):
        self.use_categorical = use_categorical
        super(LabelEncoder, self).__init__()

    def _check_array(self, y):
        if isinstance(y, (dd.Series, pd.DataFrame)):
            y = y.squeeze()

            if y.ndim > 1:
                raise ValueError("Expected a 1-D array or Series.")

        if not self.use_categorical:
            if isinstance(y, dd.Series):
                y = da.asarray(y)
            elif isinstance(y, pd.Series):
                y = np.asarray(y)

        if isinstance(y, dd.Series):
            if pd.api.types.is_categorical_dtype(y):
                # TODO(dask-3784): just call y.cat.as_known()
                # https://github.com/dask/dask/issues/3784
                if not y.cat.known:
                    y = y.cat.as_known()
            else:
                y = da.asarray(y)
        return y

    def fit(self, y):
        y = self._check_array(y)

        if isinstance(y, da.Array):
            classes_ = da.unique(y)
            classes_ = classes_.compute()
            dtype = None
        elif _is_categorical(y):
            # This may not be sorted.
            classes_ = np.array(y.cat.categories)
            dtype = y.dtype
        else:
            classes_ = np.unique(y)
            dtype = None

        self.classes_ = classes_
        self.dtype_ = dtype

        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def transform(self, y):
        check_is_fitted(self, "classes_")
        y = self._check_array(y)

        if isinstance(y, da.Array):
            return da.map_blocks(
                np.searchsorted, self.classes_, y, dtype=self.classes_.dtype
            )
        elif isinstance(y, (pd.Series, dd.Series)):
            assert y.dtype.categories.equals(self.dtype_.categories)
            return y.cat.codes.values
        else:
            return np.searchsorted(self.classes_, y)

    def inverse_transform(self, y):
        check_is_fitted(self, "classes_")
        y = self._check_array(y)

        if isinstance(y, da.Array):
            if getattr(self, "dtype_", None):
                # -> Series[category]
                result = (
                    dd.from_dask_array(y)
                    .astype("category")
                    .cat.set_categories(np.arange(len(self.classes_)))
                    .cat.rename_categories(self.dtype_.categories)
                )
                if self.dtype_.ordered:
                    result = result.cat.as_ordered()
                return result
            else:
                return da.map_blocks(
                    getitem, self.classes_, y, dtype=self.classes_.dtype
                )
        else:
            y = np.asarray(y)
            if getattr(self, "dtype_", None):
                return pd.Series(
                    pd.Categorical.from_codes(
                        y,
                        categories=self.dtype_.categories,
                        ordered=self.dtype_.ordered,
                    )
                )
            else:
                return self.classes_[y]


def _is_categorical(y):
    return isinstance(y, (dd.Series, pd.Series)) and pd.api.types.is_categorical_dtype(
        y
    )
