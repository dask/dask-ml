from __future__ import division

from operator import getitem

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.sparse
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
                y = y.to_dask_array(lengths=True)
            elif isinstance(y, pd.Series):
                y = np.asarray(y)

        if isinstance(y, dd.Series):
            if pd.api.types.is_categorical_dtype(y):
                # TODO(dask-3784): just call y.cat.as_known()
                # https://github.com/dask/dask/issues/3784
                if not y.cat.known:
                    y = y.cat.as_known()
            else:
                y = y.to_dask_array(lengths=True)
        return y

    def fit(self, y):
        y = self._check_array(y)

        if isinstance(y, da.Array):
            classes_ = _encode_dask_array(y)
            self.classes_ = classes_.compute()
            self.dtype_ = None
        elif _is_categorical(y):
            self.classes_ = _encode_categorical(y)
            self.dtype_ = y.dtype
        else:
            self.dtype_ = None
            return super(LabelEncoder, self).fit(y)

        return self

    def fit_transform(self, y):
        y = self._check_array(y)

        if isinstance(y, da.Array):
            self.classes_, y = _encode_dask_array(y, encode=True)
            self.dtype_ = None
        elif _is_categorical(y):
            self.classes_, y = _encode_categorical(y, encode=True)
            self.dtype_ = y.dtype
        else:
            return super(LabelEncoder, self).fit_transform(y)

        return y

    def transform(self, y):
        check_is_fitted(self, "classes_")
        y = self._check_array(y)

        if isinstance(y, da.Array):
            return _encode_dask_array(y, self.classes_, encode=True)[1]
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
                    getitem,
                    self.classes_,
                    y,
                    dtype=self.classes_.dtype,
                    chunks=y.chunks,
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


def _encode_categorical(values, uniques=None, encode=False):
    # type: (Union[dd.Series['category'], pd.Series['category']], bool) -> Any
    new_uniques = np.asarray(values.cat.categories)

    if uniques is not None:
        diff = list(np.setdiff1d(uniques, new_uniques, assume_unique=True))
        if diff:
            raise ValueError("y comtains previously unseen labels: {}".format(diff))

    uniques = new_uniques

    if encode:
        return uniques, values.cat.codes
    else:
        return uniques


def _check_and_search_block(arr, uniques, onehot_dtype=None, block_info=None):
    diff = list(np.setdiff1d(arr, uniques, assume_unique=True))

    if diff:
        msg = (
            "Block contains previously unseen values {}.\nBlock info:\n\n"
            "{}".format(diff, block_info)
        )
        raise ValueError(msg)

    label_encoded = np.searchsorted(uniques, arr)
    if onehot_dtype:
        return _construct(label_encoded, uniques)
    else:
        return label_encoded


def _construct(x, categories):
    """Make a sparse matrix from an encoded array.

    >>> construct(np.array([0, 1, 0]), np.array([0, 1])).toarray()
    array([[1., 0.],
           [0., 1.],
           [1., 0.]])
    """
    # type: (np.ndarray, np.ndarray) -> scipy.sparse.csr_matrix
    data = np.ones(len(x))
    rows = np.arange(len(x))
    columns = x.ravel()
    return scipy.sparse.csr_matrix(
        (data, (rows, columns)), shape=(len(x), len(categories))
    )


def _encode_dask_array(values, uniques=None, encode=False, onehot_dtype=None):
    """One-hot or label encode a dask array.

    Parameters
    ----------
    values : da.Array, shape [n_samples,]
    unqiques : np.ndarray, shape [n_uniques,]
    encode : bool, default False
        Whether to encode the values (True) or just discover the uniques.
    onehot_dtype : np.dtype, optional
        Optional dtype for the resulting one-hot encoded array. This changes
        the shape, dtype, and underlying storage of the returned dask array.

        ======= ================= =========================
        thing   onehot_dtype=None onehot_dtype=onehot_dtype
        ======= ================= =========================
        shape   (n_samples,)      (n_samples, len(uniques))
        dtype   np.intp           onehot_dtype
        storage np.ndarray        scipy.sparse.csr_matrix
        ======= ================= =========================

    Returns
    -------
    uniques : ndarray
        The discovered uniques (uniques=None) or just `uniques`
    encoded : da.Array, optional
        The encoded values. Only returend when ``encode=True``.
    """

    if uniques is None:
        if encode and onehot_dtype:
            raise ValueError("Cannot use 'encode` and 'onehot_dtype' simultaneously.")
        if encode:
            uniques, encoded = da.unique(values, return_inverse=True)
            return uniques, encoded
        else:
            return da.unique(values)

    if encode:
        if onehot_dtype:
            dtype = onehot_dtype
            new_axis = 1
            chunks = values.chunks + (len(uniques),)
        else:
            dtype = np.intp
            new_axis = None
            chunks = values.chunks

        return (
            uniques,
            values.map_blocks(
                _check_and_search_block,
                uniques,
                onehot_dtype=onehot_dtype,
                dtype=dtype,
                new_axis=new_axis,
                chunks=chunks,
            ),
        )
    else:
        return uniques


def _encode(values, uniques=None, encode=False):
    if isinstance(values, (pd.Series, dd.Series)) and _is_categorical(values):
        return _encode_categorical(values, uniques=uniques, encode=encode)
    elif isinstance(values, da.Array):
        return _encode_dask_array(values, uniques=uniques, encode=encode)
    else:
        raise ValueError("Unknown type {}".format(type(values)))


def _is_categorical(y):
    return isinstance(y, (dd.Series, pd.Series)) and pd.api.types.is_categorical_dtype(
        y
    )
