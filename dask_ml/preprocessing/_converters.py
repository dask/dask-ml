import dask.array as da
import dask.dataframe as dd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ArrayConverter(BaseEstimator, TransformerMixin):
    """Convert values to a Dask or NumPy array.

    Parameters
    ----------
    lengths : Union[None, bool, Sequence[int]], default None
        How the chunk sizes for the output Dask array should be determined.

        * None : The output array has unknown chunk sizes
        * True : the output array's chunk lengths are immediately computed
        * Sequence[int] : The chunk lengths are set to `lengths`. Theses
          values are *not* validated for correctness.

        When the data passed to `fit` is not a dask DataFrame, this parameter
        has no effect.

    Returns
    -------
    dask.array.Array or numpy.ndarray

    Notes
    -----
    This is most useful in a pipeline when you load your data as a dask
    dataframe, but some estimator later in the pipeline requires that the
    input be a dask array with known chunk lengths.

    Examples
    --------
    >>> import dask.dataframe as dd
    >>> import pandas as pd
    >>> df = dd.from_pandas(pd.DataFrame({"A": range(20)}), npartitions=5)
    >>> # Get a dataframe with unknown divisions
    >>> df = df.reset_index(drop=True)
    >>> converter = ArrayConverter(lengths=(4, 4, 4, 4, 4))
    >>> converter.fit_transform(df)

    If you don't know the lengths ahead of time, pass ``lengths=True``. This
    will immediate compute the lengths, which can be expensive. You might
    consider pre-computing them with
    ``lengths = tuple(df.map_partitions(len).compute())``, as long as no
    stages in the pipeline modify the number of samples.
    """

    def __init__(self, lengths=None):
        self.lengths = lengths

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, dd.DataFrame):
            return X.to_dask_array(lengths=self.lengths)
        elif isinstance(X, da.Array):
            return X
        else:
            return np.asarray(X)
