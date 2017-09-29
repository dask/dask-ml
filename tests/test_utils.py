from daskml.utils import slice_columns, handle_zeros_in_scale
from dask.array.utils import assert_eq as assert_eq_ar
from dask.array.utils import assert_eq as assert_eq_df
from daskml.datasets import make_classification
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np


df = dd.from_pandas(pd.DataFrame(5*[range(42)]).T, npartitions=5)
s = dd.from_pandas(pd.Series([0, 1, 2, 3, 0]), npartitions=5)
a = da.from_array(np.array([0, 1, 2, 3, 0]), chunks=3)
X, y = make_classification(chunks=2)


def test_slice_columns():
    columns = [2, 3]
    df2 = slice_columns(df, columns)
    X2 = slice_columns(X, columns)

    assert list(df2.columns) == columns
    assert_eq_df(df[columns].compute(), df2.compute())
    assert_eq_ar(X.compute(), X2.compute())


def test_handle_zeros_in_scale():
    s2 = handle_zeros_in_scale(s)
    a2 = handle_zeros_in_scale(a)

    assert list(s2.compute()) == [1, 1, 2, 3, 1]
    assert list(a2.compute()) == [1, 1, 2, 3, 1]
