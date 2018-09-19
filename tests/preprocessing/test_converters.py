import dask.array as da
import dask.dataframe as dd
import pandas as pd
import pytest

from dask_ml.preprocessing import ArrayConverter


@pytest.mark.parametrize("lengths", [True, None, (4, 4, 4, 4, 4)])
def test_array_converter(lengths):
    df = pd.DataFrame({"A": range(20)})
    ddf = dd.from_pandas(df, npartitions=5)
    expected = ddf.values
    # Get a dataframe with unknown divisions
    ddf = ddf.reset_index(drop=True)
    converter = ArrayConverter(lengths=lengths)

    # dask dataframe
    result = converter.fit_transform(ddf)
    da.utils.assert_eq(result, expected)

    # dask array
    result = converter.fit_transform(expected)
    da.utils.assert_eq(result, expected)

    # pandas dataframe
    result = converter.fit_transform(df)
    da.utils.assert_eq(result, expected)

    # numpy array
    result = converter.fit_transform(df.values)
    da.utils.assert_eq(result, expected)
