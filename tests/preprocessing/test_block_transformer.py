import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas.util.testing as tm
import pytest
import sklearn.preprocessing as spp
from dask import compute
from dask.array.utils import assert_eq as assert_eq_ar
from dask.dataframe.utils import assert_eq as assert_eq_df

import dask_ml.preprocessing as dpp
from dask_ml.datasets import make_classification
from dask_ml.utils import assert_estimator_equal, check_array
from pytest_mock import mocker

X = np.arange(100).reshape((25, 4))
X_dask = da.from_array(X, chunks=(5, 4))
df_dask = X_dask.to_dask_dataframe().rename(columns=str)
df = df_dask.compute()


class TestBlockTransformer:
    @pytest.mark.parametrize("func", [lambda x: 2 * x])
    @pytest.mark.parametrize("preserve", [True, False])
    @pytest.mark.parametrize("validation", [True, False])
    @pytest.mark.parametrize("daskify", [True, False])
    def test_multiple_two(self, daskify, validation, preserve, func):
        X = np.arange(100).reshape((25, 4))
        df = pd.DataFrame(X).rename(columns=str)
        if daskify:
            X = da.from_array(X, chunks=(5, 4))
            df = dd.from_pandas(df, npartitions=2)
        bt = dpp.BlockTransformer(
            func, validate=validation, preserve_dataframe=preserve
        )

        if daskify:
            assert dask.is_dask_collection(bt.transform(X))
        else:
            if preserve:
                assert_eq_df(bt.transform(df), func(df))

        # assert dask.is_dask_collection(bt.transform(df_dask))
        # assert_eq_df(bt.transform(df_dask), 2 * df)

    def test_validate(self, mocker):
        m = mocker.patch("dask_ml.preprocessing._block_transformer.check_array")
        bt = dpp.BlockTransformer(lambda x: x)
        _ = bt.transform(X)
        m.assert_called_once()
        m.reset_mock()
        bt = dpp.BlockTransformer(lambda x: x, validate=False)
        _ = bt.transform(X)
        m.assert_not_called()
