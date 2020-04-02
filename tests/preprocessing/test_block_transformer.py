import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from dask_ml.preprocessing import BlockTransformer
from dask_ml.utils import assert_estimator_equal

X = np.arange(100).reshape((25, 4))
X_dask = da.from_array(X, chunks=(5, 4))
df_dask = X_dask.to_dask_dataframe().rename(columns=str)
df = df_dask.compute()


def multiply(x, factor=4):
    return factor * x


class TestBlockTransformer:
    @pytest.mark.parametrize("factor", [2, 4, None])
    @pytest.mark.parametrize("validation", [True, False])
    @pytest.mark.parametrize("daskify", [True, False])
    def test_block_transform_multiply(self, daskify, validation, factor):
        X = np.arange(100).reshape((25, 4))
        df = pd.DataFrame(X).rename(columns=str)
        if daskify:
            X = da.from_array(X, chunks=(5, 4))
            df = dd.from_pandas(df, npartitions=2)
        if factor:
            bt = BlockTransformer(multiply, validate=validation, factor=factor)
        else:
            bt = BlockTransformer(multiply, validate=validation)
        if daskify:
            assert dask.is_dask_collection(bt.transform(X))
            assert dask.is_dask_collection(bt.transform(df))
        if factor:
            da.utils.assert_eq(bt.transform(X), multiply(X, factor=factor))
            dd.utils.assert_eq(bt.transform(df), multiply(df, factor=factor))
        else:
            da.utils.assert_eq(bt.transform(X), multiply(X))
            dd.utils.assert_eq(bt.transform(df), multiply(df))

    @pytest.mark.parametrize("validate", [True, False])
    @pytest.mark.parametrize("daskify", [True, False])
    def test_validate(self, mocker, daskify, validate):
        X = np.arange(100).reshape((25, 4))
        df = pd.DataFrame(X).rename(columns=str)
        if daskify:
            X = da.from_array(X, chunks=(5, 4))
            df = dd.from_pandas(df, npartitions=2)
        m = mocker.patch("dask_ml.preprocessing._block_transformer.check_array")
        bt = BlockTransformer(lambda x: x, validate=validate)
        if validate:
            _ = bt.transform(X)
            m.assert_called_once()
            m.reset_mock()
            _ = bt.transform(df)
            m.assert_called_once()
        else:
            _ = bt.transform(X)
            m.assert_not_called()
            _ = bt.transform(df)
            m.assert_not_called()

    def test_idempotence(self):
        X = np.arange(100).reshape((25, 4))
        bt = BlockTransformer(lambda x: 2 * x)
        assert_estimator_equal(bt.fit(X), bt.fit(X).fit(X))
