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
from dask_ml.utils import assert_estimator_equal

X = np.arange(100).reshape((25, 4))
X_dask = da.from_array(X, chunks=(5, 4))


class TestBlockTransformer:
    def test_multiple_two(self):
        multiply_by_two = lambda x: 2 * x
        bt = dpp.BlockTransformer(multiply_by_two)

        assert_eq_ar(bt.transform(X), 2 * X)
