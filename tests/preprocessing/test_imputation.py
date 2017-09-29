from daskml.datasets import make_classification
from sklearn.preprocessing import Imputer as Imputer_
from daskml.preprocessing import Imputer
from dask.array.utils import assert_eq as assert_eq_ar
import dask.dataframe as dd
import pandas as pd
import numpy as np


X, y = make_classification(chunks=2)
df = X.to_dask_dataframe()
df2 = dd.from_pandas(pd.DataFrame(5*[range(42)]+[42*[np.nan]]).T,
                     npartitions=5)


def _get_scaler_attributes(scaler):
    return filter(lambda a: a.endswith("_") and not a.startswith("__"),
                  dir(scaler))


class TestImputer(object):
    def test_basic(self):
        mask = [3, 4]
        a = Imputer(columns=mask)
        b = Imputer_()

        a.fit(df2)
        b.fit(df2[mask].compute())

        for attr in _get_scaler_attributes(self):
            assert_eq_ar(getattr(a, attr), getattr(b, attr))
