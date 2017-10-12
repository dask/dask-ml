from daskml.datasets import make_classification
from sklearn.preprocessing import Imputer as Imputer_
from daskml.preprocessing import Imputer
import dask.dataframe as dd
import pandas as pd
from dask.array.utils import assert_eq as assert_eq_ar
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
        a = Imputer()
        b = Imputer_()

        a.fit(df)
        b.fit(df.values.compute())

        assert_eq_ar(a.transform(df).values.compute(),
                     b.transform(df.values.compute()))
