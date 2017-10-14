import dask_ml.preprocessing as dpp
import sklearn.preprocessing as spp
from dask_ml.datasets import make_classification
from dask.array.utils import assert_eq as assert_eq_ar
from dask_ml.utils import assert_estimator_equal


X, y = make_classification(chunks=2)
X[X < 0] = 0
df = X.to_dask_dataframe().rename(columns=str)
df = df.mask(df > 1)


class TestImputer(object):
    def test_array_fit(self):
        a = dpp.Imputer(missing_values=0)
        b = spp.Imputer(missing_values=0)

        a.fit(X)
        b.fit(X.compute())
        assert_estimator_equal(a, b)

    def test_df_fit(self):
        mask = ['1', '2']
        mask_ix = list(map(int, mask))
        a = dpp.Imputer(columns=mask)
        b = spp.Imputer()

        a.fit(df)
        b.fit(df.values.compute())

        assert_eq_ar(a.statistics_.values.compute(),
                     b.statistics_[mask_ix])
