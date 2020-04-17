from datetime import date

import dask
import dask.array as da
import numpy as np
import pytest
from dask.array.utils import assert_eq

import dask_ml.datasets


@pytest.mark.parametrize(
    "generator", [dask_ml.datasets.make_blobs, dask_ml.datasets.make_regression]
)
def test_make_blobs_raises_chunk_second_axis(generator):
    with pytest.raises(ValueError) as m:
        generator(n_features=50, chunks=(100, 40))
    assert m.match("partitioned along the first axis.")


def test_make_regression():
    X, y = dask_ml.datasets.make_regression(
        n_samples=150, n_features=75, chunks=(50, 75), random_state=0
    )
    assert isinstance(X, da.Array)
    assert X.shape == (150, 75)
    assert X.compute().shape == X.shape

    assert isinstance(y, da.Array)
    assert y.shape == (150,)
    assert y.compute().shape == y.shape

    X, y, coef = dask_ml.datasets.make_regression(
        n_samples=150,
        n_features=75,
        chunks=(50, 75),
        random_state=0,
        coef=True,
        bias=1.0,
        noise=2.0,
    )
    assert isinstance(coef, np.ndarray)

    e = y - X.dot(coef)
    mean, std = dask.compute(e.mean(0), e.std(0))

    assert abs(mean - 1.0) <= 0.1
    assert abs(std - 2.0) <= 0.1


@pytest.mark.parametrize(
    "generator",
    [
        dask_ml.datasets.make_blobs,
        dask_ml.datasets.make_classification,
        dask_ml.datasets.make_counts,
        dask_ml.datasets.make_regression,
    ],
)
def test_deterministic(generator, scheduler):
    a, t = generator(chunks=100, random_state=10)
    b, u = generator(chunks=100, random_state=10)
    assert_eq(a, b)
    assert_eq(t, u)


def test_make_classification_df():
    X_df, y_series = dask_ml.datasets.make_classification_df(
        n_samples=100,
        n_features=5,
        random_state=123,
        chunks=100,
        dates=(date(2014, 1, 1), date(2015, 1, 1)),
    )

    assert X_df is not None
    assert y_series is not None
    assert "date" in X_df.columns
    assert len(X_df.columns) == 6
    assert len(X_df) == 100
    assert len(y_series) == 100
    assert isinstance(y_series, dask.dataframe.core.Series)
