import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from dask.delayed import Delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

import dask_ml.feature_extraction.text
from dask_ml._partial import fit, predict
from dask_ml.datasets import make_classification
from dask_ml.wrappers import Incremental

x = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [0, 2], [3, 3], [4, 4]])

y = np.array([1, 1, 1, 1, -1, -1, 0, 0])

z = np.array([[1, -1], [-1, 1], [10, -10], [-10, 10]])

X = da.from_array(x, chunks=(3, 2))
Y = da.from_array(y, chunks=(3,))
Z = da.from_array(z, chunks=(2, 2))


def test_fit():
    with dask.config.set(scheduler="single-threaded"):
        sgd = SGDClassifier(max_iter=5, tol=1e-3)

        sgd = fit(sgd, X, Y, classes=np.array([-1, 0, 1]))

        sol = sgd.predict(z)
        result = predict(sgd, Z)
        assert result.chunks == ((2, 2),)
        assert result.compute().tolist() == sol.tolist()


def test_no_compute():
    sgd = SGDClassifier(max_iter=5, tol=1e-3)

    result = fit(sgd, X, Y, classes=np.array([-1, 0, 1]), compute=False)
    assert isinstance(result, Delayed)


def test_fit_rechunking():
    n_classes = 2
    X, y = make_classification(chunks=20, n_classes=n_classes)
    X = X.rechunk({1: 10})

    assert X.numblocks[1] > 1

    clf = Incremental(SGDClassifier(max_iter=5, tol=1e-3))
    clf.fit(X, y, classes=list(range(n_classes)))


def test_fit_shuffle_blocks():
    N = 10
    X = da.from_array(1 + np.arange(N).reshape(-1, 1), chunks=1)
    y = da.from_array(np.ones(N), chunks=1)
    classes = [0, 1]

    sgd = SGDClassifier(
        max_iter=5, random_state=0, fit_intercept=False, shuffle=False, tol=1e-3
    )

    sgd1 = fit(clone(sgd), X, y, random_state=0, classes=classes)
    sgd2 = fit(clone(sgd), X, y, random_state=42, classes=classes)
    assert len(sgd1.coef_) == len(sgd2.coef_) == 1
    assert not np.allclose(sgd1.coef_, sgd2.coef_)

    X, y = make_classification(random_state=0, chunks=20)
    sgd_a = fit(clone(sgd), X, y, random_state=0, classes=classes, shuffle_blocks=False)
    sgd_b = fit(
        clone(sgd), X, y, random_state=42, classes=classes, shuffle_blocks=False
    )
    assert np.allclose(sgd_a.coef_, sgd_b.coef_)

    with pytest.raises(ValueError, match="cannot be used to seed"):
        fit(
            sgd,
            X,
            y,
            classes=np.array([-1, 0, 1]),
            shuffle_blocks=True,
            random_state=da.random.RandomState(42),
        )


def test_dataframes():
    df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    ddf = dd.from_pandas(df, npartitions=2)

    with dask.config.set(scheduler="single-threaded"):
        sgd = SGDClassifier(max_iter=5, tol=1e-3)

        sgd = fit(sgd, ddf[["x"]], ddf.y, classes=[0, 1])

        sol = sgd.predict(df[["x"]])
        result = predict(sgd, ddf[["x"]])

        da.utils.assert_eq(sol, result)


def test_bag():
    x = db.from_sequence(range(10), npartitions=2)
    vect = dask_ml.feature_extraction.text.HashingVectorizer()
    vect = fit(vect, x, None)
    y = vect.transform(x)
    assert y.shape[1] == vect.n_features


def test_no_partial_fit_raises():
    X, y = make_classification(chunks=50)
    with pytest.raises(ValueError, match="RandomForestClassifier"):
        fit(RandomForestClassifier(), X, y)
