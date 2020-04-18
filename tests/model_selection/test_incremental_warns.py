import warnings
import numpy as np
import pytest
from dask_ml.datasets import make_classification
from dask_ml.utils import ConstantFunction
from dask_ml.model_selection import IncrementalSearchCV, InverseDecaySearchCV
from distributed.utils_test import gen_cluster



@gen_cluster(client=True)
def test_warns_decay_rate(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=10)

    params = {"value": np.random.RandomState(42).rand(1000)}
    model = ConstantFunction()

    search = IncrementalSearchCV(model, params, max_iter=5, n_initial_parameters=5)
    match = r"deprecated since Dask-ML v1.4.0."
    with pytest.warns(FutureWarning, match=match):
        yield search.fit(X, y)

    # Make sure the printed warning message works
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, module='dask_ml')
        yield search.fit(X, y)


@gen_cluster(client=True)
def test_warns_decay_rate_wanted(c, s, a, b):
    X, y = make_classification(n_samples=100, n_features=5, chunks=10)

    params = {"value": np.random.RandomState(42).rand(1000)}
    model = ConstantFunction()

    search = IncrementalSearchCV(model, params, max_iter=5, n_initial_parameters=5, decay_rate=1)
    match = "decay_rate is deprecated .* Use InverseDecaySearchCV"
    with pytest.warns(UserWarning, match=match):
        yield search.fit(X, y)

    # Make sure old behavior is retained w/o warning
    search = InverseDecaySearchCV(model, params, decay_rate=1)
    yield search.fit(X, y)
