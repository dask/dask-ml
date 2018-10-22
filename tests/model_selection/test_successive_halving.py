import numpy as np
from distributed.utils_test import cluster, gen_cluster, loop  # noqa: F401
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from dask_ml.model_selection._successive_halving import SuccessiveHalving


@gen_cluster(client=True)
def test_basic_successive_halving(c, s, a, b):
    model = SGDClassifier(tol=1e-3)
    params = {'alpha': np.logspace(-3, 0, num=1000)}
    n, r = 10, 5
    search = SuccessiveHalving(model, params, n, r)

    X, y = make_classification()
    yield search.fit(X, y, classes=np.unique(y))
    assert search.best_score_ > 0
    assert isinstance(search.best_estimator_, SGDClassifier)
