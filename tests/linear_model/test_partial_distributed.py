import dask
import pytest

from dask_ml.linear_model import PartialSGDClassifier
from dask_ml.datasets import make_classification

distributed = pytest.importorskip("distributed", minversion="1.21.8")
cluster = pytest.importorskip("distributed.utils_test").cluster


def test_distributed_threaded_get():
    with cluster() as (s, [a, b]):
        with distributed.Client(s['address']):
            X, y = make_classification(chunks=(10, 20), random_state=0)
            X, y = dask.persist(X, y)

            clf = PartialSGDClassifier(classes=[0, 1, 2])
            clf.fit(X, y)
