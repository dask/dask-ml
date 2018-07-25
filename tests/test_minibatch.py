import pytest
from sklearn import cluster as cluster_

from dask_ml import cluster
from dask_ml.utils import assert_estimator_equal


@pytest.mark.filterwarnings("ignore:'Partial:FutureWarning")
class TestMiniBatchKMeans(object):
    def test_basic(self, single_chunk_blobs):
        X, y = single_chunk_blobs
        a = cluster.PartialMiniBatchKMeans(n_clusters=3, random_state=0)
        b = cluster_.MiniBatchKMeans(n_clusters=3, random_state=0)
        a.fit(X)
        b.partial_fit(X)
        assert_estimator_equal(a, b, exclude=["random_state_"])
