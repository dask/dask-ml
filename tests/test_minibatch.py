from sklearn import cluster as cluster_
from daskml import cluster

from .test_utils import assert_estimator_equal


class TestMiniBatchKMeans(object):

    def test_basic(self, single_chunk_blobs):
        X, y = single_chunk_blobs
        a = cluster.BigMiniBatchKMeans(n_clusters=3, random_state=0)
        b = cluster_.MiniBatchKMeans(n_clusters=3, random_state=0)
        a.fit(X)
        b.partial_fit(X)
        assert_estimator_equal(a, b, exclude=['random_state_'])
