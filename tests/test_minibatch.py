from sklearn import cluster as cluster_
from daskml import cluster

from dask.array.utils import assert_eq


class TestMiniBatchKMeans(object):

    def test_basic(self, single_chunk_blobs):
        X, y = single_chunk_blobs
        a = cluster.BigMiniBatchKMeans(n_clusters=3, random_state=0)
        b = cluster_.MiniBatchKMeans(n_clusters=3, random_state=0)
        a.fit(X)
        b.partial_fit(X)

        assert_eq(a.cluster_centers_, b.cluster_centers_)
