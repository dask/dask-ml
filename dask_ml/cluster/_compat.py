from .._compat import SK_024

if SK_024:
    from sklearn.cluster._kmeans import _kmeans_plusplus  # noqa

    __all__ = ["_kmeans_plusplus"]

else:
    from sklearn.cluster._kmeans import _k_init

    __all__ = ["_k_init"]
