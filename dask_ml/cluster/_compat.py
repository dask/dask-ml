from .._compat import SK_022, SK_0221

if SK_0221:
    from sklearn.cluster._kmeans import _k_init
elif SK_022:
    from sklearn.cluster._k_means import _k_init
else:
    from sklearn.cluster.k_means_ import _k_init


__all__ = ["_k_init"]
