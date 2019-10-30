import warnings

from .._compat import SK_022

if SK_022:
    with warnings.catch_warnings():
        # these should only be used in tests.
        warnings.simplefilter("ignore", FutureWarning)
        from sklearn.decomposition._pca import _assess_dimension_, _infer_dimension_
else:
    from sklearn.decomposition.pca import _assess_dimension_, _infer_dimension_


__all__ = ["_assess_dimension_", "_infer_dimension_"]
