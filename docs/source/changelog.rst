Changelog
=========

Version 0.4.0
~~~~~~~~~~~~~

Enhancements
------------

- Added :meth:`dask_ml.cluster.KMeans.predict` (:issue:`83`).
- Added :class:`dask_ml.decomposition.PCA` and
  :class:`dask_ml.decomposition.TruncatedSVD (:issue:`78`).

API Changes
-----------

- Changed the fitted attributes on ``MinMaxScaler`` and ``StandardScaler`` to be
  concrete NumPy or pandas objects, rather than persisted dask objects
  (:issue:`75`).
- Correctly handle ``random_state`` in ``KMeans.fit`` for both ``k-means||`` and
  ``k-means++`` initialization (:issue:`80`).
