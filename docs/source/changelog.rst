Changelog
=========

Version 0.3.0
~~~~~~~~~~~~~

Enhancements
------------

- Added :meth:`KMeans.predict` (:issue:`83`)

API Changes
-----------

- Changed the fitted attributes on ``MinMaxScaler`` and ``StandardScaler`` to be
  concrete NumPy or pandas objects, rather than persisted dask objects
  (:issue:`75`).
