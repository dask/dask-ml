Changelog
=========

Version 0.7.0
~~~~~~~~~~~~~

Enhancements
------------

- Added ``sample_weight`` support for :meth:`dask_ml.metrics.accuracy_score`. (:pr:`217`)
- Improved performance of training on :class:`dask_ml.cluster.SpectralClustering` (:pr:`152`)
- Added :class:`dask_ml.preprocessing.LabelEncoder`. (:pr:`226`)
- Added the ``max_iter`` parameter to :class:`dask_ml.wrappers.Incremental` to make multiple passes over the training data (:pr:`258`)

API Breaking Changes
--------------------

- Removed the ``basis_inds_`` attribute from :class:`dask_ml.cluster.SpectralClustering` as its no longer used (:pr:`152`)
- Change :meth:`dask_ml.wrappers.Incremental.fit` to clone the underlying estimator before training (:pr:`258`). This induces a few changes

  1. The underlying estimator no longer gives access to learned attributes like ``coef_``. We recommend using
     ``Incremental.coef_``.
  2. State no longer leaks between successive ``fit`` calls. Note that :meth:`Incremental.partial_fit` is still available
     if you want state, like learned attributes or random seeds, to be re-used. This is useful if you're making multiple
     passes over the training data.
- Changed ``get_params`` and ``set_params`` for :class:`dask_ml.wrappers.Incremental` to no longer magically get / set parameters
  for the underlying estimator (:pr:`258`). To specify parameters for the underlying estimator, use the double-underscore prefix convention
  established by scikit-learn:

  .. code-block:: python

     inc.set_params('estimator__alpha': 10)


Version 0.6.0
~~~~~~~~~~~~~

API Breaking Changes
--------------------

- Removed the `get` keyword from the incremental learner ``fit`` methods. (:pr:`187`)
- Deprecated the various ``Partial*`` estimators in favor of the :class:`dask_ml.wrappers.Incremental` meta-estimator (:pr:`190`)

Enhancements
------------

- Added a new meta-estimator :class:`dask_ml.wrappers.Incremental` for wrapping any estimator with a `partial_fit` method. See :ref:`incremental.blockwise-metaestimator` for more. (:pr:`190`)
- Added an R2-score metric :meth:`dask_ml.metrics.r2_score`.

Version 0.5.0
~~~~~~~~~~~~~

API Breaking Changes
--------------------

- The `n_samples_seen_` attribute on :class:`dask_ml.preprocessing.StandardScalar` is now consistently ``numpy.nan`` (:issue:`157`).
- Changed the algorithm for :meth:`dask_ml.datasets.make_blobs`, :meth:`dask_ml.datasets.make_regression` and :meth:`dask_ml.datasets.make_classfication` to reduce the single-machine peak memory usage (:issue:`67`)

Enhancements
------------

- Added :func:`dask_ml.model_selection.train_test_split` and :class:`dask_ml.model_selection.ShuffleSplit` (:issue:`172`)
- Added :func:`dask_ml.metrics.classification_score`, :func:`dask_ml.metrics.mean_absolute_error`, and :func:`dask_ml.metrics.mean_squared_error`.


Bug Fixes
---------

- :class:`dask_ml.preprocessing.StandardScalar` now works on DataFrame inputs (:issue:`157`).
-

Version 0.4.1
~~~~~~~~~~~~~

This release added several new estimators.

Enhancements
------------

Added :class:`dask_ml.preprocessing.RobustScaler`
"""""""""""""""""""""""""""""""""""""""""""""""""

Scale features using statistics that are robust to outliers. This mirrors
:class:`sklearn.preprocessing.RobustScalar` (:issue:`62`).

Added :class:`dask_ml.preprocessing.OrdinalEncoder`
"""""""""""""""""""""""""""""""""""""""""""""""""""

Encodes categorical features as ordinal, in one ordered feature (:issue:`119`).

Added :class:`dask_ml.wrappers.ParallelPostFit`
"""""""""""""""""""""""""""""""""""""""""""""""

A meta-estimator for fitting with any scikit-learn estimator, but post-processing
(``predict``, ``transform``, etc.) in parallel on dask arrays.
See :ref:`parallel-meta-estimators` for more (:issue:`132`).

Version 0.4.0
~~~~~~~~~~~~~

API Changes
-----------

- Changed the arguments of the dask-glm based estimators in
  ``dask_glm.linear_model`` to match scikit-learn's API (:issue:`94`).

  * To specify ``lambuh`` use ``C = 1.0 / lambduh`` (the default of 1.0 is
    unchanged)
  * The ``rho``, ``over_relax``, ``abstol`` and ``reltol`` arguments have been
    removed. Provide them in ``solver_kwargs`` instead.

  This affects the ``LinearRegression``, ``LogisticRegression`` and
  ``PoissonRegression`` estimators.

Enhancements
------------

- Accept ``dask.dataframe`` for dask-glm based estimators (:issue:`84`).

Version 0.3.2
~~~~~~~~~~~~~

Enhancements
------------

- Added :meth:`dask_ml.preprocessing.TruncatedSVD` and
  :meth:`dask_ml.preprocessing.PCA` (:issue:`78`)

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
