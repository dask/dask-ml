Changelog
=========

Version 1.7.0
~~~~~~~~~~~~~

* Improved documentation for working with PyTorch models, see :ref:`pytorch` (:pr:`699`)
* Improved documentation for working with Keras / TensorFlow models, see :ref:`keras` (:pr:`713`)
* Fixed handling of remote vocabularies in :class:`dask_ml.feature_extraction.text.HashingVectorizer` (:pr:`719`)
* Added :func:`dask_ml.metrics.regression.mean_squared_log_error` (:pr:`725`)
* Allow user-provided categories in :class:`dask_ml.preprocessing.OneHotEncoder` (:pr:`727`)
* Added :meth:`dask_ml.linear_model.LogisticRegression.decision_function` (:pr:`728`)
* Added ``compute`` argument to :class:`dask_ml.decomposition.TruncatedSVD` (:pr:`743`)
* Fixed sign stability in incremental PCA (:pr:`742`)

Version 1.6.0
~~~~~~~~~~~~~

* Improved documentation for RandomizedSearchCV
* Improved logging in :class:`dask_ml.cluster.KMeans`  (:pr:`688`)
* Added support for ``dask.dataframe`` objects in :class:`dask_ml.model_selection.HyperbandSearchCV` (:pr:`701`)
* Added ``squared=True`` option to :class:`dask_ml.metrics.mean_squared_error` (:pr:`707`)
* Added :class:`dask_ml.feature_extraction.text.CountVectorizer` (:pr:`705`)

Version 1.5.0
~~~~~~~~~~~~~

* Support for Python 3.8 (:pr:`669`)
* Compatibility with Scikit-Learn 0.23.0 (:pr:`669`)
* Scikit-Learn 0.23.0 or newer is now required (:pr:`669`)
* Removed previously deprecated Partial classes. Use :class:`dask_ml.wrappers.Incremental` instead (:pr:`674`)

Version 1.4.0
~~~~~~~~~~~~~

* Added :class:`dask_ml.decomposition.IncrementalPCA` for out-of-core / distributed incremental PCA (:pr:`619`)
* Improved logging and monitoring in incremental model selection (:pr:`528`)
* Added :class:`dask_ml.ensemble.BlockwiseVotingClassifier` and :class:`dask_ml.ensemble.BlockwiseVotingRegressor` for blockwise training and ensemble prediction (:pr:`657`)
* Improved documentation for :ref:`hyper-parameter-search` (:pr:`432`)

Version 1.3.0
~~~~~~~~~~~~~

- Added ``shuffle`` support to :func:`dask_ml.model_selection.train_test_split` for ``DataFrame`` input (:pr:`625`)
- Improved performance of :class:`dask_ml.model_selection.GridSearchCV` by re-using cached tasks (:pr:`622`)
- Add support for ``DataFrame`` to :class:`dask_ml.model_selection.GridSearchCV` (:pr:`612`)
- Fixed :meth:`dask_ml.linear_model.LinearRegression.score` to use ``r2_score`` rather than ``mse`` (:pr:`614`)
- Handle missing data in :class:`dask_ml.preprocessing.StandardScaler` (:pr:`608`)

Version 1.2.0
~~~~~~~~~~~~~

- Changed the name of the second positional argument in :class:`model_selection.IncrementalSearchCV` from ``param_distribution``
  to ``parameters`` to match the name of the base class.
- Compatibility with scikit-learn 0.22.1.
- Added :class:`dask_ml.preprocessing.BlockTransfomer` an extension of scikit-learn's `FunctionTransformer` (:issue:`366`).
- Added :class:`dask_ml.feature_extraction.FeatureHasher` which is similar to scikit-learn's implementation.

Version 1.1.1
~~~~~~~~~~~~~

- Fixed an issue with the 1.1.0 wheel (:issue:`575`)
- Make svd_flip work even when arrays are read only (:issue:`592`)

Version 1.1.0
~~~~~~~~~~~~~

- Non-arrays (e.g. Dask Bags and DataFrames) are now allowed in :class:`dask_ml.wrappers.Incremental`. This is useful for text classification pipelines (pr:`570`)
- The index is now preserved in :class:`dask_ml.preprocessing.PolynomialFeatures` for DataFrame inputs (:pr:`563`)
- :class:`dask_ml.decomposition.PCA` now works with DataFrame inputs (:pr:`543`)
- :class:`dask_ml.cluster.KMeans` handles inputes where some blocks are length-0 (:pr:`559`)
- Improved error reporting for mixed inputs to :func:`dask_ml.model_selection.train_test_split` (:pr:`552`)
- Removed deprecated ``dask_ml.joblib`` module. Use ``joblib.parallel_backend`` instead (:pr:`545`)
- :class:`dask_ml.preprocessing.QuantileTransformer` now handles DataFrame input (:pr:`533`)


Version 1.0.0
~~~~~~~~~~~~~

- Added new hyperparameter search meta-estimators for hyperparameter search on distributed datasets: :class:`~dask_ml.model_selection.HyperbandSearchCV` and :class:`~dask_ml.model_selection.SuccessiveHalvingSearchCV`
- Dropped Python 2 support (:pr:`500`)

Version 0.13.0
~~~~~~~~~~~~~~

- Compatibility with scikit-learn 0.21.1
- Cross-validation results in ``GridSearchCV`` and ``RandomizedSearchCV`` are now gathered as completed, in case a worker is lost (:issue:`433`)
- Fixed bug in :func:`dask_ml.model_selection.train_test_split` when only one of train / test size is provided (:issue:`502`)
- Consistent random state for :class:`dask_ml.model_selection.IncrementalSearchCV`
- Fixed various issues with 32-bit Windows builds (:issue:`487`)

.. note::

   dask-ml 0.13.0 will be the last release to support Python 2.

Version 0.12.0
~~~~~~~~~~~~~~

API Breaking Changes
--------------------

- :class:`dask_ml.model_selection.IncrementalSearchCV` now returns Dask objects for post-fit methods like ``.predict``, etc (:issue:`423`).


Version 0.11.0
~~~~~~~~~~~~~~

Note that this version of Dask-ML requires scikit-learn >= 0.20.0.

Enhancements
------------

- Added :class:`dask_ml.model_selection.IncrementalSearchCV`, a meta-estimator for hyperparamter optimization on larger-than-memory datasets (:pr:`356`). See :ref:`hyperparameter.incremental` for more.
- Added :class:`dask_ml.preprocessing.PolynomialTransformer`, a drop-in replacement for the scikit-learn version (:issue:`347`).
- Added auto-rechunking to Dask Arrays with more than one block along the features in :class:`dask_ml.model_selection.ParallelPostFit` (:issue:`376`)
- Added support for Dask DataFrame inputs to :class:`dask_ml.cluster.KMeans` (:issue:`390`)
- Added a ``compute`` keyword to :meth:`dask_ml.wrappers.ParallelPostFit.score` to support lazily evaluating a model's score (:pr:`402`)

Bug Fixes
---------

- Changed :class:`dask_ml.wrappers.ParallelPostFit` to automatically rechunk input arrays to methods like ``predict`` when they
  have more than one block along the features (:issue:`376`).
- Bug in :class:`dask_ml.impute.SimpleImputer` with Dask DataFrames filling the count of the most frequent item, rather than the item itself (:issue:`385`).
- Bug in :class:`dask_ml.model_selection.ShuffleSplit` returning the same split when the ``random_state`` was set (:issue:`380`).

Version 0.10.0
~~~~~~~~~~~~~~

Enhancements
------------

- Added support for :class:`dask.dataframe.DataFrame` to :meth:`dask_ml.model_selection.train_test_split` (:issue:`351`)

Version 0.9.0
~~~~~~~~~~~~~

Enhancements
------------

- Added :class:`dask_ml.model_selection.ShuffleSplit` (:pr:`340`)

Bug Fixes
---------

- Fixed handling of errors in the predict and score steps of :class:`dask_ml.model_selection.GridSearchCV` and :class:`dask_ml.model_selection.RandomizedSearchCV` (:pr:`339`)
- Compatability with Dask 0.18 for :class:`dask_ml.preprocessing.LabelEncoder` (you'll also notice improved performance) (:pr:`336`).

Documentation Updates
---------------------

- Added a :ref:`roadmap`. Please `open an issue <https://github.com/dask/dask-ml>`__ if you'd like something to be included on the roadmap. (:pr:`322`)
- Added many :ref:`examples` to the documentation and the `dask examples <https://github.com/dask/dask-examples>`__ binder.

Build Changes
-------------

We're now using `Numba <http://numba.pydata.org/>`__ for performance-sensitive parts of Dask-ML.
Dask-ML is now a pure-python project, so we can provide universal wheels.

Version 0.8.0
~~~~~~~~~~~~~

Enhancements
------------

- Automatically replace default scikit-learn scorers with dask-aware versions in Incremental (:issue:`200`)
- Added the :func:`dask_ml.metrics.log_loss` loss function and ``neg_log_loss`` scorer (:pr:`318`)
- Fixed handling of array-like fit-parameters to GridSearchCV and BaseSearchCV (:pr:`320`)

Bug Fixes
---------

- Fixed dtype in :meth:`LabelEncoder.fit_transform` to be integer, rather than the dtype of the classes for dask arrays (:pr:`311`)

Version 0.7.0
~~~~~~~~~~~~~

Enhancements
------------

- Added ``sample_weight`` support for :meth:`dask_ml.metrics.accuracy_score`. (:pr:`217`)
- Improved performance of training on :class:`dask_ml.cluster.SpectralClustering` (:pr:`152`)
- Added :class:`dask_ml.preprocessing.LabelEncoder`. (:pr:`226`)
- Fixed issue in ``model_selection`` meta-estimators not respecting the default Dask scheduler (:pr:`260`)

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

Reorganization
--------------

Dask-SearchCV is now being developed in the ``dask/dask-ml`` repository. Users
who previously installed ``dask-searchcv`` should now just install ``dask-ml``.

Bug Fixes
---------

- Fixed random seed generation on 32-bit platforms (:issue:`230`)


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
