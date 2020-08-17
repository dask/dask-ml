.. _api:

=============
API Reference
=============

This page lists all of the estimators and top-level functions in ``dask_ml``.
Unless otherwise noted, the estimators implemented in ``dask-ml`` are
appropriate for parallel and distributed training.

:mod:`dask_ml.model_selection`: Model Selection
===============================================

.. automodule:: dask_ml.model_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: dask_ml

Dask-ML has a few cross validation utilities.

.. autosummary::
   :toctree: generated/

   model_selection.train_test_split

:func:`model_selection.train_test_split` is a simple helper that
uses :class:`model_selection.ShuffleSplit` internally.

.. autosummary::
   :toctree: generated
   :template: class.rst

   model_selection.ShuffleSplit
   model_selection.KFold

Dask-ML provides drop-in replacements for grid and randomized search.
These are appropriate for datasets where the CV splits fit in memory.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.GridSearchCV
   model_selection.RandomizedSearchCV

For hyperparameter optimization on larger-than-memory datasets, Dask-ML
provides the following:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.IncrementalSearchCV
   model_selection.HyperbandSearchCV
   model_selection.SuccessiveHalvingSearchCV
   model_selection.InverseDecaySearchCV


:mod:`dask_ml.ensemble`: Ensemble Methods
=========================================

.. automodule:: dask_ml.ensemble
   :no-members:
   :no-inherited-members:

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ensemble.BlockwiseVotingClassifier
   ensemble.BlockwiseVotingRegressor


:mod:`dask_ml.linear_model`: Generalized Linear Models
======================================================

.. automodule:: dask_ml.linear_model
   :no-members:
   :no-inherited-members:

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.LinearRegression
   linear_model.LogisticRegression
   linear_model.PoissonRegression

:mod:`dask_ml.wrappers`: Meta-Estimators
========================================

dask-ml provides some meta-estimators that help use regular
estimators that follow the scikit-learn API.
These meta-estimators make the underlying estimator work well
with Dask Arrays or DataFrames.

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   wrappers.ParallelPostFit
   wrappers.Incremental

:mod:`dask_ml.cluster`: Clustering
==================================

.. automodule:: dask_ml.cluster
   :no-members:
   :no-inherited-members:

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   cluster.KMeans
   cluster.SpectralClustering


:mod:`dask_ml.decomposition`: Matrix Decomposition
====================================================

.. automodule:: dask_ml.decomposition
   :no-members:
   :no-inherited-members:

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   decomposition.IncrementalPCA
   decomposition.PCA
   decomposition.TruncatedSVD


:mod:`dask_ml.preprocessing`: Preprocessing Data
================================================

.. automodule:: dask_ml.preprocessing

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.StandardScaler
   preprocessing.RobustScaler
   preprocessing.MinMaxScaler
   preprocessing.QuantileTransformer
   preprocessing.StandardScaler
   preprocessing.Categorizer
   preprocessing.DummyEncoder
   preprocessing.OrdinalEncoder
   preprocessing.LabelEncoder
   preprocessing.PolynomialFeatures
   preprocessing.BlockTransformer

:mod:`dask_ml.feature_extraction.text`: Feature extraction
==========================================================


.. automodule:: dask_ml.preprocessing.text:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_extraction.text.CountVectorizer
   feature_extraction.text.HashingVectorizer
   feature_extraction.text.FeatureHasher


:mod:`dask_ml.compose`: Composite Estimators
============================================

Meta-estimators for building composite models with transformers.

.. automodule:: dask_ml.compose

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generted/
   :template: class.rst

   compose.ColumnTransformer

.. autosummary::
   :toctree: generted/

   compose.make_column_transformer


:mod:`dask_ml.impute`: Imputing Missing Data
============================================

.. automodule:: dask_ml.impute

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   impute.SimpleImputer


:mod:`dask_ml.metrics`: Metrics
===============================

Score functions, performance metrics, and pairwise distance computations.

Regression Metrics
------------------

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/

   metrics.mean_absolute_error
   metrics.mean_squared_error
   metrics.mean_squared_log_error
   metrics.r2_score


Classification Metrics
----------------------

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/

   metrics.accuracy_score
   metrics.log_loss


:mod:`dask_ml.xgboost`: XGBoost
===============================

.. automodule:: dask_ml.xgboost

.. currentmodule:: dask_ml.xgboost

.. autosummary::
   :toctree: generated/
   :template: class.rst

   XGBClassifier
   XGBRegressor

.. autosummary::
   :toctree: generated/

   train
   predict

:mod:`dask_ml.datasets`: Datasets
======================================================

dask-ml provides some utilities for generating toy datasets.

.. automodule:: dask_ml.datasets

.. currentmodule:: dask_ml.datasets

.. autosummary::
   :toctree: generated/

   make_counts
   make_blobs
   make_regression
   make_classification
   make_classification_df
