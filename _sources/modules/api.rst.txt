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

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.GridSearchCV
   model_selection.RandomizedSearchCV


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

Meta-estimators for scikit-learn
================================

dask-ml provides some meta-estimators that help use regular scikit-learn
compatible estimators with Dask arrays.

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

:mod:`dask_ml.impute`: Imputing Missing Data
============================================

.. automodule:: dask_ml.impute

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.impute.SimpleImputer


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
   metrics.r2_score


Classification Metrics
----------------------

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/

   metrics.accuracy_score
   metrics.log_loss


:mod:`dask_ml.tensorflow`: Tensorflow
=====================================

.. automodule:: dask_ml.tensorflow

.. currentmodule:: dask_ml.tensorflow

.. autosummary::
   :toctree: generated/

   start_tensorflow



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
