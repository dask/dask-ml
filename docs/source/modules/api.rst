.. _api:

=============
API Reference
=============

:mod:`dask_ml.model_selection`: Model Selection
===============================================

.. automodule:: dask_ml.model_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.GridSearchCV
   model_selection.RandomizedSearchCV
   sklearn.pipeline.make_pipeline


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

Incremental Learning
====================

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   cluster.PartialMiniBatchKMeans
   linear_model.PartialPassiveAggressiveClassifier
   linear_model.PartialPassiveAggressiveRegressor
   linear_model.PartialPerceptron
   linear_model.PartialSGDClassifier
   linear_model.PartialSGDRegressor
   naive_bayes.PartialBernoulliNB
   naive_bayes.PartialMultinomialNB


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


:mod:`dask_ml.decomponosition`: Matrix Decomposition
====================================================

.. automodule:: dask_ml.decompositoin
   :no-members:
   :no-inherited-members:

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   decomposition.TruncatedSVD



:mod:`dask_ml.preprocessing`: Preprocessing Data
================================================

.. automodule:: dask_ml.preprocessing

.. currentmodule:: dask_ml

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.Imputer
   preprocessing.MinMaxScaler
   preprocessing.QuantileTransformer
   preprocessing.StandardScaler
   preprocessing.Categorizer
   preprocessing.DummyEncoder


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
