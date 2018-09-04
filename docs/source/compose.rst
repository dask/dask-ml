.. _compose:

Pipelines and Composite Estimators
==================================

Dask-ML estimators follow the scikit-learn API. This means Dask-ML estimators like
:class:`dask_ml.decomposition.PCA` can be placed inside a regular :class:`sklearn.pipeline.Pipeline`.

See http://scikit-learn.org/dev/modules/compose.html for more on using pipelines in general.

.. ipython:: python

   from sklearn.pipeline import Pipeline  # regular scikit-learn pipeline
   from dask_ml.cluster import KMeans
   from dask_ml.decomposition import PCA
   estimators = [('reduce_dim', PCA()), ('cluster', KMeans())]
   pipe = Pipeline(estimators)
   pipe

The pipeline ``pipe`` can now be used with Dask arrays.

ColumnTransformer for Heterogeneous Data
----------------------------------------

:class:`dask_ml.compose.ColumnTransformer` is a clone of the scikit-learn version that works well
with Dask objects.

See http://scikit-learn.org/dev/modules/compose.html#columntransformer-for-heterogeneous-data for an
introduction to ``ColumnTransformer``.
