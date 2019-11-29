Preprocessing
=============

:mod:`dask_ml.preprocessing` contains some scikit-learn style transformers that
can be used in ``Pipelines`` to perform various data transformations as part
of the model fitting process. These transformers will work well on dask
collections (``dask.array``, ``dask.dataframe``), NumPy arrays, or pandas
dataframes. They'll fit and transform in parallel.

Scikit-Learn Clones
-------------------

Some of the transformers are (mostly) drop-in replacements for their
scikit-learn counterparts.

.. currentmodule:: dask_ml.preprocessing

.. autosummary::

   MinMaxScaler
   QuantileTransformer
   RobustScaler
   StandardScaler
   LabelEncoder
   OneHotEncoder
   PolynomialFeatures

These can be used just like the scikit-learn versions, except that:

1. They operate on dask collections in parallel
2. ``.transform`` will return a ``dask.array`` or ``dask.dataframe``
   when the input is a dask collection

See :mod:`sklearn.preprocessing` for more information about any particular
transformer. Scikit-learn does have some transforms that are alternatives to
the large-memory tasks that Dask serves. These include `FeatureHasher`_ (a
good alternative to `DictVectorizer`_ and `CountVectorizer`_) and `HashingVectorizer`_
(best suited for use in text over `CountVectorizer`_). They are not
stateful, which allows easy use with Dask with ``map_partitions``:

.. ipython:: python

    import dask.bag as db
    from sklearn.feature_extraction import FeatureHasher

    D = [{'dog': 1, 'cat':2, 'elephant':4}, {'dog': 2, 'run': 5}]
    b = db.from_sequence(D)
    h = FeatureHasher()

    b.map_partitions(h.transform).compute()

.. _FeatureHasher: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
.. _HashingVectorizer: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
.. _DictVectorizer: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
.. _CountVectorizer: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

.. note::

   :class:`dask_ml.preprocessing.LabelEncoder` and
   :class:`dask_ml.preprocessing.OneHotEncoder`
   will use the categorical dtype information for a dask or pandas Series with
   a :class:`pandas.api.types.CategoricalDtype`.
   This improves performance, but may lead to different encodings depending on the
   categories. See the class docstrings for more.

Encoding Categorical Features
-----------------------------

:class:`dask_ml.preprocessing.OneHotEncoder` can be useful for "one-hot" (or
"dummy") encoding features.

See `the scikit-learn documentation <http://scikit-learn.org/dev/modules/preprocessing.html#preprocessing-categorical-features>`_
for a full discussion. This section focuses only on the differences from
scikit-learn.

Dask-ML Supports pandas' Categorical dtype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dask-ML supports and uses the type information from pandas Categorical dtype.
See https://pandas.pydata.org/pandas-docs/stable/categorical.html for an introduction.
For large datasets, using categorical dtypes is crucial for achieving performance.

This will have a couple effects on the learned attributes and transformed
values.

1. The learned ``categories_`` may differ. Scikit-Learn requires the categories
   to be sorted. With a ``CategoricalDtype`` the categories do not need to be sorted.
2. The output of :meth:`OneHotEncoder.transform` will be the same type as the
   input. Passing a pandas DataFrame returns a pandas Dataframe, instead of a
   NumPy array. Likewise, a Dask DataFrame returns a Dask DataFrame.

Dask-ML's Sparse Support
~~~~~~~~~~~~~~~~~~~~~~~~

The default behavior of OneHotEncoder is to return a sparse array. Scikit-Learn
returns a SciPy sparse matrix for ndarrays passed to ``transform``.

When passed a Dask Array, :meth:`OneHotEncoder.transform` returns a Dask Array
*where each block is a scipy sparse matrix*. SciPy sparse matricies don't
support the same API as the NumPy ndarray, so most methods won't work on the
result. Even basic things like ``compute`` will fail. To work around this,
we currently recommend converting the sparse matricies to dense.

.. code-block:: python

   from dask_ml.preprocessing import OneHotEncoder
   import dask.array as da
   import numpy as np

   enc = OneHotEncoder(sparse=True)
   X = da.from_array(np.array([['A'], ['B'], ['A'], ['C']]), chunks=2)
   enc = enc.fit(X)
   result = enc.transform(X)
   result

Each block of ``result`` is a scipy sparse matrix

.. code-block:: python

   result.blocks[0].compute()
   # This would fail!
   # result.compute()
   # Convert to, say, pydata/sparse COO matricies instead
   from sparse import COO

   result.map_blocks(COO.from_scipy_sparse, dtype=result.dtype).compute()

Dask-ML's sparse support for sparse data is currently in flux. Reach out if you
have any issues.

Additional Tranformers
----------------------

Other transformers are specific to dask-ml.

.. autosummary::

   Categorizer
   DummyEncoder
   OrdinalEncoder


Both :class:`dask_ml.preprocessing.Categorizer` and
:class:`dask_ml.preprocessing.DummyEncoder` deal with converting non-numeric
data to numeric data. They are useful as a preprocessing step in a pipeline
where you start with heterogenous data (a mix of numeric and non-numeric), but
the estimator requires all numeric data.

In this toy example, we use a dataset with two columns. ``'A'`` is numeric and
``'B'`` contains text data. We make a small pipeline to

1. Categorize the text data
2. Dummy encode the categorical data
3. Fit a linear regression

.. ipython:: python

   from dask_ml.preprocessing import Categorizer, DummyEncoder
   from sklearn.linear_model import LogisticRegression
   from sklearn.pipeline import make_pipeline
   import pandas as pd
   import dask.dataframe as dd

   df = pd.DataFrame({"A": [1, 2, 1, 2], "B": ["a", "b", "c", "c"]})
   X = dd.from_pandas(df, npartitions=2)
   y = dd.from_pandas(pd.Series([0, 1, 1, 0]), npartitions=2)

   pipe = make_pipeline(
      Categorizer(),
      DummyEncoder(),
      LogisticRegression(solver='lbfgs')
   )
   pipe.fit(X, y)

``Categorizer`` will convert a subset of the columns in ``X`` to categorical
dtype (see `here <http://pandas.pydata.org/pandas-docs/stable/categorical.html>`_
for more about how pandas handles categorical data). By default, it converts all
the ``object`` dtype columns.

``DummyEncoder`` will dummy (or one-hot) encode the dataset. This replaces a
categorical column with multiple columns, where the values are either 0 or 1,
depending on whether the value in the original.

.. ipython:: python

   df['B']
   pd.get_dummies(df['B'])

Wherever the original was ``'a'``, the transformed now has a ``1`` in the ``a``
column and a ``0`` everywhere else.

Why was the ``Categorizer`` step necessary? Why couldn't we operate directly
on the ``object`` (string) dtype column? Doing this would be fragile,
especially when using ``dask.dataframe``, since *the shape of the output would
depend on the values present*. For example, suppose that we just saw the first
two rows in the training, and the last two rows in the tests datasets. Then,
when training, our transformed columns would be:

.. ipython:: python

   pd.get_dummies(df.loc[[0, 1], 'B'])

while on the test dataset, they would be:

.. ipython:: python

   pd.get_dummies(df.loc[[2, 3], 'B'])

Which is incorrect! The columns don't match.

When we categorize the data, we can be confident that all the possible values
have been specified, so the output shape no longer depends on the values in the
whatever subset of the data we currently see. Instead, it depends on the
``categories``, which are identical in all the subsets.
