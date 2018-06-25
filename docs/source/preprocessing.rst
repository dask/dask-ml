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

These can be used just like the scikit-learn versions, except that:

1. They operate on dask collections in parallel
2. ``.transform`` will return a ``dask.array`` or ``dask.dataframe``
   when the input is a dask collection

See :mod:`sklearn.preprocessing` for more information about any particular
transformer.

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
      LogisticRegression()
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

Why was the ``Categorizizer`` step necessary? Why couldn't we operate directly
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
