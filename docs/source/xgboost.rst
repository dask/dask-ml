XGBoost
=======

.. currentmodule:: dask_ml.xgboost

.. autosummary::
   train
   predict
   XGBClassifier
   XGBRegressor

XGBoost_ is a powerful and popular library for gradient boosted trees.  For
larger datasets or faster training XGBoost also provides a distributed
computing solution.  Dask-ML can set up distributed XGBoost for you and hand
off data from distributed dask.dataframes.  This automates much of the hassle
of preprocessing and setup while still letting XGBoost do what it does well.

Example
-------

.. code-block:: python

   from dask.distributed import Client
   client = Client('scheduler-address:8786')

   import dask.dataframe as dd
   df = dd.read_parquet('s3://...')

   # Split into training and testing data
   train, test = df.random_split([0.8, 0.2])

   # Separate labels from data
   train_labels = train.x > 0
   test_labels = test.x > 0

   del train['x']  # remove informative column from data
   del test['x']  # remove informative column from data

   # from xgboost import XGBRegressor  # change import
   from dask_ml.xgboost import XGBRegressor

   est = XGBRegressor(...)
   est.fit(train, train_labels)

   prediction = est.predict(test)

How this works
--------------

Dask sets up XGBoost's master process on the Dask scheduler and XGBoost's worker
processes on Dask's worker processes.  Then it moves all of the Dask
dataframes' constituent Pandas dataframes to XGBoost and lets XGBoost train.
Fortunately, because XGBoost has an excellent Python interface, all of this can
happen in the same process without any data transfer.  The two distributed
services can operate together on the same data.

When XGBoost is finished training Dask cleans up the XGBoost infrastructure and
continues on as normal.

This work was a collaboration with XGBoost and SKLearn maintainers.  See
relevant GitHub issue here: `dmlc/xgboost #2032 <https://github.com/dmlc/xgboost/issues/2032>`_

- :doc:`examples/xgboost`

.. _XGBoost: https://xgboost.readthedocs.io/
