XGBoost & LightGBM
==================

XGBoost_ is a powerful and popular library for gradient boosted trees.  For
larger datasets or faster training XGBoost also provides a distributed
computing solution. LightGBM_ is another library similar to XGBoost; it also
natively supplies native distributed training for decision trees.

Both XGBoost or LightGBM provided Dask implementations for distributed
training. These can take Dask objects like Arrays and DataFrames as input.
This allows one to do any initial loading and processing of data with Dask
before handing over to XGBoost/LightGBM to do what they do well.

The XGBoost implementation can be found at https://github.com/dmlc/xgboost and documentation can be found at
https://xgboost.readthedocs.io/en/latest/tutorials/dask.html.

The LightGBM implementation can be found at https://github.com/microsoft/LightGBM and documentation can be found at
https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html#dask.

.. _XGBoost: https://xgboost.readthedocs.io/
.. _LightGBM: https://lightgbm.readthedocs.io/
