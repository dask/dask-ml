"""Train an XGBoost model on dask arrays or dataframes.

This may be used for training an XGBoost model on a cluster. XGBoost
will be setup in distributed mode alongside your existing
``dask.distributed`` cluster.
"""
from dask_xgboost import *  # noqa
