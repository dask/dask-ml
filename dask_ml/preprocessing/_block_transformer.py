import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import compute
from pandas.api.types import is_categorical_dtype
from scipy import stats
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted, check_random_state

from dask_ml._utils import copy_learned_attributes
from dask_ml.utils import check_array, handle_zeros_in_scale


class BlockTransformer(FunctionTransformer):
    pass
