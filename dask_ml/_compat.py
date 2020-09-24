import contextlib
import os
from collections.abc import Mapping  # noqa
from typing import Any, List, Optional, Union

import dask
import dask.array as da
import distributed
import packaging.version
import pandas
import sklearn
import sklearn.utils.validation

SK_VERSION = packaging.version.parse(sklearn.__version__)
DASK_VERSION = packaging.version.parse(dask.__version__)
PANDAS_VERSION = packaging.version.parse(pandas.__version__)
DISTRIBUTED_VERSION = packaging.version.parse(distributed.__version__)

SK_0_23_2 = SK_VERSION >= packaging.version.parse("0.23.2")
SK_024 = SK_VERSION >= packaging.version.parse("0.24.0.dev0")
DASK_240 = DASK_VERSION >= packaging.version.parse("2.4.0")
DASK_2130 = DASK_VERSION >= packaging.version.parse("2.13.0")
DASK_2_20_0 = DASK_VERSION >= packaging.version.parse("2.20.0")
DASK_2_26_0 = DASK_VERSION >= packaging.version.parse("2.26.0")
DASK_2_28_0 = DASK_VERSION > packaging.version.parse("2.27.0")
DISTRIBUTED_2_5_0 = DISTRIBUTED_VERSION > packaging.version.parse("2.5.0")
DISTRIBUTED_2_11_0 = DISTRIBUTED_VERSION > packaging.version.parse("2.10.0")  # dev
WINDOWS = os.name == "nt"


@contextlib.contextmanager
def dummy_context(*args: Any, **kwargs: Any):
    # Not needed if Python >= 3.7 is required
    # https://docs.python.org/3/library/contextlib.html#contextlib.nullcontext
    yield


blockwise = da.blockwise


def check_is_fitted(est, attributes: Optional[Union[str, List[str]]] = None):
    args: Any = ()

    return sklearn.utils.validation.check_is_fitted(est, *args)


def _check_multimetric_scoring(estimator, scoring=None):
    # TODO: See if scikit-learn 0.24 solves the need for using
    # a private method
    from sklearn.metrics._scorer import _check_multimetric_scoring
    from sklearn.metrics import check_scoring

    if SK_024:
        if callable(scoring) or isinstance(scoring, (type(None), str)):
            scorers = {"score": check_scoring(estimator, scoring=scoring)}
            return scorers, False
        return _check_multimetric_scoring(estimator, scoring), True
    return _check_multimetric_scoring(estimator, scoring)
