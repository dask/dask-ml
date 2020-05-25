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

SK_022 = SK_VERSION >= packaging.version.parse("0.22")
SK_0221 = SK_VERSION >= packaging.version.parse("0.22.1")
DASK_240 = DASK_VERSION >= packaging.version.parse("2.4.0")
DASK_2130 = DASK_VERSION >= packaging.version.parse("2.13.0")
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
    if SK_022:
        args: Any = ()
    else:
        args = (attributes,)

    return sklearn.utils.validation.check_is_fitted(est, *args)


def _check_multimetric_scoring(estimator, scoring=None):
    if SK_022:
        from sklearn.metrics._scorer import _check_multimetric_scoring
    else:
        from sklearn.metrics.scorer import _check_multimetric_scoring

    return _check_multimetric_scoring(estimator, scoring)
