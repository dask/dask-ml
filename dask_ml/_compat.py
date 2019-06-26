import contextlib
from collections.abc import Mapping  # noqa

import dask
import dask.array as da
import packaging.version
import pandas
import sklearn

SK_VERSION = packaging.version.parse(sklearn.__version__)
DASK_VERSION = packaging.version.parse(dask.__version__)
PANDAS_VERSION = packaging.version.parse(pandas.__version__)

SK_022 = SK_VERSION >= packaging.version.parse("0.22.dev0")
DASK_200 = DASK_VERSION >= packaging.version.parse("2.0.0")


@contextlib.contextmanager
def dummy_context(*args, **kwargs):
    yield


if DASK_VERSION < packaging.version.parse("1.1.0"):
    blockwise = da.atop
else:
    blockwise = da.blockwise
