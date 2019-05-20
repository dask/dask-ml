import contextlib

import dask
import dask.array as da
import packaging.version
import pandas
import six
import sklearn

SK_VERSION = packaging.version.parse(sklearn.__version__)
DASK_VERSION = packaging.version.parse(dask.__version__)
PANDAS_VERSION = packaging.version.parse(pandas.__version__)

SK_022 = SK_VERSION >= packaging.version.parse("0.22.dev0")


@contextlib.contextmanager
def dummy_context(*args, **kwargs):
    yield


if six.PY2:
    from collections import Mapping
else:
    from collections.abc import Mapping  # noqa

if DASK_VERSION < packaging.version.parse("1.1.0"):
    blockwise = da.atop
else:
    blockwise = da.blockwise
