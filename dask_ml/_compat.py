import contextlib

import dask
import packaging.version
import pandas
import six
import sklearn

SK_VERSION = packaging.version.parse(sklearn.__version__)
DASK_VERSION = packaging.version.parse(dask.__version__)
PANDAS_VERSION = packaging.version.parse(pandas.__version__)


@contextlib.contextmanager
def dummy_context(*args, **kwargs):
    yield


if six.PY2:
    from collections import Mapping
else:
    from collections.abc import Mapping  # noqa
