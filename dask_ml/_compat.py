import contextlib

import dask
import packaging.version
import sklearn

SK_VERSION = packaging.version.parse(sklearn.__version__)
DASK_VERSION = packaging.version.parse(dask.__version__)


@contextlib.contextmanager
def dummy_context(*args, **kwargs):
    yield


SK_GE_020 = SK_VERSION >= packaging.version.parse("0.20.0.dev0")
