import contextlib

import packaging.version
import sklearn
import dask


SK_VERSION = packaging.version.parse(sklearn.__version__)
DASK_VERSION = packaging.version.parse(dask.__version__)


@contextlib.contextmanager
def dummy_context(*args, **kwargs):
    yield
