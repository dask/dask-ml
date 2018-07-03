import contextlib

import packaging.version
from distutils.version import LooseVersion
import sklearn
import dask


SK_VERSION = packaging.version.parse(sklearn.__version__)
_SK_VERSION = LooseVersion(sklearn.__version__)
_HAS_MULTIPLE_METRICS = _SK_VERSION >= '0.19.0'

DASK_VERSION = packaging.version.parse(dask.__version__)


@contextlib.contextmanager
def dummy_context(*args, **kwargs):
    yield
