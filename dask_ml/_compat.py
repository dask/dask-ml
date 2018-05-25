import contextlib

import packaging.version
import sklearn


SK_VERSION = packaging.version.parse(sklearn.__version__)


@contextlib.contextmanager
def dummy_context(*args, **kwargs):
    yield
