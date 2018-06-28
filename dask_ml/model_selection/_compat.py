from distutils.version import LooseVersion
from sklearn import __version__

_SK_VERSION = LooseVersion(__version__)

_HAS_MULTIPLE_METRICS = _SK_VERSION >= '0.19.0'
