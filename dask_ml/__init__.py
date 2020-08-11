from pkg_resources import DistributionNotFound, get_distribution

# Ensure we always register tokenizers
from dask_ml.model_selection import _normalize

__all__ = []

try:
    __version__ = get_distribution(__name__).version
    __all__.append("__version__")
except DistributionNotFound:
    # package is not installed
    pass


del DistributionNotFound
del get_distribution
del _normalize
