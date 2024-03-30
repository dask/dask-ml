# Ensure we always register tokenizers
from dask_ml.model_selection import _normalize  # noqa: F401

from ._version import __version__

__all__ = ["__version__"]
