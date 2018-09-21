"""Meta-estimators for composing models with multiple transformers.

These estimators are useful for working with heterogenous tabular data.
"""
from .._compat import SK_VERSION
from packaging.version import parse

__all__ = []

if SK_VERSION >= parse("0.20.0.dev0"):
    from ._column_transformer import ColumnTransformer, make_column_transformer  # noqa

    __all__.extend(["ColumnTransformer", "make_column_transformer"])

del SK_VERSION
del parse
