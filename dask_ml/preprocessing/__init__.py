"""Utilties for Preprocessing data.
"""
from .._compat import SK_VERSION
from packaging.version import parse


from .data import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    Categorizer,
    DummyEncoder,
    OrdinalEncoder,
)
from .label import LabelEncoder


__all__ = [
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "QuantileTransformer",
    "Categorizer",
    "DummyEncoder",
    "OrdinalEncoder",
    "LabelEncoder",
]

if SK_VERSION >= parse("0.20.0.dev0"):
    from ._encoders import OneHotEncoder  # noqa

    __all__.append("OneHotEncoder")

del SK_VERSION
del parse
