"""Utilties for Preprocessing data.
"""
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


# Requires scikit-learn >= 0.20.0
try:
    from ._encoders import OneHotEncoder
except ImportError:
    pass
else:
    __all__.append("OneHotEncoder")
