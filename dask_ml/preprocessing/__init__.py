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
from .imputation import Imputer
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
    "Imputer",
]


# Requires scikit-learn >= 0.20.0
try:
    from ._encoders import OneHotEncoder  # noqa
except ImportError:
    pass
else:
    __all__.append("OneHotEncoder")
