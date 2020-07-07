from .classification import accuracy_score, log_loss, poisson_deviance
from .pairwise import (
    euclidean_distances,
    pairwise_distances,
    pairwise_distances_argmin_min,
)
from .regression import mean_absolute_error, mean_squared_error, r2_score
from .scorer import SCORERS, check_scoring, get_scorer

__all__ = [
    "accuracy_score",
    "log_loss",
    "poisson_deviance",
    "euclidean_distances",
    "pairwise_distances",
    "pairwise_distances_argmin_min",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
    "SCORERS",
    "check_scoring",
    "get_scorer",
]
