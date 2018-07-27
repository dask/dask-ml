from .pairwise import (  # noqa
    pairwise_distances,
    pairwise_distances_argmin_min,
    euclidean_distances,
)
from .regression import mean_absolute_error, mean_squared_error, r2_score  # noqa
from .classification import accuracy_score, log_loss  # noqa

from .scorer import get_scorer, check_scoring, SCORERS  # noqa
