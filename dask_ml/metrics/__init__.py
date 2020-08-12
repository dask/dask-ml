from .classification import accuracy_score, log_loss  # noqa
from .pairwise import (  # noqa
    euclidean_distances,
    pairwise_distances,
    pairwise_distances_argmin_min,
)
from .regression import (  # noqa
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error,
)
from .scorer import SCORERS, check_scoring, get_scorer  # noqa
