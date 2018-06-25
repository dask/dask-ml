from .pairwise import (  # noqa
    pairwise_distances,
    pairwise_distances_argmin_min,
    euclidean_distances,
)
from .regression import (  # noqa
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from .classification import (  # noqa
    accuracy_score,
)

from .scorer import (  # noqa
    get_scorer,
    check_scoring,
    SCORERS,
)
