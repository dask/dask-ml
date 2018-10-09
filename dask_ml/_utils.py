import dask.array as da
import numpy as np


def copy_learned_attributes(from_estimator, to_estimator):
    attrs = {k: v for k, v in vars(from_estimator).items() if k.endswith("_")}

    for k, v in attrs.items():
        setattr(to_estimator, k, v)


def draw_seed(random_state, low, high=None, size=None, dtype=None, chunks=None):
    kwargs = {"size": size}
    if chunks is not None:
        kwargs["chunks"] = chunks

    seed = random_state.randint(low, high, **kwargs)
    if dtype is not None and isinstance(seed, (da.Array, np.ndarray)):
        seed = seed.astype(dtype)

    return seed
