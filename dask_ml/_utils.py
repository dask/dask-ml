def copy_learned_attributes(from_estimator, to_estimator):
    attrs = {k: v for k, v in vars(from_estimator).items() if k.endswith("_")}

    for k, v in attrs.items():
        setattr(to_estimator, k, v)


def draw_seed(random_state, low=0, high=None, size=None, dtype=None, chunks=None):
    kwargs = {"size": size}
    if chunks is not None:
        kwargs["chunks"] = chunks
    if dtype is not None:
        kwargs["dtype"] = dtype
    return random_state.randint(low, high, **kwargs)
