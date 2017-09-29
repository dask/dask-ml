import dask.dataframe as dd
import dask.array as da


def slice_columns(X, columns):
    if isinstance(X, dd.DataFrame):
        return X[columns or list(X.columns)]
    else:
        return X


def handle_zeros_in_scale(scale):
    scale = scale.copy()
    if isinstance(scale, da.Array):
        scale[scale == 0.0] = 1.0
    elif isinstance(scale, dd.Series):
        scale = scale.map(lambda x: 1 if x == 0 else x)
    return scale
