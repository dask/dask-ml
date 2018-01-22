"""
"""
from multipledispatch import dispatch
import dask.dataframe as dd
import dask.array as da


@dispatch(dd._Frame)
def exp(A):
    return da.exp(A)


@dispatch(dd._Frame)
def absolute(A):
    return da.absolute(A)


@dispatch(dd._Frame)
def sign(A):
    return da.sign(A)


@dispatch(dd._Frame)
def log1p(A):
    return da.log1p(A)


# @dispatch(da.Array, da.Array)
# def dot(A, B):
#     return da.dot(A, B)


@dispatch(dd.DataFrame)
def add_intercept(X):
    columns = X.columns
    if 'intercept' in columns:
        raise ValueError("'intercept' column already in 'X'")
    return X.assign(intercept=1)[['intercept'] + list(columns)]
