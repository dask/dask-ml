from multipledispatch import Dispatcher
import dask.array as da
import numpy as np

dot = Dispatcher('dot')
dot.register(np.ndarray, np.ndarray)(np.dot)
dot.register(da.Array, da.Array)(da.dot)


@dot.register(da.Array, np.ndarray)
def dot_dask_numpy(a, b):
    return da.dot(a, da.from_array(b, chunks=b.shape))


@dot.register(np.ndarray, da.Array)
def dot_numpy_dask(a, b):
    return da.dot(da.from_array(a, chunks=a.shape), b)


exp = Dispatcher('exp')
exp.register(np.ndarray)(np.exp)
exp.register(da.Array)(da.exp)


log1p = Dispatcher('log1p')
log1p.register(np.ndarray)(np.log1p)
log1p.register(da.Array)(da.log1p)
