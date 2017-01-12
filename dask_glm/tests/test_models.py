from dask_glm.models import Optimizer

def generate_2pt_line():
    '''Generates trivial data.'''
    X = da.from_array(np.array([[0], [1]]), chunks=2)
    y = da.from_array(np.array([[0], [1]]), chunks=2)
    return y,X
