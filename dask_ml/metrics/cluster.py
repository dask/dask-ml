from dask.array.random import doc_wraps
import dask.array as da
from sklearn import metrics
import dask
from math import log
import numpy as np


def contingency_matrix(labels_true, labels_pred, n_labels):
    # Since uniqueness is global phenomenon not local to a block
    if n_labels == None:
        classes = (da.unique(labels_true)).compute()
        n_labels = classes.shape[0]

    contingency = da.bincount(n_labels * (labels_true) + (labels_pred), minlength=n_labels*n_labels)
    return contingency.reshape(n_labels, n_labels)

# if contingency==None:
#             contingency = contingency_matrix(labels_true, labels_pred, n_labels)
        
#         contingency_sum = contingency.sum()
#         pi = contingency.sum(axis=1)
#         pj = contingency.sum(axis=0)
#         nzx, nzy = da.nonzero(contingency)
#         outer = da.take(pi,nzx) * da.take(pj,nzy)
        
#         contingency = contingency.compute()
#         nz_val = contingency[nzx, nzy]

#         log_contingency_nm = da.log(nz_val)
#         contingency_nm = nz_val / contingency_sum

#         # Don't need to calculate the full outer product, just for non-zeroes
#         log_outer = -da.log(outer) + da.log(pi.sum()) + da.log(pj.sum())

#         log_outer = log_outer.compute()
#         mi = (contingency_nm * (log_contingency_nm - da.log(contingency_sum)) +
#             contingency_nm * log_outer)

#         return mi.sum()

from dask import delayed

@doc_wraps(metrics.mutual_info_score)
def mutual_info_score(labels_true, labels_pred, n_labels=None,contingency=None, compute=True):
    if not (dask.is_dask_collection(labels_true) and dask.is_dask_collection(labels_pred) and (dask.is_dask_collection(contingency) or contingency == None)):
            return metrics.mutual_info_score(
                labels_true,
                labels_pred,
                contingency
            )
    else:
        if n_labels == None:
            classes = (da.unique(labels_true)).compute()
            n_labels = classes.shape[0]
        
        contingency = da.bincount(n_labels * (labels_true) + (labels_pred), minlength=n_labels*n_labels)
        contingency = contingency.reshape(n_labels, n_labels)
        contingency_sum = contingency.sum()

        pi = da.ravel(contingency.sum(axis=1))
        pj = da.ravel(contingency.sum(axis=0))
        
        outer = pi * pj

        # x = da.flatnonzero(contingency)
        # nz_val = da.take(contingency.flatten(),x)
        
        log_contingency_nm = da.log(contingency.flatten())
        contingency_nm =  contingency.flatten() / contingency_sum

        log_outer = -da.log(outer) + da.log(pi.sum()) + da.log(pj.sum())

        with np.errstate(all="ignore"):
            mi = (contingency_nm * (log_contingency_nm - da.log(contingency_sum)) +
                contingency_nm * log_outer)

        if compute:
            return mi.sum().compute()

        return mi.sum()
        

        