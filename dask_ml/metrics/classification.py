from typing import Optional

import dask
import dask.array as da
import numpy as np
import sklearn.metrics
import sklearn.utils.multiclass

from .._typing import ArrayLike


def accuracy_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    normalize: bool = True,
    sample_weight: Optional[ArrayLike] = None,
    compute: bool = True,
) -> ArrayLike:
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Read more in the :ref:`User Guide <accuracy_score>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    sample_weight : 1d array-like, optional
        Sample weights.

        .. versionadded:: 0.7.0

    Returns
    -------
    score : scalar dask Array
        If ``normalize == True``, return the correctly classified samples
        (float), else it returns the number of correctly classified samples
        (int).

        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.

    Notes
    -----
    In binary and multiclass classification, this function is equal
    to the ``jaccard_similarity_score`` function.

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> from dask_ml.metrics import accuracy_score
    >>> y_pred = da.from_array(np.array([0, 2, 1, 3]), chunks=2)
    >>> y_true = da.from_array(np.array([0, 1, 2, 3]), chunks=2)
    >>> accuracy_score(y_true, y_pred)
    dask.array<mean_agg-aggregate, shape=(), dtype=float64, chunksize=()>
    >>> _.compute()
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False).compute()
    2

    In the multilabel case with binary label indicators:

    >>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    0.5
    """

    if y_true.ndim > 1:
        differing_labels = ((y_true - y_pred) == 0).all(1)
        score = differing_labels != 0
    else:
        score = y_true == y_pred

    if normalize:
        score = da.average(score, weights=sample_weight)
    elif sample_weight is not None:
        score = da.dot(score, sample_weight)
    else:
        score = score.sum()

    if compute:
        score = score.compute()
    return score


def _log_loss_inner(
    x: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike], **kwargs
):
    # da.map_blocks wasn't able to concatenate together the results
    # when we reduce down to a scalar per block. So we make an
    # array with 1 element.
    if sample_weight is not None:
        sample_weight = sample_weight.ravel()
    return np.array(
        [sklearn.metrics.log_loss(x, y, sample_weight=sample_weight, **kwargs)]
    )


def log_loss(
    y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None
):
    if not (dask.is_dask_collection(y_true) and dask.is_dask_collection(y_pred)):
        return sklearn.metrics.log_loss(
            y_true,
            y_pred,
            eps=eps,
            normalize=normalize,
            sample_weight=sample_weight,
            labels=labels,
        )

    if y_pred.ndim > 1 and y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        drop_axis: Optional[int] = 1
        if sample_weight is not None:
            sample_weight = sample_weight.reshape(-1, 1)
    else:
        drop_axis = None

    result = da.map_blocks(
        _log_loss_inner,
        y_true,
        y_pred,
        sample_weight,
        chunks=(1,),
        drop_axis=drop_axis,
        dtype="f8",
        eps=eps,
        normalize=normalize,
        labels=labels,
    )
    if normalize and sample_weight is not None:
        sample_weight = sample_weight.ravel()
        block_weights = sample_weight.map_blocks(np.sum, chunks=(1,), keepdims=True)
        return da.average(result, 0, weights=block_weights)
    elif normalize:
        return result.mean()
    else:
        return result.sum()


log_loss.__doc__ = getattr(sklearn.metrics.log_loss, "__doc__")
