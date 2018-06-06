def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None,
                   compute=True):
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

    sample_weight : None
        For compatibility with scikit-learn. This is currently not
        supported.

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

    if sample_weight is not None:
        raise ValueError("'sample_weight' is not supported.")

    if y_true.ndim > 1:
        differing_labels = ((y_true - y_pred) == 0).all(1)
        score = differing_labels != 0
    else:
        score = y_true == y_pred

    if normalize:
        score = score.mean()
    else:
        score = score.sum()

    if compute:
        score = score.compute()
    return score
