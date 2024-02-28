from typing import Optional

import dask.array as da
import numpy as np
import sklearn.metrics
from dask import is_dask_collection
from dask.utils import derived_from

from .._typing import ArrayLike


def _check_sample_weight(sample_weight: Optional[ArrayLike]):
    if sample_weight is not None:
        raise ValueError("'sample_weight' is not supported.")


def _check_reg_targets(
    y_true: ArrayLike, y_pred: ArrayLike, multioutput: Optional[str]
):
    if multioutput is not None and (
        is_dask_collection(multioutput) or multioutput != "uniform_average"
    ):
        raise NotImplementedError("'multioutput' must be 'uniform_average'")

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    # TODO: y_type, multioutput
    return None, y_true, y_pred, multioutput


@derived_from(sklearn.metrics)
def mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Optional[str] = "uniform_average",
    squared: bool = True,
    compute: bool = True,
) -> ArrayLike:
    _check_sample_weight(sample_weight)
    output_errors = ((y_pred - y_true) ** 2).mean(axis=0)

    if isinstance(multioutput, str) or multioutput is None:
        if multioutput == "raw_values":
            if compute:
                return output_errors.compute()
            else:
                return output_errors
    else:
        raise ValueError("Weighted 'multioutput' not supported.")
    result = output_errors.mean()
    if not squared:
        result = da.sqrt(result)
    if compute:
        result = result.compute()
    return result


@derived_from(sklearn.metrics)
def mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Optional[str] = "uniform_average",
    compute: bool = True,
) -> ArrayLike:
    _check_sample_weight(sample_weight)
    output_errors = abs(y_pred - y_true).mean(axis=0)

    if isinstance(multioutput, str) or multioutput is None:
        if multioutput == "raw_values":
            if compute:
                return output_errors.compute()
            else:
                return output_errors
    else:
        raise ValueError("Weighted 'multioutput' not supported.")
    result = output_errors.mean()
    if compute:
        result = result.compute()
    return result


def mean_absolute_percentage_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Optional[str] = "uniform_average",
    compute: bool = True,
) -> ArrayLike:
    """Mean absolute percentage error regression loss.

    Note here that we do not represent the output as a percentage in range
    [0, 100]. Instead, we represent it in range [0, 1/eps]. Read more in
    https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    compute : bool
        Whether to compute this result (default ``True``)

    Returns
    -------
    loss : float or array-like of floats in the range [0, 1/eps]
        If multioutput is 'raw_values', then mean absolute percentage error
        is returned for each output separately.
        If multioutput is 'uniform_average' or ``None``, then the
        equally-weighted average of all output errors is returned.
        MAPE output is non-negative floating point. The best value is 0.0.
        But note the fact that bad predictions can lead to arbitarily large
        MAPE values, especially if some y_true values are very close to zero.
        Note that we return a large value instead of `inf` when y_true is zero.
    """
    _check_sample_weight(sample_weight)
    epsilon = np.finfo(np.float64).eps
    mape = abs(y_pred - y_true) / da.maximum(y_true, epsilon)
    output_errors = mape.mean(axis=0)

    if isinstance(multioutput, str) or multioutput is None:
        if multioutput == "raw_values":
            if compute:
                return output_errors.compute()
            else:
                return output_errors
    else:
        raise ValueError("Weighted 'multioutput' not supported.")
    result = output_errors.mean()
    if compute:
        result = result.compute()
    return result


@derived_from(sklearn.metrics)
def r2_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Optional[str] = "uniform_average",
    compute: bool = True,
) -> ArrayLike:
    _check_sample_weight(sample_weight)
    _, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, multioutput)
    weight = 1.0

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype="f8")
    denominator = (weight * (y_true - y_true.mean(axis=0)) ** 2).sum(axis=0, dtype="f8")

    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_chunks = getattr(y_true, "chunks", [None, None])[1]
    output_scores = da.ones([y_true.shape[1]], chunks=output_chunks)
    with np.errstate(all="ignore"):
        output_scores[valid_score] = 1 - (
            numerator[valid_score] / denominator[valid_score]
        )
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    result = output_scores.mean(axis=0)
    if compute:
        result = result.compute()
    return result


@derived_from(sklearn.metrics)
def mean_squared_log_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    multioutput: Optional[str] = "uniform_average",
    compute: bool = True,
) -> ArrayLike:
    result = mean_squared_error(
        np.log1p(y_true),
        np.log1p(y_pred),
        sample_weight=sample_weight,
        multioutput=multioutput,
        compute=compute,
    )
    return result
