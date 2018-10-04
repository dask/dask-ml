import dask.array as da
import numpy as np
import six
import sklearn.metrics
from dask.array.random import doc_wraps


def _check_sample_weight(sample_weight):
    if sample_weight is not None:
        raise ValueError("'sample_weight' is not supported.")


def _check_reg_targets(y_true, y_pred, multioutput):
    if multioutput != "uniform_average":
        raise NotImplementedError("'multioutput' must be 'uniform_average'")

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    # TODO: y_type, multioutput
    return None, y_true, y_pred, multioutput


@doc_wraps(sklearn.metrics.mean_squared_error)
def mean_squared_error(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average", compute=True
):
    _check_sample_weight(sample_weight)
    output_errors = ((y_pred - y_true) ** 2).mean(axis=0)

    if isinstance(multioutput, six.string_types):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
    else:
        raise ValueError("Weighted 'multioutput' not supported.")
    result = output_errors.mean()
    if compute:
        result = result.compute()
    return result


@doc_wraps(sklearn.metrics.mean_squared_error)
def mean_absolute_error(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average", compute=True
):
    _check_sample_weight(sample_weight)
    output_errors = abs(y_pred - y_true).mean(axis=0)

    if isinstance(multioutput, six.string_types):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
    else:
        raise ValueError("Weighted 'multioutput' not supported.")
    result = output_errors.mean()
    if compute:
        result = result.compute()
    return result


@doc_wraps(sklearn.metrics.r2_score)
def r2_score(
    y_true, y_pred, sample_weight=None, multioutput="uniform_average", compute=True
):
    _check_sample_weight(sample_weight)
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    weight = 1.0

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype="f8")
    denominator = (weight * (y_true - y_true.mean(axis=0)) ** 2).sum(axis=0, dtype="f8")

    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = da.ones([y_true.shape[1]], chunks=y_true.chunks[1])
    with np.errstate(all="ignore"):
        output_scores[valid_score] = 1 - (
            numerator[valid_score] / denominator[valid_score]
        )
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    result = output_scores.mean(axis=0)
    if compute:
        result = result.compute()
    return result
