import six

import packaging.version
import dask.array as da
from dask.array.random import doc_wraps
import sklearn.metrics

from .._compat import DASK_VERSION


def _check_sample_weight(sample_weight):
    if sample_weight is not None:
        no_average = DASK_VERSION <= packaging.version.parse("0.18.0")
        if no_average:
            raise NotImplementedError("'sample_weight' is only supported for "
                                      "dask versions > 0.18.0.")


def _check_reg_targets(y_true, y_pred, multioutput):
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, six.string_types):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}"
                             "".format(allowed_multioutput_str, multioutput))
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    # TODO: y_type, multioutput
    return None, y_true, y_pred, multioutput


@doc_wraps(sklearn.metrics.mean_squared_error)
def mean_squared_error(y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average',
                       compute=True):
    _check_sample_weight(sample_weight)
    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    output_errors = da.average((y_pred - y_true) ** 2, axis=0,
                               weights=sample_weight)

    if isinstance(multioutput, six.string_types):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to da.average: uniform mean
            multioutput = None
    result = da.average(output_errors, weights=multioutput)

    if compute:
        result = result.compute()
    return result


@doc_wraps(sklearn.metrics.mean_squared_error)
def mean_absolute_error(y_true, y_pred,
                        sample_weight=None,
                        multioutput='uniform_average',
                        compute=True):
    _check_sample_weight(sample_weight)
    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    output_errors = da.average(abs(y_pred - y_true), axis=0,
                               weights=sample_weight)

    if isinstance(multioutput, six.string_types):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to da.average: uniform mean
            multioutput = None
    result = da.average(output_errors, weights=multioutput)
    if compute:
        result = result.compute()
    return result


@doc_wraps(sklearn.metrics.r2_score)
def r2_score(y_true, y_pred, sample_weight=None,
             multioutput="uniform_average",
             compute=True):
    _check_sample_weight(sample_weight)
    _, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    if sample_weight is not None:
        weight = sample_weight
        if weight.ndim == 1:
            weight = weight.reshape((-1, 1))
    else:
        weight = 1.0

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0, dtype='f8')
    denominator = (weight * (y_true - da.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0, dtype='f8')

    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = da.ones([y_true.shape[1]], chunks=y_true.chunks[1])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.

    if isinstance(multioutput, six.string_types):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores.compute() if compute else output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
    else:
        avg_weights = multioutput

    result = da.average(output_scores, weights=avg_weights)
    if compute:
        result = result.compute()
    return result
