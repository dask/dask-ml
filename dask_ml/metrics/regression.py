import six

from dask.array.random import doc_wraps
import sklearn.metrics


def _check_sample_weight(sample_weight):
    if sample_weight is not None:
        raise ValueError("'sample_weight' is not supported.")


@doc_wraps(sklearn.metrics.mean_squared_error)
def mean_squared_error(y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average'):
    _check_sample_weight(sample_weight)
    output_errors = ((y_pred - y_true) ** 2).mean(axis=0)

    if isinstance(multioutput, six.string_types):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None
    else:
        raise ValueError("Weighted 'multioutput' not supported.")
    return output_errors.mean()


@doc_wraps(sklearn.metrics.mean_squared_error)
def mean_absolute_error(y_true, y_pred,
                        sample_weight=None,
                        multioutput='uniform_average'):
    _check_sample_weight(sample_weight)
    output_errors = abs(y_pred - y_true).mean(axis=0)

    if isinstance(multioutput, six.string_types):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None
    else:
        raise ValueError("Weighted 'multioutput' not supported.")
    return output_errors.mean()
