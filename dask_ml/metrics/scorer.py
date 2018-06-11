import six
from sklearn.metrics import make_scorer
from sklearn.metrics.scorer import check_scoring as sklearn_check_scoring

from . import (
    accuracy_score,
    mean_squared_error,
    r2_score,
)

# Scorers
accuracy_scorer = make_scorer(accuracy_score)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error,
                                            greater_is_better=False)
r2_scorer = make_scorer(r2_score)


SCORERS = dict(
    accuracy=accuracy_scorer,
    neg_mean_squared_error=neg_mean_squared_error_scorer,
    r2=r2_scorer,
)


def get_scorer(scoring, compute=True):
    """Get a scorer from string

    Parameters
    ----------
    scoring : str | callable
        scoring method as string. If callable it is returned as is.

    Returns
    -------
    scorer : callable
        The scorer.
    """
    # This is the same as sklearns, only we use our SCORERS dict,
    # and don't have back-compat code
    if isinstance(scoring, six.string_types):
        try:
            scorer = SCORERS[scoring]
        except KeyError:
            raise ValueError('{} is not a valid scoring value. '
                             'Valid options are {}'.format(scoring,
                                                           sorted(SCORERS)))
    else:
        scorer = scoring

    return scorer


def check_scoring(estimator, scoring=None, **kwargs):
    sklearn_check_scoring(estimator, scoring=scoring, **kwargs)
    if callable(scoring):
        # Heuristic to ensure user has not passed a metric
        module = getattr(scoring, '__module__', None)
        if hasattr(module, 'startswith') and \
           module.startswith('dask_ml.metrics.') and \
           not module.startswith('dask_ml.metrics.scorer') and \
           not module.startswith('dask_ml.metrics.tests.'):
            raise ValueError('scoring value %r looks like it is a metric '
                             'function rather than a scorer. A scorer should '
                             'require an estimator as its first parameter. '
                             'Please use `make_scorer` to convert a metric '
                             'to a scorer.' % scoring)
