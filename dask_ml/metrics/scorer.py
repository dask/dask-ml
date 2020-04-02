from typing import Any, Callable, Tuple, Union

from sklearn.metrics import check_scoring as sklearn_check_scoring, make_scorer

from . import accuracy_score, log_loss, mean_squared_error, r2_score

# Scorers
accuracy_scorer: Tuple[Any, Any] = (accuracy_score, {})
neg_mean_squared_error_scorer = (mean_squared_error, dict(greater_is_better=False))
r2_scorer: Tuple[Any, Any] = (r2_score, {})
neg_log_loss_scorer = (log_loss, dict(greater_is_better=False, needs_proba=True))


SCORERS = dict(
    accuracy=accuracy_scorer,
    neg_mean_squared_error=neg_mean_squared_error_scorer,
    r2=r2_scorer,
    neg_log_loss=neg_log_loss_scorer,
)


def get_scorer(scoring: Union[str, Callable], compute: bool = True) -> Callable:
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
    if isinstance(scoring, str):
        try:
            scorer, kwargs = SCORERS[scoring]
        except KeyError:
            raise ValueError(
                "{} is not a valid scoring value. "
                "Valid options are {}".format(scoring, sorted(SCORERS))
            )
    else:
        scorer = scoring
        kwargs = {}

    kwargs["compute"] = compute

    return make_scorer(scorer, **kwargs)


def check_scoring(estimator, scoring=None, **kwargs):
    res = sklearn_check_scoring(estimator, scoring=scoring, **kwargs)
    if callable(scoring):
        # Heuristic to ensure user has not passed a metric
        module = getattr(scoring, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("dask_ml.metrics.")
            and not module.startswith("dask_ml.metrics.scorer")
            and not module.startswith("dask_ml.metrics.tests.")
        ):
            raise ValueError(
                "scoring value %r looks like it is a metric "
                "function rather than a scorer. A scorer should "
                "require an estimator as its first parameter. "
                "Please use `make_scorer` to convert a metric "
                "to a scorer." % scoring
            )
    if scoring in SCORERS.keys():
        func, kwargs = SCORERS[scoring]
        return make_scorer(func, **kwargs)
    return res
