import itertools
import math

import numpy as np
import toolz

from ._incremental import IncrementalSearchCV


class SuccessiveHalvingSearchCV(IncrementalSearchCV):
    """
    Perform the successive halving algorithm [1]_.

    This algorithm trains estimators for a certain number ``partial_fit``
    calls to ``partial_fit``, then kills the worst performing half.
    It trains the surviving estimators for twice as long, and repeats this
    until one estimator survives.

    The value of :math:`1/2` above is used for a clear explanation. This class
    defaults to killing the worst performing ``1 - 1 // aggressiveness``
    fraction of models, and trains estimators for ``aggressiveness`` times
    longer, and waits until the number of models left is less than
    ``aggressiveness``.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each initial hyperparameter
        combination. This is assumed to implement the scikit-learn estimator
        interface. Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed. The estimator must implement
        ``partial_fit``, ``set_params``, and work well with ``clone``.

    parameters : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    aggressiveness : float, default=3
        How aggressive to be in culling off the different estimators. Higher
        values imply higher confidence in scoring (or that
        the hyperparameters influence the ``estimator.score`` more
        than the data).

    n_initial_parameters : int, default=10
        Number of parameter settings that are sampled.
        This trades off runtime vs quality of the solution.

    max_iter : int, default 100
        Maximum number of partial fit calls per model.

    test_size : float
        Fraction of the dataset to hold out for computing test scores.
        Defaults to the size of a single partition of the input training set

        .. note::

           The training dataset should fit in memory on a single machine.
           Adjust the ``test_size`` parameter as necessary to achieve this.

    patience : int, default False
        If specified, training stops when the score does not increase by
        ``tol`` after ``patience`` calls to ``partial_fit``. Off by default.

    tol : float, default 0.001
        The required level of improvement to consider stopping training on
        that model. The most recent score must be at at most ``tol`` better
        than the all of the previous ``patience`` scores for that model.
        Increasing ``tol`` will tend to reduce training time, at the cost
        of worse models.

    scoring : string, callable, None. default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        If None, the estimator's default scorer (if available) is used.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    Attributes
    ----------
    cv_results_ : dict of np.ndarrays
        This dictionary has keys

        * ``mean_partial_fit_time``
        * ``mean_score_time``
        * ``std_partial_fit_time``
        * ``std_score_time``
        * ``test_score``
        * ``rank_test_score``
        * ``model_id``
        * ``partial_fit_calls``
        * ``params``
        * ``param_{key}``, where ``key`` is every key in ``params``.

        The values in the ``test_score`` key correspond to the last score a model
        received on the hold out dataset. The key ``model_id`` corresponds with
        ``history_``. This dictionary can be imported into Pandas.

    model_history_ : dict of lists of dict
        A dictionary of each models history. This is a reorganization of
        ``history_``: the same information is present but organized per model.

        This data has the structure  ``{model_id: hist}`` where ``hist`` is a
        subset of ``history_`` and ``model_id`` are model identifiers.

    history_ : list of dicts
        Information about each model after each ``partial_fit`` call. Each dict
        the keys

        * ``partial_fit_time``
        * ``score_time``
        * ``score``
        * ``model_id``
        * ``params``
        * ``partial_fit_calls``

        The key ``model_id`` corresponds to the ``model_id`` in ``cv_results_``.
        This list of dicts can be imported into Pandas.

    best_estimator_ : BaseEstimator
        The model with the highest validation score among all the models
        retained by the "inverse decay" algorithm.

    best_score_ : float
        Score achieved by ``best_estimator_`` on the vaidation set after the
        final call to ``partial_fit``.

    best_index_ : int
        Index indicating which estimator in ``cv_results_`` corresponds to
        the highest score.

    best_params_ : dict
        Dictionary of best parameters found on the hold-out data.

    scorer_ :
        The function used to score models, which has a call signature of
        ``scorer_(estimator, X, y)``.

    n_splits_ : int
        Number of cross validation splits.

    multimetric_ : bool
        Whether this cross validation search uses multiple metrics.


    References
    ----------
    .. [1] "Non-stochastic best arm identification and hyperparameter
           optimization" by Jamieson, Kevin and Talwalkar, Ameet. 2016.
           https://arxiv.org/abs/1502.07943

    """

    def __init__(
        self,
        estimator,
        parameters,
        n_initial_parameters=10,
        max_iter=100,
        aggressiveness=3,
        test_size=None,
        patience=False,
        tol=1e-3,
        random_state=None,
        scoring=None,
    ):
        self.n_initial_parameters = n_initial_parameters
        self.aggressiveness = aggressiveness

        super(SuccessiveHalvingSearchCV, self).__init__(
            estimator,
            parameters,
            n_initial_parameters=n_initial_parameters,
            max_iter=max_iter,
            test_size=test_size,
            patience=patience,
            tol=tol,
            random_state=random_state,
            scoring=scoring,
        )

    def _adapt(self, info, first_step_completed=False):
        if all(v[-1]["partial_fit_calls"] == 1 for v in info.values()):
            # Do all the models have one partial fit call?
            self._steps = 0
        if first_step_completed:
            # Sometimes, IncrementalSearchCV completes one step for us. We
            # recurse in this case -- see below for a note on the condition
            self._steps = 1
        n, eta = self.n_initial_parameters, self.aggressiveness
        if not hasattr(self, "_n_initial_calls"):
            self._n_initial_calls = _get_n_initial_calls(n, self.max_iter, eta)
        r = self._n_initial_calls

        n_i = int(math.floor(n * eta ** -self._steps))
        r_i = np.round(r * eta ** self._steps).astype(int)
        if r_i == 1:
            # if r_i == 1, a step has already been completed for us (because
            # IncrementalSearchCV completes 1 partial_fit call automatically)
            return self._adapt(info, first_step_completed=True)

        best = toolz.topk(n_i, info, key=lambda k: info[k][-1]["score"])
        self._steps += 1

        if len(best) == 0:
            return {id_: 0 for id_ in info}

        pf_calls = {k: info[k][-1]["partial_fit_calls"] for k in best}
        additional_calls = {k: r_i - pf_calls[k] for k in best}
        return additional_calls


def _get_max_iter(n, r, eta):
    """
    Parameters
    ----------
    n : int
        Number of intial models
    r : int
        Number of initial calls
    eta : int
        aggressiveness of the search

    Notes
    -----
    n, r and eta come from Hyperband
    """
    for k in itertools.count():
        n_i = int(math.floor(n * eta ** -k))
        r_i = np.round(r * eta ** k).astype(int)
        if n_i <= 1:
            break
    return r_i


def _get_n_initial_calls(n_initial_parameters, max_iter, eta):
    for n_initial_calls in range(n_initial_parameters, 0, -1):
        calls = _get_max_iter(n_initial_parameters, n_initial_calls, eta)
        if calls <= max_iter:
            break
    return n_initial_calls
