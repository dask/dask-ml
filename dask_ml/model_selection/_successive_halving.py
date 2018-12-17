import copy
import math
from collections import defaultdict

import numpy as np
import toolz

from ._incremental import INC_ATTRS, IncrementalSearchCV

DOC = (
    """
    Perform the successive halving algorithm.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each initial hyperparameter
        combination. This is assumed to implement the scikit-learn estimator
        interface. Either estimator needs to provide a `score`` function,
        or ``scoring`` must be passed. The estimator must implement
        ``partial_fit``, ``set_params``, and work well with ``clone``.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_initial_parameters : int, default=10
        Number of parameter settings that are sampled.
        This trades off runtime vs quality of the solution.

        Alternatively, you can set this to ``"grid"`` to do a full grid search.

    resource : int
        Number of times to call partial fit initially. Estimators are trained
        for ``resource`` calls to ``partial_fit`` at first, then additional
        times increasing by ``aggressivenes``.

    aggressizeness : float, default=3
        How aggressive to be in culling off the models. Higher
        values correspond to being more aggressive in killing off
        models. The "infinite horizon" theory suggests ``aggressizeness == np.e``
        is optimal.

    kwargs : dict
        Parameters to pass to
        :class:`~dask_ml.model_selection.IncrementalSearchCV`.

    """
    + INC_ATTRS
)


class SuccessiveHalvingSearchCV(IncrementalSearchCV):
    __doc__ = DOC

    def __init__(
        self,
        estimator,
        param_distribution,
        n_initial_parameters,
        resource,
        aggressiveness=3,
        limit=None,
        **kwargs,
    ):
        self._steps = 0
        self._pf_calls = {}
        self.n_initial_parameters = n_initial_parameters
        self.resource = resource
        self.aggressiveness = aggressiveness
        self.limit = limit
        self.estimator = estimator
        super(SuccessiveHalvingSearchCV, self).__init__(
            estimator,
            param_distribution,
            n_initial_parameters=n_initial_parameters,
            **kwargs,
        )

    def _adapt(self, info):
        n, r, eta = self.n_initial_parameters, self.resource, self.aggressiveness
        n_i = int(math.floor(n * eta ** -self._steps))
        r_i = np.round(r * eta ** self._steps).astype(int)
        self._pf_calls.update({k: v[-1]["partial_fit_calls"] for k, v in info.items()})

        self.metadata_ = {
            "models": len(self._pf_calls),
            "partial_fit_calls": sum(self._pf_calls.values()),
        }

        # Initial case
        # partial fit has already been called once
        if r_i == 1:
            # if r_i == 1, a step has already been completed for us
            assert self._steps == 0
            self._steps = 1
            pf_calls = {k: info[k][-1]["partial_fit_calls"] for k in info}
            return self._adapt(info)

        # this ordering is important; typically r_i==1 only when steps==0
        if self._steps == 0:
            # we have r_i - 1 more steps to train to
            self._steps = 1
            return {k: r_i - 1 for k in info}

        best = toolz.topk(n_i, info, key=lambda k: info[k][-1]["score"])
        self._steps += 1

        if (self.limit is None and len(best) in {0, 1}) or (
            self.limit is not None and self._steps > self.limit
        ):
            return {id_: 0 for id_ in info}

        pf_calls = {k: info[k][-1]["partial_fit_calls"] for k in best}
        addtl_calls = {k: r_i - pf_calls[k] for k in best}
        return addtl_calls
