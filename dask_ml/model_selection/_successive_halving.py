import copy
import numpy as np
import math
import toolz
from time import time
from collections import defaultdict
from ._incremental import AdaptiveSearchCV


class SuccessiveHalving(AdaptiveSearchCV):
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
        """
        Perform the successive halving algorithm.

        Parameters
        ----------
        n_initial_parameters : int
            Number of models to evaluate initially
        resource : int
            Number of times to call partial fit initially
        aggressizeness : float, default=3
            How aggressive to be in culling off the models. Higher
            values correspond to being more aggressive in killing off
            models. The "infinite horizon" theory suggests aggressizeness=np.e=2.718...
            is optimal.
        """
        self._steps = 0  # TODO: set this in self.fit
        self._meta = defaultdict(list)
        self.n_initial_parameters = n_initial_parameters
        self.resource = resource
        self.aggressiveness = aggressiveness
        self.limit = limit
        self.estimator = estimator
        super(SuccessiveHalving, self).__init__(
            estimator,
            param_distribution,
            n_initial_parameters=n_initial_parameters,
            **kwargs,
        )

    def _adapt(self, info):
        n, r, eta = self.n_initial_parameters, self.resource, self.aggressiveness
        n_i = int(math.floor(n * eta ** -self._steps))
        r_i = np.round(r * eta ** self._steps).astype(int)

        tick = time()
        for k in info.keys():
            self._meta[k] += [tick]

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
