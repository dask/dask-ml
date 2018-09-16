import numpy as np
import math
import toolz
from time import time


def _get_hist(info):
    hist = [v[-1] for v in info.values()]
    for h in hist:
        h["wall_time"] = time()
    return hist


def stop_on_plateau(info, patience=10, tol=0.001, max_iter=None):
    """
    Stop training when a plateau is reaching with validation score.

    A plateau is defined to be when the validation scores are all below the
    plateau's start score, plus the tolerance. The plateau is ``patience``
    ``partial_fit`` calls wide. That is, a plateau if reached if for every
    score,

        score < plateau_start_score + tol

    This function is designed for use with
    :func:`~dask_ml.model_selection.fit`.

    Parameters
    ----------
    patience : int
        Number of partial_fit_calls that specifies the plateau's width
    tol : float
        The plateau starts at a certain score. How far above that score is still
        considered a plateau?
    max_iter : int
        How many times to call ``partial_fit`` on each model.

    Returns
    -------
    partial_fit_calls : dict
        Each key specifies wether to continue training this model.
    """
    out = {}
    for ident, records in info.items():
        pf_calls = records[-1]["partial_fit_calls"]
        if max_iter is not None and pf_calls >= max_iter:
            out[ident] = 0

        elif pf_calls >= patience:
            plateau = {
                d["partial_fit_calls"]: d["score"]
                for d in records
                if pf_calls - patience <= d["partial_fit_calls"]
            }
            if len(plateau) == 1:
                out[ident] = 1
            else:
                plateau_start = plateau[min(plateau)]
                if all(score < plateau_start + tol for score in plateau.values()):
                    out[ident] = 0
                else:
                    out[ident] = 1
        else:
            out[ident] = 1
    return out


class _HistoryRecorder:

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.history = []
        self.best_scores = {}

    def fit(self, info):
        self.best_scores.update({id_: hist[-1]["score"] for id_, hist in info.items()})
        self.best_score = max(self.best_scores.values())
        self.history += _get_hist(info)
        return self.fn(info, *self.args, **self.kwargs)


class _SHA:

    def __init__(
        self, n, r, eta=3, limit=None, patience=10, tol=0.001, bracket=0, verbose=0
    ):
        """
        Perform the successive halving algorithm.

        Parameters
        ----------
        n : int
            Number of models to evaluate initially
        r : int
            Number of times to call partial fit initially
        eta : float, default=3
            How aggressive to be in culling off the models. Higher
            values correspond to being more aggressive in killing off
            models. The "infinite horizon" theory suggests eta=np.e=2.718...
            is optimal.
        patience : int
            Passed to `stop_on_plateau`
        tol : float
            Passed to `stop_on_plateau`
        verbose : int
            Controls the verbosity of this object. Higher number means reporting
            values more often.
        """
        self.steps = 0
        self.n = n
        self.r = r
        self.eta = eta
        self.meta = []
        self.limit = limit
        self._best_scores = {}
        self._history = []
        self.patience = patience
        self.tol = tol
        self._to_reach = {}
        self._addtl = None
        self._best_score = -np.inf
        self._start_time = time()
        self.bracket = bracket
        self.verbose = verbose

    def fit(self, info):
        """
        This function provides a wrapper around `stop_on_plateau` for `self._fit`.

        It calls `self._fit` to see how many additional partial fit calls are needed,
        then scores the model more frequently for evaluating plateaus.

        Notes on variables
        ------------------
        self._best_scores : dict[int, float]
            Best scores for all models. Used so models are not deleted (the best
            model may plateau early)
        self.patience : int
            The width of the plateau
        self._addtl : dict[int, int]
            The number of additional partial fit calls needed
        self.steps : int
            The number of times self._fit has been called (but it's set in self._fit)

        Returns
        -------
        calls : dict
            Number of calls necessary. Will stop on plateau specified by self.patience.
            Will be scored on additional calls specified by self._fit or every half
            plateau.

        """
        # If there's no patience, don't even bother wrapping stop on plateau
        # and give the function.
        if np.isinf(self.patience):
            return self._fit(info)

        # Score the model every patience/2 to have at least 3 points for every
        # plateau
        patience = self.patience // 2

        # Case: we need to call _fit again to see how long to train
        if self._addtl is None or sum(self._addtl.values()) == 0:
            self._addtl = self._fit(info)

        # End case: we shouldn't call _fit anymore
        if self.steps > self.limit:
            return {k: 0 for k in self._best_scores}

        # Stop on plateau integration
        keep_training = stop_on_plateau(info, patience=self.patience, tol=self.tol)
        assert set(keep_training.values()).issubset({0, 1})
        if sum(keep_training.values()) == 0:
            return {k: 0 for k in self._best_scores}
        info = {k: info[k] for k, v in keep_training.items() if v}
        models_to_keep = set(info).intersection(set(self._addtl))
        self._addtl = {k: self._addtl[k] for k in models_to_keep}

        # Case: we haven't reached the partial fit calls required, so need to
        # train further
        ret = {k: min(patience, v) for k, v in self._addtl.items()}
        self._addtl = {k: v - ret[k] for k, v in self._addtl.items()}

        dont_train = {k: 0 for k in self._best_scores if k not in ret}
        ret.update(dont_train)
        return ret

    def _fit(self, info):
        self._history += _get_hist(info)
        for ident, hist in info.items():
            self._best_scores[ident] = hist[-1]["score"]

        n, r, eta = self.n, self.r, self.eta
        n_i = int(math.floor(n * eta ** -self.steps))
        r_i = np.round(r * eta ** self.steps).astype(int)

        if self.verbose > 0:
            msg = (
                "[CV] Found score={score} at {time:0.3f}s for Hyperband bracket={bracket}"
            )
            _time = time() - self._start_time
            _bracket = self.bracket
            _score = min(self._best_scores.values())
            print(msg.format(bracket=_bracket, score=_score, time=_time))

        # Initial case
        # partial fit has already been called once
        if r_i == 1:
            # if r_i == 1, a step has already been completed for us
            assert self.steps == 0
            self.steps = 1
            pf_calls = {k: info[k][-1]["partial_fit_calls"] for k in info}
            return self._fit(info)
        # this ordering is important; typically r_i==1 only when steps==0
        if self.steps == 0:
            # we have r_i - 1 more steps to train to
            self.steps = 1
            self._to_reach = {k: r_i for k in info}
            return {k: r_i - 1 for k in info}

        best = toolz.topk(n_i, info, key=lambda k: info[k][-1]["score"])
        self.steps += 1

        if self.steps > self.limit or (self.limit is None and len(best) in {0, 1}):
            max_score = max(self._best_scores.values())
            best_ids = {k for k, v in self._best_scores.items() if v == max_score}
            return {best_id: 0 for best_id in best_ids}

        pf_calls = {k: info[k][-1]["partial_fit_calls"] for k in best}
        addtl_pf_calls = {k: r_i - pf_calls[k] for k in best}
        self._to_reach.update({k: r_i for k in addtl_pf_calls})
        dont_train = {k: 0 for k in self._best_scores if k not in addtl_pf_calls}
        assert set(addtl_pf_calls).intersection(dont_train) == set()
        addtl_pf_calls.update(dont_train)
        return addtl_pf_calls
