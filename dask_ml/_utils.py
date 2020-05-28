import time

import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator


def copy_learned_attributes(from_estimator, to_estimator):
    attrs = {k: v for k, v in vars(from_estimator).items() if k.endswith("_")}

    for k, v in attrs.items():
        setattr(to_estimator, k, v)


def draw_seed(random_state, low, high=None, size=None, dtype=None, chunks=None):
    kwargs = {"size": size}
    if chunks is not None:
        kwargs["chunks"] = chunks

    seed = random_state.randint(low, high, **kwargs)
    if dtype is not None and isinstance(seed, (da.Array, np.ndarray)):
        seed = seed.astype(dtype)

    return seed


class ConstantFunction(BaseEstimator):
    def __init__(self, value=0, sleep=0, **kwargs):
        self.value = value
        self._partial_fit_called = False
        self.sleep = sleep
        super(ConstantFunction, self).__init__(**kwargs)

    def _fn(self):
        return self.value

    def partial_fit(self, X, y=None, **kwargs):
        time.sleep(self.sleep)
        self._partial_fit_called = True

        # Mirroring sklearn's SGDClassifier epoch counting
        if not hasattr(self, "t_"):
            self.t_ = 1
        self.t_ += X.shape[0]
        self.coef_ = X[0]
        return self

    def score(self, *args, **kwargs):
        return self._fn()

    def fit(self, *args):
        return self


class LoggingContext:
    """
    Optionally change the logging level and add a logging handler purely
    in the scope of the context manager:

    If you specify a level value, the logger’s level is set to that value
    in the scope of the with block covered by the context manager. If you
    specify a handler, it is added to the logger on entry to the block
    and removed on exit from the block. You can also ask the manager to
    close the handler for you on block exit - you could do this if you
    don’t need the handler any more.

    Stolen from [1]

    [1]:https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
    """

    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)

        # The reasonsing behind the last part of the below if statement:
        # What if this context is called multiple times with the same logger?
        # Then, only add loggers if they have different output streams
        if self.handler and not any(
            h.stream == self.handler.stream for h in self.logger.handlers
        ):
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions
