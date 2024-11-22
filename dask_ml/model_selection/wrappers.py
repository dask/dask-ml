"""Dask-ML model_selection-compatible wrappers - EARLY POC"""

import copy
import numpy as np
from .utils import _safe_indexing

try:
    import xgboost as xgb
    has_xgboost = True
except ImportError:
    xgb = None
    has_xgboost = False

class XGBoostWrapper:
    """Lightweight, sklearn-like wrapper for XGBoost training
    that takes DMatrix as input.
    This is a VERY basic Poc."""
    def __init__(self, num_boost_round, score_function, **xgb_params):
        if not has_xgboost:
            raise ImportError("XGBoost is not installed")
        self.xgb_params = xgb_params
        self.num_boost_round = num_boost_round
        self.booster_ = None
        self.score_function = score_function

    def fit(self, X_dmat, y=None):
        self.booster_ = xgb.train(self.xgb_params,
                                  X_dmat,
                                  num_boost_round=self.num_boost_round)
        return self

    def predict(self, data, output_margin=False):
        if isinstance(data, xgb.DMatrix):
            test_dmatrix = data
        else:
            # XXX: base_margin, missing unsupported
            test_dmatrix = xgb.DMatrix(data)
        class_probs = self.booster_.predict(test_dmatrix,
                                            output_margin=output_margin)
        if output_margin:
            # If output_margin is active, simply return the scores
            return class_probs

        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > 0.5] = 1

        # Note: no label encoding, unlike sklearn version
        return column_indexes

    def score(self, X, y=None):
        y_pred = self.predict(X)
        y_label = X.get_label()
        if y_label is None:
            y_label = y # XXX not sure if this is right
        return self.score_function(y_label, y_pred)

    def get_params(self, deep=False):
        if deep:
            params = copy.deepcopy(self.xgb_params)
        else:
            params = copy.copy(self.xgb_params)
        params["num_boost_round"] = self.num_boost_round
        return params

    def set_params(self, **params):
        params_in = copy.copy(params)
        if "num_boost_round" in params_in:
            self.num_boost_round = params_in["num_boost_round"]
            del params_in["num_boost_round"]
        if "xgb_params" in params_in:
            self.xgb_params.update(copy.copy(params_in["xgb_params"]))
            del params_in["xgb_params"]
        self.xgb_params.update(params_in)
        return self


def extract_dmatrix(cv, X, y, n, is_x=True, is_train=True):
    """Custom dask-ml extract function, returning DMatrix instead of numpy"""
    if not has_xgboost:
        raise ImportError("XGBoost is not installed")

    if not is_x:
        return None

    # XXX maybe the interface should just pass in splits instead of cv?
    inds = cv.splits[n][0] if is_train else cv.splits[n][1]
    x_part = _safe_indexing(X, inds)
    y_part = _safe_indexing(y, inds)

    # TODO: in practice, there may be additional params like weights
    result = xgb.DMatrix(x_part, y_part)

    return result
