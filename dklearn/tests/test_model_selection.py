from __future__ import absolute_import, division, print_function

import os

import pytest
from dask.utils import tmpdir
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.svm import LinearSVC

from dklearn import DaskGridSearchCV


def test_visualize():
    pytest.importorskip('graphviz')

    X, y = make_classification(n_samples=100, n_classes=2, flip_y=.2,
                               random_state=0)
    clf = LinearSVC(random_state=0)
    grid = {'C': [.1, .5, .9]}
    gs = DaskGridSearchCV(clf, grid).fit(X, y)

    assert hasattr(gs, 'dask_graph_')
    assert hasattr(gs, 'dask_keys_')

    with tmpdir() as d:
        gs.visualize(filename=os.path.join(d, 'mydask'))
        assert os.path.exists(os.path.join(d, 'mydask.png'))

    # Doesn't work if not fitted
    gs = DaskGridSearchCV(clf, grid)
    with pytest.raises(NotFittedError):
        gs.visualize()
