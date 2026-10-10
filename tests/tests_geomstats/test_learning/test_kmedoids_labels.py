import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.kmedoids import RiemannianKMedoids


@pytest.mark.parametrize("max_iter", [0, 1, 2, 100])
def test_labels_match_final_medoids(monkeypatch, max_iter):
    """Labels use the returned medoids, including an iteration-limit exit."""
    points = gs.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
    estimator = RiemannianKMedoids(Euclidean(1), n_clusters=2, max_iter=max_iter)
    monkeypatch.setattr(
        estimator, "_initialize_medoids", lambda distances: gs.array([0, 2])
    )

    estimator.fit(points)

    expected = gs.argmin(
        gs.abs(points - gs.reshape(estimator.cluster_centers_, (1, -1))), axis=1
    )
    assert gs.all(estimator.labels_ == expected)
    assert gs.all(estimator.labels_ == estimator.predict(points))
