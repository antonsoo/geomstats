import pytest

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.kmedoids import RiemannianKMedoids


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_medoid_per_distinct_sample(seed):
    """Requesting one cluster per sample retains every distinct observation."""
    points = gs.reshape(gs.arange(10, dtype=gs.float64), (-1, 1))
    gs.random.seed(seed)
    estimator = RiemannianKMedoids(Euclidean(1), n_clusters=len(points)).fit(points)

    assert gs.all(gs.sort(estimator.medoid_indices_) == gs.arange(len(points)))
    assert len(gs.unique(estimator.predict(points))) == len(points)


def test_more_medoids_than_samples():
    """A distinct medoid cannot be chosen for more clusters than samples."""
    estimator = RiemannianKMedoids(Euclidean(1), n_clusters=3)
    with pytest.raises(ValueError):
        estimator.fit(gs.array([[0.0], [1.0]]))
