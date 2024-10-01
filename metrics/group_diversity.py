import numpy as np
from typing import Tuple, Optional

from metrics.knn_uniqueness import knn_uniqueness
from metrics.kde_uniqueness import kde_uniqueness


def group_diversity(
        embeddings: np.ndarray,
        threshold: float,
        diversity_metric: Optional[str] = 'kde',
        num_neighbors: Optional[int] = 5
        ) -> Tuple[bool, float]:
    """Calculate group diversity based on uniqueness metric:
    KDE or KNN with mean euclidean distance (5 neighbors by default).

    Parameters
    ----------
    embeddings: np.ndarray :
        An array of item embeddings.
    threshold: float :
       A threshold for group diversity.
    diversity_metric: Optional[str] :
       A metric for group diversity: 'kde' or 'knn'.
    num_neighbors: Optional[int] :
       Number of neighbors for KNN uniqueness metric.

    Returns
    -------
    Tuple[bool, float]
        reject: bool
            Whether the group should be rejected.
        diversity: float
            Group diversity calculated as a sum of uniqueness estimates
            divided by number of items in the group.

    """
    if len(embeddings) == 0:
        return True, 0.0

    if diversity_metric == 'kde':
        uniqueness = kde_uniqueness(embeddings)
    elif diversity_metric == 'knn':
        uniqueness = knn_uniqueness(embeddings, num_neighbors)
    else:
        raise ValueError(f"Unknown diversity metric: {diversity_metric}")

    diversity = np.sum(uniqueness) / len(embeddings)
    reject = diversity < threshold
    return reject, diversity
