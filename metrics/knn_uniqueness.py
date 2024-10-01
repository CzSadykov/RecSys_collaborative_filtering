import numpy as np
from scipy.spatial.distance import euclidean as distance


def knn_uniqueness(
        embeddings: np.ndarray, num_neighbors: int = 5
        ) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group
    as a mean euclidean distance with its K nearest neighbors.

    Parameters
    ----------
    embeddings: np.ndarray :
        An array of item embeddings.
    num_neighbors: int :
        Chosen number of neighbors to estimate uniqueness of items.

    Returns
    -------
    np.ndarray
        Uniqueness estimates for each item in the embeddings group.

    """
    estimates = []

    for item in embeddings:
        distances = []
        for neighbor in embeddings:
            distances.append(distance(item, neighbor))

        distances = sorted(distances)[1:]

        neighbors_count = min(num_neighbors, len(embeddings) - 1)
        nearest_neighbor_distances = distances[:neighbors_count]

        uniqueness = (sum(nearest_neighbor_distances) / neighbors_count
                      if neighbors_count > 0 else 0)
        estimates.append(uniqueness)

    return np.array(estimates)
