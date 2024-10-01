import numpy as np
from sklearn.neighbors import KernelDensity


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group
    using kernel density estimation.

    Parameters
    ----------
    embeddings: np.ndarray :
        An array of item embeddings.

    Returns
    -------
    np.ndarray
        Uniqueness estimates for each item in the embeddings group.

    """
    kde = KernelDensity().fit(embeddings)

    estimates = []

    for item in embeddings:
        # KDE score is a log of probability density function
        # so we take an exponential to get the probability
        # and then take a reciprocal to get the uniqueness
        estimates.append(1/np.exp(kde.score_samples([item])[0]))

    return np.array(estimates)
