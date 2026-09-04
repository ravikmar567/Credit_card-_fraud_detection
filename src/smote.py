"""
A minimal, from-scratch implementation of SMOTE (Synthetic Minority
Over-sampling Technique, Chawla et al., 2002).

Written by hand (using sklearn's NearestNeighbors) instead of pulling in the
`imbalanced-learn` package, since this environment's package index is
restricted to what's already pre-installed. The algorithm itself is the
standard one: for each minority-class sample, find its k nearest minority
neighbors and generate a synthetic point along the line segment to a
randomly chosen neighbor.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_oversample(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int = 5, random_state: int = 42):
    """Generate `n_synthetic` synthetic samples for a minority class.

    Parameters
    ----------
    X_minority : array of shape (n_minority_samples, n_features)
    n_synthetic : number of synthetic samples to create
    k_neighbors : neighborhood size used for interpolation
    """
    rng = np.random.RandomState(random_state)
    n_minority = X_minority.shape[0]
    k = min(k_neighbors, n_minority - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_minority)
    _, neighbor_idx = nn.kneighbors(X_minority)
    neighbor_idx = neighbor_idx[:, 1:]  # drop self-match

    # Vectorized generation: pick a random base sample + a random neighbor of
    # that sample for every synthetic point at once, instead of a Python loop.
    sample_idx = rng.randint(0, n_minority, size=n_synthetic)
    neighbor_choice = rng.randint(0, k, size=n_synthetic)
    neighbor_idx_chosen = neighbor_idx[sample_idx, neighbor_choice]
    gaps = rng.rand(n_synthetic, 1)

    base_points = X_minority[sample_idx]
    neighbor_points = X_minority[neighbor_idx_chosen]
    synthetic = base_points + gaps * (neighbor_points - base_points)

    return synthetic


def smote_balance(X: np.ndarray, y: np.ndarray, minority_label=1, k_neighbors: int = 5, random_state: int = 42):
    """Balance a binary-labelled dataset by oversampling the minority class to 1:1."""
    X = np.asarray(X)
    y = np.asarray(y)
    minority_mask = y == minority_label
    majority_mask = ~minority_mask

    n_majority = majority_mask.sum()
    n_minority = minority_mask.sum()
    n_needed = n_majority - n_minority
    if n_needed <= 0:
        return X, y

    synthetic_X = smote_oversample(X[minority_mask], n_needed, k_neighbors, random_state)
    synthetic_y = np.full(n_needed, minority_label)

    X_bal = np.vstack([X, synthetic_X])
    y_bal = np.concatenate([y, synthetic_y])
    return X_bal, y_bal
