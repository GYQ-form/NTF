from typing import Dict
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mode


class KNNLabelAnnotator:
    """
    KNN majority-vote annotator that transfers categorical labels
    (e.g., cell types) from reference points to arbitrary query coordinates.
    """
    def __init__(self, xyz: np.ndarray, cat_labels: Dict[str, pd.Categorical], k_neighbors: int = 5):
        self.k_neighbors = k_neighbors
        self.tree = cKDTree(xyz)
        self.n_ref = xyz.shape[0]
        self.codes = {}
        self.categories = {}
        for key, cat in cat_labels.items():
            cat = pd.Categorical(cat)
            self.codes[key] = cat.codes
            self.categories[key] = cat.categories

    def predict(self, xyz_query: np.ndarray, chunk_size: int = 2_000_000) -> Dict[str, np.ndarray]:
        """
        Predict categorical labels for query coordinates via KNN majority voting.
        Returns a dict mapping each categorical key to an array of predicted labels.
        """
        xyz_query = np.asarray(xyz_query)
        n_query = xyz_query.shape[0]
        k = min(self.k_neighbors, self.n_ref)

        voted_results = {key: np.zeros(n_query, dtype=np.int64) for key in self.codes}

        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            _, indices = self.tree.query(xyz_query[start:end], k=k)
            if k == 1:
                indices = indices.reshape(-1, 1)

            for key, codes in self.codes.items():
                neighbor_codes = codes[indices]
                try:
                    voted_modes, _ = mode(neighbor_codes, axis=1, keepdims=False)
                except TypeError:
                    voted_modes, _ = mode(neighbor_codes, axis=1)
                voted_results[key][start:end] = voted_modes.flatten()

        return {key: np.asarray(self.categories[key])[voted_results[key]] for key in self.codes}
