"""
Postprocessing functions to improve predictions.
"""
from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN

EPS = 10  # maximum distance between two samples of a cluster
MIN_NB_POINTS = 500  # minimum number of points to keep a cluster


def get_cleaned_pred(pred_array: np.ndarray) -> np.ndarray:
    """Cleans a prediction by removing small clusters.

    Args:
        pred_array (np.ndarray): prediction of type numpy array.

    Returns:
        np.ndarray: cleaned prediction.
    """
    # Transform prediction to an array of points
    x_arr, y_arr = np.where(pred_array > 0)
    X = np.array([[x, y] for x, y in zip(x_arr, y_arr)])

    # Case where no roads
    if len(X) == 0:
        return pred_array

    # Clustering
    dbscan = DBSCAN(eps=EPS)
    labels = dbscan.fit_predict(X)

    # Count number of points per cluster
    points_per_cluster = Counter(labels)

    # Drop small clusters
    cluster_to_keep = [
        i for i in points_per_cluster if points_per_cluster[i] >= MIN_NB_POINTS
    ]
    mask = np.isin(labels, cluster_to_keep)

    # Create new prediction with removed small clusters
    new_pred_array = np.zeros_like(pred_array)
    new_pred_array[x_arr[mask], y_arr[mask]] = 255

    return new_pred_array
