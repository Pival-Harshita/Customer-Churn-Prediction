import numpy as np
from sklearn.metrics import pairwise_distances

def run_kmedoids(X, k, max_iter=100):
    np.random.seed(42)
    n = X.shape[0]
    medoids = np.random.choice(n, k, replace=False)

    for _ in range(max_iter):
        distances = pairwise_distances(X, X[medoids])
        labels = np.argmin(distances, axis=1)

        new_medoids = []

        for i in range(k):
            cluster_points = np.where(labels == i)[0]

            if len(cluster_points) == 0:
                new_medoids.append(medoids[i])
                continue

            cluster_distances = pairwise_distances(X[cluster_points])
            total_distances = cluster_distances.sum(axis=1)

            new_medoid = cluster_points[np.argmin(total_distances)]
            new_medoids.append(new_medoid)

        new_medoids = np.array(new_medoids)

        if np.all(medoids == new_medoids):
            break

        medoids = new_medoids

    return labels, medoids


def predict_kmedoids(X_train, X_test, medoids):
    distances = pairwise_distances(X_test, X_train[medoids])
    return np.argmin(distances, axis=1)