import numpy as np

def run_random(X_train, X_test, k):
    np.random.seed(42)
    train_clusters = np.random.randint(0, k, size=len(X_train))
    test_clusters = np.random.randint(0, k, size=len(X_test))

    return train_clusters, test_clusters