from sklearn.cluster import KMeans

def run_kmeans(X_train, X_test, k):
    model = KMeans(n_clusters=k, random_state=42,n_init=10)

    train_clusters = model.fit_predict(X_train)
    test_clusters = model.predict(X_test)

    return train_clusters, test_clusters