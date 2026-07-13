from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_xmeans(X_train, X_test, max_k=10):
    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42,n_init=10)
        labels = model.fit_predict(X_train)
        score = silhouette_score(X_train, labels)

        if score > best_score:
            best_score = score
            best_k = k

    final_model = KMeans(n_clusters=best_k, random_state=42)

    train_clusters = final_model.fit_predict(X_train)
    test_clusters = final_model.predict(X_test)

    return train_clusters, test_clusters, best_k