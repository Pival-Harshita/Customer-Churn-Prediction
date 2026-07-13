import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, recall_score,
                              precision_score, f1_score,
                              mean_squared_error, mean_absolute_error)

np.random.seed(42)

def evaluate(y_true, y_pred):
    acc  = round(accuracy_score (y_true, y_pred) * 100, 2)
    rec  = round(recall_score   (y_true, y_pred, average='macro', zero_division=0) * 100, 2)
    pre  = round(precision_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2)
    f1   = round(f1_score       (y_true, y_pred, average='macro', zero_division=0) * 100, 2)
    mse  = round(mean_squared_error (y_true, y_pred), 4)
    rmse = round(np.sqrt(mse), 4)
    mae  = round(mean_absolute_error(y_true, y_pred), 4)
    return acc, rec, pre, f1, rmse, mse, mae


# --------------------------------------------------
# CLASSIFIERS — tuned for best accuracy
# --------------------------------------------------
def get_classifiers():
    return {
        "GBT"  : GradientBoostingClassifier(
                     n_estimators=300, learning_rate=0.05,
                     max_depth=5, subsample=0.8, random_state=42),
        "DT"   : DecisionTreeClassifier(random_state=42),
        "RF"   : RandomForestClassifier(
                     n_estimators=300, random_state=42, n_jobs=-1),
        "KNN"  : KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "DL"   : MLPClassifier(
                     hidden_layer_sizes=(128, 64, 32), max_iter=1000,
                     early_stopping=True, random_state=42),
        "NB"   : GaussianNB(),
        "NB(K)": GaussianNB(var_smoothing=1e-8),
    }


CLUSTER_COLS = {
    "kmeans_label"  : "K-means",
    "kmedoids_label": "K-med",
    "xmeans_label"  : "X-means",
    "random_label"  : "Random",
}

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    print("\nLoading Clustered Dataset\n")

    # Load CSVs — these contain base features + cluster labels + target
    train_df = pd.read_csv("../clustering/clustered_X_train_Single.csv")
    test_df  = pd.read_csv("../clustering/clustered_X_test_Single.csv")

    print("Train shape:", train_df.shape)
    print("Test  shape:", test_df.shape)
    print("\nColumns found:")
    print(list(train_df.columns))

    # --------------------------------------------------
    # Separate target from features
    # clustering code saved target column inside the CSV
    # --------------------------------------------------
    y_train = train_df["target"].values
    y_test  = test_df["target"].values

    train_df = train_df.drop(columns=["target"])
    test_df  = test_df.drop(columns=["target"])

    # --------------------------------------------------
    # Validate cluster columns exist
    # --------------------------------------------------
    missing = [c for c in CLUSTER_COLS if c not in train_df.columns]
    if missing:
        print("\n" + "=" * 55)
        print("ERROR: These cluster columns are missing:")
        print(missing)
        print("Re-run clustering/main.py first, then retry.")
        print("=" * 55)
        return

    # All cluster column names
    all_cluster_cols = list(CLUSTER_COLS.keys())

    # Base feature columns (no cluster labels)
    base_cols = [c for c in train_df.columns if c not in all_cluster_cols]

    print(f"\nBase features : {len(base_cols)}")
    print(f"Cluster cols  : {all_cluster_cols}")

    # Extract base features
    X_train_base = train_df[base_cols]
    X_test_base  = test_df[base_cols]

    classifiers = get_classifiers()
    all_results = []

    # --------------------------------------------------
    # LOOP: each cluster method x each classifier
    # --------------------------------------------------
    for cluster_col, short_name in CLUSTER_COLS.items():

        print("\n" + "=" * 62)
        print(f"  CLUSTERING : {short_name}  (column: {cluster_col})")
        print("=" * 62)
        print(f"{'Technique':<22} {'Accuracy':>10} {'Recall':>8} "
              f"{'Precision':>10} {'F-measure':>10}")
        print("-" * 62)

        # Combine base features + this cluster column only
        X_train_new = X_train_base.copy()
        X_test_new  = X_test_base.copy()
        X_train_new[cluster_col] = train_df[cluster_col].values
        X_test_new[cluster_col]  = test_df[cluster_col].values

        for clf_name, model in classifiers.items():

            model.fit(X_train_new, y_train)
            y_pred = model.predict(X_test_new)

            acc, rec, pre, f1, rmse, mse, mae = evaluate(y_test, y_pred)

            technique = f"{short_name}+{clf_name}"
            print(f"{technique:<22} {acc:>10} {rec:>8} {pre:>10} {f1:>10}")

            all_results.append({
                "Technique" : technique,
                "cluster"   : cluster_col,
                "classifier": clf_name,
                "accuracy"  : acc,
                "recall"    : rec,
                "precision" : pre,
                "f1"        : f1,
                "rmse"      : rmse,
                "mse"       : mse,
                "mae"       : mae
            })

    # --------------------------------------------------
    # FULL SUMMARY TABLE
    # --------------------------------------------------
    print("\n\n" + "=" * 62)
    print("  FULL SUMMARY TABLE")
    print("=" * 62)
    print(f"{'Technique':<22} {'Accuracy':>10} {'Recall':>8} "
          f"{'Precision':>10} {'F-measure':>10}")
    print("-" * 62)

    for r in all_results:
        print(f"{r['Technique']:<22} {r['accuracy']:>10} {r['recall']:>8} "
              f"{r['precision']:>10} {r['f1']:>10}")

    # Average per clustering method
    print("\n" + "-" * 62)
    print("AVERAGE PER CLUSTERING METHOD")
    print("-" * 62)
    print(f"{'Clustering':<22} {'Accuracy':>10} {'Recall':>8} "
          f"{'Precision':>10} {'F-measure':>10}")
    print("-" * 62)

    for cluster_col, short_name in CLUSTER_COLS.items():
        subset = [r for r in all_results if r["cluster"] == cluster_col]
        print(f"{short_name:<22} "
              f"{round(np.mean([r['accuracy']  for r in subset]), 2):>10} "
              f"{round(np.mean([r['recall']    for r in subset]), 2):>8} "
              f"{round(np.mean([r['precision'] for r in subset]), 2):>10} "
              f"{round(np.mean([r['f1']        for r in subset]), 2):>10}")

    print("=" * 62)

    # --------------------------------------------------
    # BEST MODEL — highest accuracy (no hardcoding)
    # --------------------------------------------------
    best = max(all_results, key=lambda r: r["accuracy"])

    print("\n" + "-" * 45)
    print("\nBEST HYBRID MODEL (Single Classifier)\n")
    print(f"Model     : {best['Technique']}")
    print(f"Accuracy  : {best['accuracy']}")
    print(f"Recall    : {best['recall']}")
    print(f"Precision : {best['precision']}")
    print(f"F1 Score  : {best['f1']}")
    print(f"RMSE      : {best['rmse']}")
    print(f"MSE       : {best['mse']}")
    print(f"MAE       : {best['mae']}")
    print("\n" + "-" * 45)


if __name__ == "__main__":
    main()