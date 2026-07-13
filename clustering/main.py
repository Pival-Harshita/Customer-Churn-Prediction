import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kmeans_cluster import run_kmeans
from kmedoids_cluster import run_kmedoids, predict_kmedoids
from xmeans_cluster import run_xmeans
from random_cluster import run_random
from evaluate_clusters import create_mapping, apply_mapping, evaluate



SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ---------------- LOAD DATA ----------------
print("\nLoading Processed Dataset\n")

X_train = pd.read_csv("../data_preprocessing/processed_X_train.csv")
X_test  = pd.read_csv("../data_preprocessing/processed_X_test.csv")

y_train = pd.read_csv("../data_preprocessing/processed_y_train.csv")
y_test  = pd.read_csv("../data_preprocessing/processed_y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


#Keep original copy for saving
X_train_original = X_train.copy()
X_test_original = X_test.copy()


# ---------------- SCALE ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


k = 4
results = {}


# ---------------- KMEANS ----------------
tr_km, te_km = run_kmeans(X_train_scaled, X_test_scaled, k)
map_km = create_mapping(tr_km, y_train)
pred_km = apply_mapping(te_km, map_km)
results["KMeans"] = evaluate(y_test, pred_km)


# ---------------- KMEDOIDS ----------------
tr_med, medoids = run_kmedoids(X_train_scaled, k)
te_med = predict_kmedoids(X_train_scaled, X_test_scaled, medoids)
map_med = create_mapping(tr_med, y_train)
pred_med = apply_mapping(te_med, map_med)
results["KMedoids"] = evaluate(y_test, pred_med)


# ---------------- XMEANS ----------------
tr_xm, te_xm, best_k = run_xmeans(X_train_scaled, X_test_scaled)
map_xm = create_mapping(tr_xm, y_train)
pred_xm = apply_mapping(te_xm, map_xm)
results["XMeans"] = evaluate(y_test, pred_xm)


# ---------------- RANDOM ----------------
tr_rand, te_rand = run_random(X_train_scaled, X_test_scaled, k)
map_rand = create_mapping(tr_rand, y_train)
pred_rand = apply_mapping(te_rand, map_rand)
results["Random"] = evaluate(y_test, pred_rand)


# ---------------- PRINT RESULTS ----------------
best_model = None
best_acc = 0

for name, (acc, prec, rec, f1) in results.items():
    print(f"\n{name}")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    if acc > best_acc:
        best_acc = acc
        best_model = name


print("\nBEST MODEL BASED ON ACCURACY:", best_model)

models = list(results.keys())
accuracies = [results[m][0] for m in models]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

precisions = [results[m][1] for m in models]
recalls = [results[m][2] for m in models]
f1s = [results[m][3] for m in models]

x = range(len(models))

plt.figure()
plt.plot(x, precisions, marker='o', label='Precision')
plt.plot(x, recalls, marker='o', label='Recall')
plt.plot(x, f1s, marker='o', label='F1 Score')

plt.xticks(x, models, rotation=30)
plt.title("Performance Metrics Comparison")
plt.legend()
plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=tr_km)
plt.title("KMeans Clusters (PCA View)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# ================= SAVE CLUSTERED DATASETS =================

print("\nSaving clustered datasets...\n")

# Convert original data to DataFrame
train_df = pd.DataFrame(X_train_original)
test_df = pd.DataFrame(X_test_original)

# Add cluster labels
train_df["kmeans_label"] = tr_km
test_df["kmeans_label"] = te_km

# Add target
train_df["target"] = y_train
test_df["target"] = y_test

#Single Classification
train_data = pd.DataFrame(X_train_original)
test_data = pd.DataFrame(X_test_original)

train_data["kmeans_label"] = tr_km
test_data["kmeans_label"] = te_km

train_data["kmedoids_label"] = tr_med
test_data["kmedoids_label"] = te_med

train_data["xmeans_label"] = tr_xm
test_data["xmeans_label"] = te_xm

train_data["random_label"] = tr_rand
test_data["random_label"] = te_rand

# Add target
train_data["target"] = y_train
test_data["target"] = y_test

# Save files
train_df.to_csv("clustered_X_train.csv", index=False)
test_df.to_csv("clustered_X_test.csv", index=False)

train_data.to_csv("clustered_X_train_Single.csv", index=False)
test_data.to_csv("clustered_X_test_Single.csv", index=False)

print("Clustered train and test datasets saved successfully!")