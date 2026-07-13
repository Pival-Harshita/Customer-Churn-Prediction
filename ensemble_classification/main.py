import pandas as pd

# =========================
# MODELS
# =========================
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================
train = pd.read_csv("../clustering/clustered_X_train.csv")
test  = pd.read_csv("../clustering/clustered_X_test.csv")

target = "target"

X_train = train.drop(columns=[target])
y_train = train[target]

X_test = test.drop(columns=[target])
y_test = test[target]

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =========================
# BASE MODELS
# =========================
models = {
    "GBT": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42
    ),

    "DT": DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=5,
        random_state=42
    ),

    "RF": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        weights='distance'
    ),

    "NB": GaussianNB(),

    "SVM": SVC(
        C=10,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    ),

    "DL": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )
}

# =========================
# COMBINATIONS
# =========================
combinations = [
    ("GBT+DT+RF", ["GBT","DT","RF"]),
    ("GBT+DT+KNN", ["GBT","DT","KNN"]),
    ("GBT+DT+DL", ["GBT","DT","DL"]),
    ("GBT+DT+NB", ["GBT","DT","NB"]),
    ("GBT+DT+NB(K)", ["GBT","DT","SVM"]),
    ("DT+RF+KNN", ["DT","RF","KNN"]),
    ("DT+RF+DL", ["DT","RF","DL"]),
    ("DT+RF+NB", ["DT","RF","NB"]),
    ("RF+KNN+DL", ["RF","KNN","DL"]),
    ("RF+KNN+NB", ["RF","KNN","NB"]),
    ("RF+KNN+NB(K)", ["RF","KNN","SVM"]),
    ("NB+NB(K)+KNN", ["NB","SVM","KNN"]),
    ("NB+NB(K)+DL", ["NB","SVM","DL"]),
]

results = []

# =========================
# METRICS FUNCTION
# =========================
def evaluate(y_test, y_pred):
    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average='weighted'),
        recall_score(y_test, y_pred, average='weighted'),
        f1_score(y_test, y_pred, average='weighted')
    )

# =========================
# LOOP THROUGH COMBINATIONS
# =========================
for name, model_keys in combinations:

    print(f"\n========== KMeans + {name} ==========")

    # FIX: UNIQUE NAMES
    estimators = []
    name_count = {}

    for k in model_keys:
        if k in name_count:
            name_count[k] += 1
            new_name = f"{k}_{name_count[k]}"
        else:
            name_count[k] = 1
            new_name = k

        estimators.append((new_name, models[k]))

    # ================= VOTING =================
    voting = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[2 if "GBT" in n else 2 if "RF" in n else 1 for n, _ in estimators]
    )
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_test)

    acc, pre, rec, f1 = evaluate(y_test, y_pred)

    print("\nVoting")
    print("Accuracy :", acc)
    print("Precision:", pre)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    results.append({"Model": f"KMeans + {name}+Voting", "Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1})

    # ================= BAGGING =================
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    y_pred = bagging.predict(X_test)

    acc, pre, rec, f1 = evaluate(y_test, y_pred)

    print("\nBagging")
    print("Accuracy :", acc)
    print("Precision:", pre)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    results.append({"Model": f" KMeans+{name}+Bagging", "Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1})

    # ================= ADABOOST =================
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=200,
        learning_rate=0.5,
        random_state=42
    )
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)

    acc, pre, rec, f1 = evaluate(y_test, y_pred)

    print("\nAdaBoost")
    print("Accuracy :", acc)
    print("Precision:", pre)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    results.append({"Model": f"KMeans+{name}+AdaBoost", "Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1})

    # ================= STACKING =================
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        ),
        passthrough=True
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)

    acc, pre, rec, f1 = evaluate(y_test, y_pred)

    print("\nStacking")
    print("Accuracy :", acc)
    print("Precision:", pre)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    results.append({"Model": f"KMeans+{name}+Stacking", "Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1})

# =========================
# BEST MODEL
# =========================
results_df = pd.DataFrame(results)

best = results_df.loc[results_df["Accuracy"].idxmax()]

print("\n==============================")
print("BEST ENSEMBLE MODEL")
print("==============================")
print(best)
