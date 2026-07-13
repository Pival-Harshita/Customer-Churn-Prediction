import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def create_mapping(cluster_labels, true_labels):
    mapping = {}

    for cluster in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster)
        majority = np.bincount(true_labels[indices]).argmax()
        mapping[cluster] = majority

    return mapping


def apply_mapping(cluster_labels, mapping):
    return np.array([mapping[c] for c in cluster_labels])


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return acc, prec, rec, f1