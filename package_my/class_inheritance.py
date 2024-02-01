from sklearn.ensemble._forest import ForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def predict_my(self, X):
    proba = self.predict_proba(X)
    if self.n_outputs_ == 1:
        return self.classes_.take(np.argmax(proba, axis=1), axis=0), proba
    else:
        n_samples = proba[0].shape[0]
        class_type = self.classes_[0].dtype
        predictions = np.empty((n_samples, self.n_outputs_), dtype=class_type)
        for k in range(self.n_outputs_):
            predictions[:, k] = self.classes_[k].take(
                np.argmax(proba[k], axis=1), axis=0
            )
        return predictions


def score_my(self, X, y, sample_weight=None):
    pred_y = self.predict(X)
    acc = accuracy_score(y, pred_y, sample_weight=sample_weight)
    _, _, mf1, _ = precision_recall_fscore_support(y, pred_y, average='macro', zero_division=1)
    return acc, mf1


def class_inheritance():
    ForestClassifier.predict = predict_my
    ClassifierMixin.score = score_my
