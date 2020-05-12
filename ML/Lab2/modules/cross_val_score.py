from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score
import numpy as np
def cross_val_score(model, X, y):
    acc_scores = []
    prec_scores = []
    roc_scores = []

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict_classes(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        prec_score = precision_score(y_test, y_pred)
        roc_score = roc_auc_score(y_test, y_pred)
        acc_scores.append(acc_score)
        prec_scores.append(prec_score)
        roc_scores.append(roc_score)

    score = np.mean(acc_scores), np.mean(prec_scores), np.mean(roc_scores)
    return score

