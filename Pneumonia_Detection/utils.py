import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_features_with_knn(X, y, selected_features, test_size=0.2, random_state=42):
    """يقيّم مجموعة من السمات باستخدام مصنف KNN ويحسب مؤشرات الأداء"""
    X_selected = X[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=random_state)

    model = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)

    return accuracy, f1, auc, y_score, y_test
