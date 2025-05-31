import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from utils import evaluate_features_with_knn

def evaluate_chi2_with_knn(X, y, k=800):
    """
    اختيار السمات باستخدام طريقة Chi2 وتقييم الأداء بمصنف KNN.
    
    المعاملات:
    - X: مصفوفة السمات الأصلية
    - y: المتجه الهدف
    - k: عدد السمات المراد اختيارها (افتراضي = 800)

    القيم المُعادة:
    - selected_indices: فهارس السمات المنتقاة
    - acc: الدقة
    - f1: معيار F1
    - auc: المساحة تحت منحنى ROC
    """
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)

    # التقييم باستخدام دالة موحّدة
    acc, f1, auc, _, _ = evaluate_features_with_knn(X, y, selected_indices)

    return selected_indices, acc, f1, auc
