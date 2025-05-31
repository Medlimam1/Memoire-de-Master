import numpy as np
import matplotlib.pyplot as plt
import os
from utils import evaluate_features_with_knn
from sklearn.metrics import roc_curve, auc

from feature_selection_gmo import GMO
from feature_selection_ga import ga_feature_selection
from feature_selection_pso import pso_feature_selection
from feature_selection_chi2 import evaluate_chi2_with_knn

# تحميل البيانات من ملفات محفوظة مسبقاً
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

save_path = "GMO_Results"
os.makedirs(save_path, exist_ok=True)

methods = {
    "GMO": lambda: GMO(X, y).optimize(),
    "GA": lambda: ga_feature_selection(X, y),
    "PSO": lambda: pso_feature_selection(X, y),
    "Chi2": lambda: evaluate_chi2_with_knn(X, y, k=800)
}

colors = {
    "GMO": 'blue',
    "GA": 'green',
    "PSO": 'orange',
    "Chi2": 'red'
}

plt.figure(figsize=(8, 6))

for name, method in methods.items():
    selected_indices, acc, f1, roc_auc = method()
    print(f"{name} -> Accuracy: {acc:.4f}, F1-score: {f1:.4f}, AUC: {roc_auc:.4f}, Selected features: {len(selected_indices)}")

    # ROC Curve
    _, _, _, y_score, y_test = evaluate_features_with_knn(X, y, selected_indices)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=colors[name])

# إعدادات الرسم
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Comparative ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
roc_path = os.path.join(save_path, "comparative_roc_curve.png")
plt.savefig(roc_path)
print("تم حفظ منحنى ROC في:", roc_path)
plt.show()
