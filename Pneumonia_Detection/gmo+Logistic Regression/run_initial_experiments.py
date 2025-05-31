# -*- coding: utf-8 -*-
"""
Script to run classification experiments for pneumonia detection.
"""
import numpy as np
import os
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
import pandas as pd
import time

from run_lr_with_gmo_improved import plot_pr_curve
from matplotlib import font_manager, rcParams
# أضف مسار الخط اللي نزّلته
font_manager.fontManager.addfont(
    "C:/Users/ADMIN/Desktop/gmo lr/الخط/Amiri-Regular.ttf"
)
# حدّد العائلة الافتراضية
rcParams['font.family'] = "Amiri"
# منع عرض علامة الطرح (−) بشكل غريب
rcParams['axes.unicode_minus'] = False

# --- Configuration ---
FEATURE_FILE = "X_features.npy"
LABEL_FILE = "y_labels.npy"
RESULTS_DIR = "experiment_results"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Helper Functions ---
def load_data(feature_path, label_path):
    """Loads features and labels."""
    print(f"تحميل السمات من {feature_path}")
    X = np.load(feature_path)
    print(f"تحميل الملصقات من {label_path}")
    y = np.load(label_path)
    print(f"شكل السمات: {X.shape}, شكل الملصقات: {y.shape}")
    return X, y

def arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

def plot_roc_curve(y_true, y_prob, classifier_name, fold_name, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=arabic_text(f'منحنى ROC (المساحة = {roc_auc:.2f})'))
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel(arabic_text('معدل الإيجابية الخاطئة (False Positive Rate)'))
    plt.ylabel(arabic_text('معدل الإيجابية الصحيحة (True Positive Rate)'))
    plt.title(arabic_text(f'منحنى ROC للمصنف {classifier_name} - {fold_name}'))
    
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"تم حفظ منحنى ROC في: {save_path}")
    
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classifier_name, fold_name, class_names, save_path=None):
    # إعادة تشكيل أسماء الفئات بالعربية
    class_names_arabic = [arabic_text(name) for name in class_names]
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_arabic)
    
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title(arabic_text(f'مصفوفة الارتباك للمصنف {classifier_name} - {fold_name}'))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"تم حفظ مصفوفة الارتباك في: {save_path}")
    
    plt.close()

def train_evaluate_classifier(X_train, y_train, X_test, y_test, classifier, classifier_name, results_dir):
    """Trains a classifier and evaluates it."""
    print(f"\n--- تدريب وتقييم: {classifier_name} ---")
    os.makedirs(results_dir, exist_ok=True)

    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = classifier.predict(X_test)
    inference_time = time.time() - start_time

    y_prob = np.zeros(len(y_test))
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test)[:, 1]
    else: # For SVM with no probability
        y_prob = classifier.decision_function(X_test)
        # Scale to [0,1] for ROC consistency if needed, though roc_auc_score handles it
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8) # Add epsilon to avoid division by zero


    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"الدقة: {accuracy:.4f}")
    print(f"الضبط: {precision:.4f}")
    print(f"الاستدعاء: {recall:.4f}")
    print(f"مقياس F1: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"زمن التدريب: {train_time:.4f} ثانية")
    print(f"زمن الاستدلال (لكامل مجموعة الاختبار): {inference_time:.4f} ثانية")
    print(f"زمن الاستدلال (للعينة الواحدة): {inference_time/len(y_test):.6f} ثانية")


    # Plotting
    class_names = ['طبيعي', 'التهاب رئوي']
    cm_path = os.path.join(results_dir, f"{classifier_name.replace(' ', '_')}_confusion_matrix.png")
    roc_path = os.path.join(results_dir, f"{classifier_name.replace(' ', '_')}_roc_curve.png")
    plot_confusion_matrix(y_test, y_pred, classifier_name, "Full Features", class_names, cm_path)
    plot_roc_curve(y_test, y_prob, classifier_name, "Full Features", roc_path)
    pr_path = os.path.join(results_dir, f"{classifier_name.replace(' ', '_')}_GMO_pr_curve.png")
    plot_pr_curve(y_test, y_prob, classifier_name, "GMO Features", pr_path)


    return {
        "classifier": classifier_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "train_time_seconds": train_time,
        "inference_time_total_seconds": inference_time,
        "inference_time_per_sample_seconds": inference_time/len(y_test),
        "confusion_matrix_path": cm_path,
        "roc_curve_path": roc_path
    }

# --- Main Execution ---
def main():
    print("بدء تجارب التصنيف...")
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"تم إنشاء مجلد النتائج: {RESULTS_DIR}")

    X, y = load_data(FEATURE_FILE, LABEL_FILE)

    # Split data
    print(f"تقسيم البيانات: حجم الاختبار = {TEST_SIZE}, الحالة العشوائية = {RANDOM_STATE}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"أحجام مجموعات التدريب: السمات {X_train.shape}, الملصقات {y_train.shape}")
    print(f"أحجام مجموعات الاختبار: السمات {X_test.shape}, الملصقات {y_test.shape}")

    # Scale features
    print("تحجيم السمات باستخدام StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("تم تحجيم السمات.")

    # Define classifiers
    classifiers = {
        "الانحدار اللوجستي": LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', max_iter=1000),
        "SVM الخطي": SVC(kernel='linear', probability=True, random_state=RANDOM_STATE), # probability=True for ROC AUC
        "Naive Bayes (Gaussian)": GaussianNB(),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric='manhattan') # Original KNN setup
    }

    all_results = []

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        classifier_results_dir = os.path.join(RESULTS_DIR, name.replace(' ', '_'))
        result = train_evaluate_classifier(X_train_scaled, y_train, X_test_scaled, y_test, clf, name, classifier_results_dir)
        all_results.append(result)

    # Save all results to a CSV file
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "all_classifier_results_full_features.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig') # utf-8-sig for Arabic characters in Excel
    print(f"\nتم حفظ جميع النتائج في: {csv_path}")

    print("\nانتهت جميع التجارب.")

if __name__ == "__main__":
    # Change working directory to the script's directory to ensure relative paths work
    # This is important because the shell_exec might run from /home/ubuntu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"تم تغيير دليل العمل إلى: {script_dir}")
    main()

