#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
نص برمجي متكامل لتشغيل الانحدار اللوجستي مع اختيار الميزات GMO
باستخدام 5-Fold Cross-Validation.

تم تطويره لتلبية متطلبات مذكرة:
Geometric Mean Optimizer for Solving a Problem of a Real-Life Embedded System Application
تطبيق: اكتشاف الالتهاب الرئوي (Pneumonia Detection) باستخدام اختيار الميزات بخوارزمية GMO
وتحسين أداء مصنف الانحدار اللوجستي (Logistic Regression Classifier).

المميزات:
- استخدام 5-Fold Cross-Validation مع StratifiedKFold
- ضبط random_seed مختلف لكل طية
- استخدام خوارزمية GMO المعدّلة مع الانحدار اللوجستي كمقوم داخلي
- حساب جميع مؤشرات الأداء: Accuracy، Precision، Recall، F1-Score، ROC AUC
- حساب المتوسط والانحراف المعياري لكل مقياس
- رسم وحفظ الرسومات التالية لكل طية: Confusion Matrix، ROC Curve، PR Curve
- رسم Boxplot شامل لكل مقياس أداء
- رسم Bar chart للمقاييس الأساسية مع أشرطة الانحراف المعياري
"""

import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    precision_recall_curve, average_precision_score
)
import arabic_reshaper
from bidi.algorithm import get_display  # لدعم اللغة العربية في الرسومات
from matplotlib import font_manager, rcParams
# أضف مسار الخط اللي نزّلته
font_manager.fontManager.addfont(
    "C:/Users/ADMIN/Desktop/gmo lr/الخط/Amiri-Regular.ttf"
)
# حدّد العائلة الافتراضية
rcParams['font.family'] = "Amiri"
# منع عرض علامة الطرح (−) بشكل غريب
rcParams['axes.unicode_minus'] = False

# استيراد GMO المحسن وأدوات اختيار الميزات المحدثة
from GMO_improved import GMO
from feature_selection_utils_improved import (
    Data as FSData, 
    Solution as FSSolution, 
    compute_fitness,
    initialize_agents,
    sort_agents,
    validate_FS,
    plot_pr_curve
)

# --- التكوين ---
FEATURE_FILE = "X_features.npy"
LABEL_FILE = "y_labels.npy"
RESULTS_DIR = "experiment_results/LogisticRegression_GMO_CV"
BASE_RANDOM_STATE = 42
NUM_FOLDS = 5
GMO_NUM_AGENTS = 20
GMO_MAX_ITER = 30
CLASS_NAMES = ['طبيعي', 'التهاب رئوي']

# --- دوال مساعدة ---
def load_data(feature_path, label_path):
    """تحميل البيانات من ملفات numpy"""
    print(f"تحميل السمات من {feature_path}")
    X = np.load(feature_path)
    print(f"تحميل الملصقات من {label_path}")
    y = np.load(label_path)
    print(f"شكل السمات: {X.shape}, شكل الملصقات: {y.shape}")
    return X, y

def plot_roc_curve(y_true, y_prob, classifier_name, fold_name, save_path=None):
    """رسم منحنى ROC وحساب مساحة المنحنى"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=get_display(arabic_reshaper.reshape(f'منحنى ROC (المساحة = {roc_auc:.2f})')))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(get_display(arabic_reshaper.reshape('معدل الإيجابية الخاطئة (False Positive Rate)')))
    plt.ylabel(get_display(arabic_reshaper.reshape('معدل الإيجابية الصحيحة (True Positive Rate)')))
    plt.title(get_display(arabic_reshaper.reshape(f'منحنى ROC للمصنف {classifier_name} - {fold_name}')))
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"تم حفظ منحنى ROC في: {save_path}")
    
    plt.close()
    return roc_auc

def plot_confusion_matrix(y_true, y_pred, classifier_name, fold_name, class_names, save_path=None):
    """رسم مصفوفة الارتباك"""
    cm = confusion_matrix(y_true, y_pred)

    # إعادة تشكيل وربط نصوص أسماء الفئات
    reshaped_labels = [
        get_display(arabic_reshaper.reshape(name))
        for name in class_names
    ]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=reshaped_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)

    # عنوان الدالة مع reshaping أيضاً
    title = get_display(arabic_reshaper.reshape(f'مصفوفة الارتباك للمصنف {classifier_name} - {fold_name}'))
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"تم حفظ مصفوفة الارتباك في: {save_path}")
    
    plt.close()
    return cm

def plot_boxplots(metrics_df, save_dir):
    """رسم boxplots لمقاييس الأداء المختلفة"""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=metrics_df[metrics_to_plot])
    plt.title(get_display(arabic_reshaper.reshape('توزيع مقاييس الأداء عبر الطيات')))
    plt.ylabel(get_display(arabic_reshaper.reshape('القيمة')))
    plt.xlabel(get_display(arabic_reshaper.reshape('مقياس الأداء')))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # تعريب أسماء المقاييس على المحور السيني
    arabic_names = {
        'accuracy': get_display(arabic_reshaper.reshape('الدقة')),
        'precision': get_display(arabic_reshaper.reshape('الضبط')),
        'recall': get_display(arabic_reshaper.reshape('الاستدعاء')),
        'f1_score': get_display(arabic_reshaper.reshape('مقياس F1')),
        'roc_auc': get_display(arabic_reshaper.reshape('مساحة ROC'))
    }
    
    plt.xticks(range(len(metrics_to_plot)), [arabic_names[m] for m in metrics_to_plot])
    
    save_path = os.path.join(save_dir, "performance_metrics_boxplot.png")
    plt.savefig(save_path)
    print(f"تم حفظ رسم صندوقي للمقاييس في: {save_path}")
    plt.close()

def plot_bar_chart_with_error_bars(metrics_summary, save_dir):
    """رسم bar chart للمقاييس الأساسية مع أشرطة الانحراف المعياري"""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    means = [metrics_summary[f'mean_{m}'] for m in metrics_to_plot]
    stds = [metrics_summary[f'std_{m}'] for m in metrics_to_plot]
    
    plt.figure(figsize=(12, 8))
    
    # تعريب أسماء المقاييس
    arabic_names = {
        'accuracy': get_display(arabic_reshaper.reshape('الدقة')),
        'precision': get_display(arabic_reshaper.reshape('الضبط')),
        'recall': get_display(arabic_reshaper.reshape('الاستدعاء')),
        'f1_score': get_display(arabic_reshaper.reshape('مقياس F1')),
        'roc_auc': get_display(arabic_reshaper.reshape('مساحة ROC'))
    }
    
    x_pos = np.arange(len(metrics_to_plot))
    
    plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, capsize=10)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.ylabel(get_display(arabic_reshaper.reshape('القيمة')))
    plt.title(get_display(arabic_reshaper.reshape('متوسط مقاييس الأداء مع الانحراف المعياري')))
    plt.xticks(x_pos, [arabic_names[m] for m in metrics_to_plot])
    
    # إضافة قيم المتوسط فوق كل عمود
    for i, v in enumerate(means):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    save_path = os.path.join(save_dir, "performance_metrics_bar_chart.png")
    plt.savefig(save_path)
    print(f"تم حفظ رسم شريطي للمقاييس في: {save_path}")
    plt.close()

def train_evaluate_fold(X_train, y_train, X_test, y_test, fold_idx, fold_random_state, results_dir):
    """تدريب وتقييم نموذج على طية محددة"""
    fold_dir = os.path.join(results_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    
    print(f"\n--- الطية {fold_idx}/{NUM_FOLDS} (الحالة العشوائية: {fold_random_state}) ---")
    
    # تحجيم السمات
    print("تحجيم السمات باستخدام StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("تم تحجيم السمات.")
    
    # --- اختيار الميزات باستخدام GMO ---
    print(f"\n--- بدء اختيار السمات باستخدام GMO (المُحسن: الانحدار اللوجستي) ---")
    print(f"عدد العملاء: {GMO_NUM_AGENTS}, أقصى عدد تكرارات: {GMO_MAX_ITER}")
    
    # استخدام الانحدار اللوجستي لتقييم اللياقة داخل GMO
    gmo_classifier_for_fitness = LogisticRegression(
        random_state=fold_random_state, 
        solver='liblinear', 
        max_iter=200
    )
    
    fs_data = FSData()
    fs_data.train_X = X_train_scaled
    fs_data.train_Y = y_train
    fs_data.val_X = X_train_scaled 
    fs_data.val_Y = y_train
    
    print("سيتم استخدام الانحدار اللوجستي لتقييم اللياقة داخل GMO.")
    gmo_start_time = time.time()
    
    # تشغيل GMO المحسن مع الانحدار اللوجستي للياقة
    gmo_instance = GMO(
        num_agents=GMO_NUM_AGENTS, 
        max_iter=GMO_MAX_ITER, 
        train_data=X_train_scaled,
        train_label=y_train,
        clf_model=gmo_classifier_for_fitness,
        save_conv_graph=True, 
        save_dir=fold_dir,
        fold=fold_idx
    )
    
    gmo_run_time = time.time() - gmo_start_time
    print(f"انتهى تشغيل GMO المحسن (مع الانحدار اللوجستي للياقة) في {gmo_run_time:.2f} ثانية.")
    
    best_agent_from_gmo = gmo_instance.best_agent
    selected_indices_gmo = np.flatnonzero(best_agent_from_gmo)
    num_selected_gmo = len(selected_indices_gmo)
    print(f"عدد السمات المختارة بواسطة GMO (مع الانحدار اللوجستي للياقة): {num_selected_gmo}")
    
    if num_selected_gmo == 0:
        print("لم يتم اختيار أي سمات بواسطة GMO. سيتم استخدام جميع السمات للانحدار اللوجستي.")
        X_train_selected = X_train_scaled
        X_test_selected = X_test_scaled
        selected_features_count = X_train_scaled.shape[1]
    else:
        X_train_selected = X_train_scaled[:, selected_indices_gmo]
        X_test_selected = X_test_scaled[:, selected_indices_gmo]
        selected_features_count = num_selected_gmo
    
    # --- تدريب وتقييم الانحدار اللوجستي على الميزات المختارة بواسطة GMO ---
    print(f"\n--- تدريب وتقييم الانحدار اللوجستي على {selected_features_count} سمة مختارة ---")
    
    # تدريب نموذج الانحدار اللوجستي
    lr_model = LogisticRegression(
        random_state=fold_random_state, 
        solver='liblinear', 
        max_iter=1000,
        C=1.0  # يمكن ضبط هذه القيمة أو استخدام GridSearchCV
    )
    
    start_time = time.time()
    lr_model.fit(X_train_selected, y_train)
    train_time = time.time() - start_time
    
    # تقييم النموذج
    start_time = time.time()
    y_pred = lr_model.predict(X_test_selected)
    inference_time = time.time() - start_time
    
    # حساب احتمالات الفئة الإيجابية للرسومات
    y_prob = lr_model.predict_proba(X_test_selected)[:, 1]
    
    # حساب مقاييس الأداء
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
    print(f"عدد السمات المختارة: {selected_features_count}")
    print(f"زمن التدريب: {train_time:.4f} ثانية")
    print(f"زمن الاستدلال (لكامل مجموعة الاختبار): {inference_time:.4f} ثانية")
    print(f"زمن الاستدلال (للعينة الواحدة): {inference_time/len(y_test):.6f} ثانية")
    
    # رسم وحفظ مصفوفة الارتباك
    cm_path = os.path.join(fold_dir, f"confusion_matrix_fold_{fold_idx}.png")
    cm = plot_confusion_matrix(y_test, y_pred, "الانحدار اللوجستي", f"الطية {fold_idx}", CLASS_NAMES, cm_path)
    
    # رسم وحفظ منحنى ROC
    roc_path = os.path.join(fold_dir, f"roc_curve_fold_{fold_idx}.png")
    plot_roc_curve(y_test, y_prob, "الانحدار اللوجستي", f"الطية {fold_idx}", roc_path)
    
    # رسم وحفظ منحنى PR
    pr_path = os.path.join(fold_dir, f"pr_curve_fold_{fold_idx}.png")
    plot_pr_curve(y_test, y_prob, "الانحدار اللوجستي", f"الطية {fold_idx}", pr_path)
    # حفظ المكونات (المصنف، السمات المختارة، والمقياس) لكل طية
    model_components = {
        'classifier': lr_model,
        'selected_features': selected_indices_gmo,
        'scaler': scaler
    }
    joblib.dump(
        model_components,
        os.path.join(fold_dir, f"pneumonia_model_fold_{fold_idx}.joblib")
    )
    print(f"تم حفظ نموذج الطية {fold_idx} في: {fold_dir}")

    # إرجاع نتائج هذه الطية
    fold_results = {
        "fold": fold_idx,
        "random_state": fold_random_state,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "num_selected_features": selected_features_count,
        "train_time_seconds": train_time,
        "inference_time_total_seconds": inference_time,
        "inference_time_per_sample_seconds": inference_time/len(y_test),
        "confusion_matrix_path": cm_path,
        "roc_curve_path": roc_path,
        "pr_curve_path": pr_path,
        "selected_features_indices": selected_indices_gmo.tolist() if num_selected_gmo > 0 else []
    }
    
    return fold_results

def calculate_metrics_summary(all_folds_results):
    """حساب ملخص المقاييس (المتوسط والانحراف المعياري)"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
               'num_selected_features', 'train_time_seconds', 
               'inference_time_total_seconds', 'inference_time_per_sample_seconds']
    
    summary = {}
    
    for metric in metrics:
        values = [fold_result[metric] for fold_result in all_folds_results]
        summary[f'mean_{metric}'] = np.mean(values)
        summary[f'std_{metric}'] = np.std(values)
        summary[f'min_{metric}'] = np.min(values)
        summary[f'max_{metric}'] = np.max(values)
    
    return summary

def save_results_to_csv(all_folds_results, metrics_summary, results_dir):
    """حفظ النتائج في ملفات CSV"""
    # حفظ نتائج كل طية
    folds_df = pd.DataFrame(all_folds_results)
    folds_csv_path = os.path.join(results_dir, "all_folds_results.csv")
    folds_df.to_csv(folds_csv_path, index=False, encoding='utf-8-sig')
    print(f"تم حفظ نتائج جميع الطيات في: {folds_csv_path}")
    
    # حفظ ملخص المقاييس
    summary_df = pd.DataFrame([metrics_summary])
    summary_csv_path = os.path.join(results_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"تم حفظ ملخص المقاييس في: {summary_csv_path}")
    
    return folds_df

# --- التنفيذ الرئيسي ---
def main():
    """الدالة الرئيسية لتنفيذ التجربة"""
    print("بدء تجربة الانحدار اللوجستي مع اختيار السمات GMO باستخدام 5-Fold Cross-Validation...")
    
    # إنشاء مجلد النتائج مع طابع زمني
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{RESULTS_DIR}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"تم إنشاء مجلد النتائج: {results_dir}")
    
    # تحميل البيانات
    X, y = load_data(FEATURE_FILE, LABEL_FILE)
    
    # إعداد التقسيم المتعدد باستخدام StratifiedKFold
    print(f"إعداد {NUM_FOLDS}-Fold Cross-Validation مع StratifiedKFold...")
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=BASE_RANDOM_STATE)
    
    # تنفيذ التجربة على كل طية
    all_folds_results = []
    
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        # اشتقاق حالة عشوائية مختلفة لكل طية
        fold_random_state = BASE_RANDOM_STATE + fold_idx
        
        # تقسيم البيانات لهذه الطية
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print(f"\nتقسيم البيانات للطية {fold_idx}: "
              f"التدريب: {X_train.shape[0]} عينة, "
              f"الاختبار: {X_test.shape[0]} عينة")
        
        # تدريب وتقييم النموذج على هذه الطية
        fold_results = train_evaluate_fold(
            X_train, y_train, X_test, y_test, 
            fold_idx, fold_random_state, results_dir
        )
        
        all_folds_results.append(fold_results)
    
    # حساب ملخص المقاييس
    print("\n--- حساب ملخص المقاييس ---")
    metrics_summary = calculate_metrics_summary(all_folds_results)
    
    # عرض ملخص النتائج
    print("\n--- ملخص النتائج ---")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        print(f"{metric}: {metrics_summary[f'mean_{metric}']:.4f} ± {metrics_summary[f'std_{metric}']:.4f}")
    
    # حفظ النتائج في ملفات CSV
    metrics_df = save_results_to_csv(all_folds_results, metrics_summary, results_dir)
    
    # رسم وحفظ الرسومات الشاملة
    print("\n--- إنشاء الرسومات الشاملة ---")
    plot_boxplots(metrics_df, results_dir)
    plot_bar_chart_with_error_bars(metrics_summary, results_dir)
    
    print("\nانتهت تجربة الانحدار اللوجستي مع اختيار السمات GMO باستخدام 5-Fold Cross-Validation.")
    print(f"تم حفظ جميع النتائج والرسومات في: {results_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"تم تغيير دليل العمل إلى: {script_dir}")
    main()
