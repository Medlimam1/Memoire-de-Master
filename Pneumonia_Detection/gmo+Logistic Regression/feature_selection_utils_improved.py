import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, classification_report,
    ConfusionMatrixDisplay, f1_score, confusion_matrix,
    precision_recall_curve, average_precision_score,
    recall_score
)
from sklearn.linear_model import LogisticRegression # استخدام الانحدار اللوجستي كنموذج افتراضي للياقة
import arabic_reshaper
from bidi.algorithm import get_display # لدعم اللغة العربية في الرسومات
from matplotlib import font_manager, rcParams
# أضف مسار الخط اللي نزّلته
font_manager.fontManager.addfont(
    "C:/Users/ADMIN/Desktop/gmo lr/الخط/Amiri-Regular.ttf"
)

# حدّد العائلة الافتراضية
rcParams['font.family'] = "Amiri"
# منع عرض علامة الطرح (−) بشكل غريب
rcParams['axes.unicode_minus'] = False

# تجاهل التحذيرات
import warnings
warnings.filterwarnings("ignore")

# حاوية الحل
class Solution:
    def __init__(self):
        self.num_features = None
        self.num_agents = None
        self.max_iter = None
        self.obj_function = None
        self.execution_time = None
        self.convergence_curve = {}
        self.best_agent = None
        self.best_fitness = None
        self.best_accuracy = None
        self.final_population = None
        self.final_fitness = None
        self.final_accuracy = None


# حاوية هيكل البيانات
class Data:
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.val_X = None # مجموعة التحقق للياقة، يمكن أن تكون نفس مجموعة التدريب للبساطة هنا
        self.val_Y = None


def initialize_agents(num_agents, num_features):
    """تهيئة العملاء مع اختيار عشوائي للميزات"""
    min_features = int(0.3 * num_features)  # محاولة اختيار بين 30٪ و 60٪ من الميزات في البداية
    max_features = int(0.6 * num_features)

    agents = np.zeros((num_agents, num_features), dtype=int)

    for agent_no in range(num_agents):
        # ضمان بذرة مختلفة لكل تهيئة عميل إذا كان time.time() خشنًا جدًا
        random.seed(int(time.time() * 1000) + agent_no)
        num = random.randint(min_features, max_features if max_features > min_features else min_features +1) # ضمان أن max > min
        if num_features > 0:
            pos = random.sample(range(num_features), num)
            agents[agent_no, pos] = 1
        else:
            # معالجة حالة وجود 0 ميزات، على الرغم من أنها غير محتملة لهذه المشكلة
            pass 

    return agents


def compute_accuracy(agent, train_X, test_X, train_Y, test_Y, clf_model):
    """حساب دقة النموذج باستخدام الميزات المحددة"""
    cols = np.flatnonzero(agent)
    if cols.size == 0:
        return 0.0  # لم يتم اختيار أي ميزات، الدقة هي 0

    # التأكد من أن clf_model هو نسخة جديدة أو تم إعادة تعيينها بشكل صحيح إذا تم تمريرها وتناسبها عدة مرات
    # بالنسبة لـ scikit-learn، فإن تناسب نسخة جديدة في كل مرة أكثر أمانًا ما لم تتم إدارة الحالة بعناية.
    # هنا، نفترض أن clf_model هو نسخة جديدة غير مناسبة أو نوع يجب إنشاؤه.
    
    # إذا كان clf_model نوعًا (مثل LogisticRegression)، قم بإنشائه.
    # إذا كانت نسخة، قم باستنساخها للتأكد من أنها جديدة.
    from sklearn.base import clone
    current_clf = clone(clf_model)

    current_clf.fit(train_X[:, cols], train_Y)
    acc = current_clf.score(test_X[:, cols], test_Y)
    return acc


def compute_fitness(agent, train_X, test_X, train_Y, test_Y, clf_model, weight_acc=0.99, dims=None):
    """حساب اللياقة باستخدام الدقة ونسبة الميزات المختارة"""
    # وزن أعلى للدقة، أقل لتقليل الميزات، وفقًا لأهداف FS النموذجية
    weight_feat = 1 - weight_acc
    num_features = dims if dims else agent.size

    acc = compute_accuracy(agent, train_X, test_X, train_Y, test_Y, clf_model)
    
    selected_features_count = np.sum(agent)
    if num_features == 0: # تجنب القسمة على صفر إذا لم تكن هناك ميزات
        feat_ratio = 0
    else:
        feat_ratio = (num_features - selected_features_count) / num_features
    
    fitness = weight_acc * acc + weight_feat * feat_ratio

    return fitness, acc


def sort_agents(agents, obj_params, data):
    """ترتيب العملاء حسب قيمة اللياقة"""
    train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
    
    # استخدام الانحدار اللوجستي كنموذج افتراضي إذا لم يتم تحديده
    if len(obj_params) == 1:
        obj_function = obj_params[0]
        clf_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=200)
        weight_acc = 0.99
    elif len(obj_params) == 2:
        obj_function, clf_model = obj_params
        weight_acc = 0.99
    else:
        obj_function, clf_model, weight_acc = obj_params
    
    # obj_function هي compute_fitness

    if agents.ndim == 1: # عميل واحد
        fitness, acc = obj_function(agents, train_X, val_X, train_Y, val_Y, clf_model, weight_acc)
        return agents, fitness, acc

    num_agents = agents.shape[0]
    fitness = np.zeros(num_agents)
    acc = np.zeros(num_agents)
    for idx, agent in enumerate(agents):
        fitness[idx], acc[idx] = obj_function(agent, train_X, val_X, train_Y, val_Y, clf_model, weight_acc)

    sort_idx = np.argsort(-fitness) # ترتيب تنازلي للياقة
    return agents[sort_idx], fitness[sort_idx], acc[sort_idx]


def plot_pr_curve(y_true, y_prob, classifier_name, fold_name, save_path=None):
    """رسم منحنى الدقة-الاستدعاء (PR Curve)"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=get_display(arabic_reshaper.reshape(f'منحنى PR (متوسط الدقة = {avg_precision:.2f})')))
    plt.axhline(y=sum(y_true)/len(y_true), color='navy', linestyle='--', 
                label=get_display(arabic_reshaper.reshape('خط الأساس')))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(get_display(arabic_reshaper.reshape('الاستدعاء (Recall)')))
    plt.ylabel(get_display(arabic_reshaper.reshape('الدقة (Precision)')))
    plt.title(get_display(arabic_reshaper.reshape(f'منحنى الدقة-الاستدعاء للمصنف {classifier_name} - {fold_name}')))
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"تم حفظ منحنى PR في: {save_path}")
    plt.close()


def validate_FS(X_val, y_val, agent, clf_model_for_validation=None, show_plot=True, save_path=None, class_names=None):
    """التحقق من صحة اختيار الميزات وعرض النتائج"""
    cols = np.flatnonzero(agent)
    if cols.size == 0:
        print("لا يوجد ميزات محددة للتحقق.")
        return 0, None, None, None, None # إرجاع None للمقاييس إذا لم يتم اختيار ميزات

    X1 = X_val[:, cols]
    
    # استخدام الانحدار اللوجستي كنموذج افتراضي إذا لم يتم تحديده
    if clf_model_for_validation is None:
        clf_model_for_validation = LogisticRegression(random_state=42, solver='liblinear', max_iter=200)
    
    from sklearn.base import clone
    model = clone(clf_model_for_validation) # استخدام نموذج المصنف المقدم
    model.fit(X1, y_val) # يجب أن يتم تناسبه على بيانات التدريب، والتحقق على بيانات الاختبار/التحقق
    
    y_pred = model.predict(X1)
    
    # حساب الاحتمالات للرسومات
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X1)[:, 1]
    else:
        y_prob = model.decision_function(X1)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
    # حساب المقاييس
    accuracy = np.mean(y_pred == y_val)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    print(f'دقة الاختيار باستخدام {type(model).__name__}: {accuracy:.6f}')
    print('-' * 50)
    # توفير أسماء الأهداف إذا كانت متاحة، وإلا قد يحذر أو يستخدم الافتراضي 0، 1
    
    # 1) إعادة تشكيل وربط أسماء الفئات العربية
    reshaped_names = [
        get_display(arabic_reshaper.reshape(name))
        for name in class_names
    ]

    # 2) تقرير التصنيف مع الأسماء المعاد تشكيلها
    report = classification_report(
        y_val,
        y_pred,
        digits=4,
        target_names=reshaped_names,
        zero_division=0
    )
    print(report)
    print('-' * 50)

    if show_plot or save_path:
        # إعداد مجلد الحفظ إذا لم يكن موجودًا
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
        
        # رسم مصفوفة الارتباك
        disp = ConfusionMatrixDisplay.from_predictions(
            y_val,
            y_pred,
            display_labels=reshaped_names,
            cmap=plt.cm.Blues
        )
        plt.title(get_display(arabic_reshaper.reshape('مصفوفة الارتباك')))
        if show_plot:
            plt.show()
        if save_path:
            disp.figure_.savefig(save_path)
            print(f"تم حفظ مصفوفة الارتباك في: {save_path}")
        plt.close(disp.figure_)

        # 4) رسم منحنى PR
        if save_path:
            pr_path = save_path.replace('.png', '_pr_curve.png')
            plot_pr_curve(y_val, y_prob, type(model).__name__, "التحقق", pr_path)

    cm = confusion_matrix(y_val, y_pred)
    if cm.size == 4:  # تصنيف ثنائي
        tn, fp, fn, tp = cm.ravel()
        print(f'الإيجابيات الحقيقية = {tp}\n'
              f'الإيجابيات الخاطئة = {fp}\n'
              f'السلبيات الخاطئة = {fn}\n'
              f'السلبيات الحقيقية = {tn}')
    else:  # متعدد الفئات
        print(f"مصفوفة الارتباك:\n{cm}")
        tn = fp = fn = tp = None

    return accuracy, precision, recall, f1, (tn, fp, fn, tp)
