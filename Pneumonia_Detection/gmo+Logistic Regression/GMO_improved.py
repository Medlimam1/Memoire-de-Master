import numpy as np
import time
import matplotlib.pyplot as plt
import os
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


# استيراد الإصدار المحسن من أدوات اختيار الميزات
from feature_selection_utils_improved import (
    Data,
    compute_fitness,
    validate_FS,
    initialize_agents,
    sort_agents,
    Solution,
    plot_pr_curve
)

def GMO(num_agents: int, max_iter: int, train_data, train_label, clf_model, obj_function=compute_fitness, save_conv_graph: bool = False, save_dir: str = "/content/drive/MyDrive/GMO_Results", fold: int = 1) -> Solution:

    """
    نسخة محسنة من خوارزمية Geometric Mean Optimizer مع:
    - سلوك إيثاري دوري
    - طفرة متكيفة مع الزمن
    - حفظ منحنى التقارب
    - حفظ أفضل العوامل
    - دعم الانحدار اللوجستي كدالة لياقة افتراضية
    - دعم اللغة العربية في الرسومات
    """

    X = np.array(train_data)
    y = np.array(train_label)
    num_features = X.shape[1]

    # إعداد مجلد الحفظ
    os.makedirs(save_dir, exist_ok=True)

    data = Data()
    data.train_X = X
    data.train_Y = y
    data.val_X = X
    data.val_Y = y

    agents = initialize_agents(num_agents, num_features)
    convergence_curve = {'fitness': np.zeros(max_iter)}
    best_agents_record = []

    agents, fitness, _ = sort_agents(agents, (obj_function, clf_model, 1.0), data)
    leader_agent = agents[0].copy()
    leader_fitness = fitness[0]

    start_time = time.time()

    for iteration in range(max_iter):
        print(f"\n--- التكرار {iteration + 1}/{max_iter} ---")

        # طفرة متكيفة: تقل مع الوقت
        mutation_prob = max(0.1, 0.3 * (1 - iteration / max_iter))

        for idx in range(num_agents):
            i, j = np.random.choice(num_agents, 2, replace=False)
            a = agents[i]
            b = agents[j]

            gm = np.sqrt(a * b)
            mask = np.random.rand(num_features) < 0.5
            candidate = np.where(mask, gm, agents[idx])

            mutation = np.random.rand(num_features) < mutation_prob
            candidate = np.where(mutation, 1 - candidate, candidate)
            candidate = np.clip(candidate, 0, 1)
            agents[idx] = (candidate > 0.5).astype(int)

        # سلوك إيثاري كل 10 تكرارات: الأفضل يساعد الأضعف
        if iteration % 10 == 0:
            agents, fitness, _ = sort_agents(agents, (obj_function, clf_model, 1.0), data)
            weakest = agents[-1]
            altruistic_mask = np.random.rand(num_features) < 0.3
            agents[-1] = np.where(altruistic_mask, leader_agent, weakest)

        agents, fitness, _ = sort_agents(agents, (obj_function, clf_model, 1.0), data)
        if fitness[0] > leader_fitness:
            leader_agent = agents[0].copy()
            leader_fitness = fitness[0]

        best_agents_record.append(leader_agent.copy())
        convergence_curve['fitness'][iteration] = leader_fitness
        print(f"  أفضل لياقة حتى الآن: {leader_fitness:.6f}")

    _, _, leader_accuracy = sort_agents(
        leader_agent[np.newaxis, :], (obj_function, clf_model, 1.0), data)

    exec_time = time.time() - start_time

    # حفظ منحنى التقارب
    plt.figure()
    plt.plot(convergence_curve['fitness'], marker='o')
    plt.title(get_display(arabic_reshaper.reshape(f'منحنى تقارب GMO - الطية {fold}')))
    plt.xlabel(get_display(arabic_reshaper.reshape('التكرار')))
    plt.ylabel(get_display(arabic_reshaper.reshape('أفضل قيمة للياقة')))
    plt.grid(True)
    if save_conv_graph:
        plt.savefig(f"{save_dir}/convergence_GMO_fold{fold}.png")
    plt.close()

    # تحليل الميزات المختارة عبر التكرارات
    best_agents_array = np.array(best_agents_record)
    selection_frequency = np.sum(best_agents_array, axis=0) / max_iter
    np.save(os.path.join(save_dir, "feature_selection_frequency.npy"), selection_frequency)

    solution = Solution()
    solution.best_agent = leader_agent
    solution.best_fitness = leader_fitness
    solution.best_accuracy = leader_accuracy[0]
    solution.convergence_curve = convergence_curve
    solution.execution_time = exec_time

    # استخدام نموذج الانحدار اللوجستي للتحقق إذا لم يتم تحديد نموذج آخر
    class_names = ['طبيعي', 'التهاب رئوي']
    validate_FS(data.val_X, data.val_Y, solution.best_agent, clf_model,
                save_path=os.path.join(save_dir, "gmo_confusion_matrix.png"),
                class_names=class_names)

    return solution
