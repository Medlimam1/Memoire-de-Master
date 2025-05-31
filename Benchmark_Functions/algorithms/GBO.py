"""
Gradient-Based Optimizer (GBO) Algorithm

المرجع الأساسي:
Ahmadianfar, I., Bozorg-Haddad, O. and Chu, X., 2020. 
Gradient-based optimizer: A new metaheuristic optimization algorithm. 
Information Sciences, 540, pp.131-159.
DOI: https://doi.org/10.1016/j.future.2019.02.028
"""

import numpy as np
import time

def GBO(objective_func, lb, ub, dim, SearchAgents_no, Max_iter, pr=0.5, beta_min=0.2, beta_max=1.2):
    """
    تنفيذ خوارزمية Gradient-Based Optimizer (GBO)
    
    المعلمات:
        objective_func: دالة الهدف المراد تحسينها
        lb: الحد الأدنى لنطاق البحث
        ub: الحد الأعلى لنطاق البحث
        dim: عدد الأبعاد
        SearchAgents_no: عدد عوامل البحث (حجم المجتمع)
        Max_iter: العدد الأقصى للتكرارات
        pr: معامل الاحتمالية، الافتراضي = 0.5
        beta_min: معامل ثابت، الافتراضي = 0.2
        beta_max: معامل ثابت، الافتراضي = 1.2
        
    المخرجات:
        best_score: أفضل قيمة تم العثور عليها
        best_pos: أفضل موقع تم العثور عليه
        convergence_curve: منحنى التقارب
    """
    
    # تهيئة أفضل وأسوأ حل
    best_pos = np.zeros(dim)
    best_score = float("inf")  # للمشاكل التي تتطلب تصغير القيمة
    worst_pos = np.zeros(dim)
    worst_score = float("-inf")  # للمشاكل التي تتطلب تصغير القيمة
    
    # تهيئة مواقع العوامل
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    fitness = np.zeros(SearchAgents_no)
    
    # تهيئة منحنى التقارب
    convergence_curve = np.zeros(Max_iter)
    
    # وقت البدء
    start_time = time.time()
    
    print(f"GBO يعمل الآن على دالة {objective_func.__name__}")
    
    # تقييم الحلول الأولية
    for i in range(SearchAgents_no):
        fitness[i] = objective_func(X[i, :])
        if fitness[i] < best_score:
            best_score = fitness[i]
            best_pos = X[i, :].copy()
        if fitness[i] > worst_score:
            worst_score = fitness[i]
            worst_pos = X[i, :].copy()
    
    # الحلقة الرئيسية
    for t in range(Max_iter):
        # تحديث معاملات الخوارزمية
        beta = beta_min + (beta_max - beta_min) * (1 - (t / Max_iter) ** 3) ** 2  # المعادلة 14.2
        alpha = np.abs(beta * np.sin(3 * np.pi / 2 + np.sin(beta * 3 * np.pi / 2)))  # المعادلة 14.1
        
        for i in range(SearchAgents_no):
            p1 = 2 * np.random.random() * alpha - alpha
            p2 = 2 * np.random.random() * alpha - alpha
            
            # اختيار أربعة مواقع عشوائية من المجتمع
            r_idx = np.random.choice([j for j in range(SearchAgents_no) if j != i], 4, replace=False)
            r1, r2, r3, r4 = r_idx[0], r_idx[1], r_idx[2], r_idx[3]
            
            # متوسط أربعة مواقع عشوائية من المجتمع
            r0 = (X[r1, :] + X[r2, :] + X[r3, :] + X[r4, :]) / 4
            
            # معاملات العشوائية
            epsilon = 5e-3 * np.random.random()
            delta = 2 * np.random.random() * np.abs(r0 - X[i, :])
            step = (best_pos - X[r1, :] + delta) / 2
            delta_x = np.random.choice(range(0, SearchAgents_no)) * np.abs(step)
            
            # حساب المواقع الوسيطة
            x1 = X[i, :] - np.random.normal() * p1 * 2 * delta_x * X[i, :] / (worst_pos - best_pos + epsilon) + \
                 np.random.random() * p2 * (best_pos - X[i, :])
            
            z = X[i, :] - np.random.normal() * 2 * delta_x * X[i, :] / (worst_pos - best_pos + epsilon)
            y_p = np.random.random() * ((z + X[i, :]) / 2 + np.random.random() * delta_x)
            y_q = np.random.random() * ((z + X[i, :]) / 2 - np.random.random() * delta_x)
            
            x2 = best_pos - np.random.normal() * p1 * 2 * delta_x * X[i, :] / (y_p - y_q + epsilon) + \
                 np.random.random() * p2 * (X[r1, :] - X[r2, :])
            
            x3 = X[i, :] - p1 * (x2 - x1)
            
            ra = np.random.random()
            rb = np.random.random()
            pos_new = ra * (rb * x1 + (1 - rb) * x2) + (1 - ra) * x3
            
            # مشغل الهروب المحلي
            if np.random.random() < pr:
                f1 = np.random.uniform(-1, 1)
                f2 = np.random.normal(0, 1)
                L1 = np.round(1 - np.random.random())
                u1 = L1 * 2 * np.random.random() + (1 - L1)
                u2 = L1 * np.random.random() + (1 - L1)
                u3 = L1 * np.random.random() + (1 - L1)
                
                L2 = np.round(1 - np.random.random())
                x_rand = np.random.uniform(0, 1, dim) * (ub - lb) + lb
                x_p = X[np.random.choice(range(0, SearchAgents_no)), :]
                x_m = L2 * x_p + (1 - L2) * x_rand
                
                if np.random.random() < 0.5:
                    pos_new = pos_new + f1 * (u1 * best_pos - u2 * x_m) + \
                              f2 * p1 * (u3 * (x2 - x1) + u2 * (X[r1, :] - X[r2, :])) / 2
                else:
                    pos_new = best_pos + f1 * (u1 * best_pos - u2 * x_m) + \
                              f2 * p1 * (u3 * (x2 - x1) + u2 * (X[r1, :] - X[r2, :])) / 2
            
            # التحقق من الحدود
            pos_new = np.clip(pos_new, lb, ub)
            
            # تقييم الحل الجديد
            fitness_new = objective_func(pos_new)
            
            # تحديث الحل إذا كان أفضل
            if fitness_new < fitness[i]:
                X[i, :] = pos_new.copy()
                fitness[i] = fitness_new
                
                # تحديث أفضل وأسوأ حل
                if fitness_new < best_score:
                    best_score = fitness_new
                    best_pos = pos_new.copy()
                
            # تحديث أسوأ حل
            if fitness[i] > worst_score:
                worst_score = fitness[i]
                worst_pos = X[i, :].copy()
        
        # تحديث منحنى التقارب
        convergence_curve[t] = best_score
        
        if t % 50 == 0:
            print(f"التكرار {t}: أفضل قيمة = {best_score}")
    
    # وقت الانتهاء
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"انتهى التنفيذ في {execution_time:.2f} ثانية")
    print(f"أفضل قيمة: {best_score}")
    
    return best_score, best_pos, convergence_curve
