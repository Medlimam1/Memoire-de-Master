"""
Equilibrium Optimizer (EO) Algorithm

المرجع الأساسي:
Faramarzi, A., Heidarinejad, M., Stephens, B. and Mirjalili, S., 2020. 
Equilibrium optimizer: A novel optimization algorithm. 
Knowledge-Based Systems, 191, p.105190.
DOI: https://doi.org/10.1016/j.knosys.2019.105190
"""

import numpy as np
import time

def EO(objective_func, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    تنفيذ خوارزمية Equilibrium Optimizer (EO)
    
    المعلمات:
        objective_func: دالة الهدف المراد تحسينها
        lb: الحد الأدنى لنطاق البحث
        ub: الحد الأعلى لنطاق البحث
        dim: عدد الأبعاد
        SearchAgents_no: عدد عوامل البحث (حجم المجتمع)
        Max_iter: العدد الأقصى للتكرارات
        
    المخرجات:
        best_score: أفضل قيمة تم العثور عليها
        best_pos: أفضل موقع تم العثور عليه
        convergence_curve: منحنى التقارب
    """
    
    # تهيئة أفضل حل
    best_pos = np.zeros(dim)
    best_score = float("inf")  # للمشاكل التي تتطلب تصغير القيمة
    
    # تهيئة مواقع العوامل
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    
    # تهيئة منحنى التقارب
    convergence_curve = np.zeros(Max_iter)
    
    # معاملات ثابتة مقترحة من المؤلفين
    V = 1
    a1 = 2
    a2 = 1
    GP = 0.5
    
    # وقت البدء
    start_time = time.time()
    
    print(f"EO يعمل الآن على دالة {objective_func.__name__}")
    
    # الحلقة الرئيسية
    for t in range(Max_iter):
        # تقييم الحلول الحالية
        fitness = np.zeros(SearchAgents_no)
        for i in range(SearchAgents_no):
            fitness[i] = objective_func(X[i, :])
            if fitness[i] < best_score:
                best_score = fitness[i]
                best_pos = X[i, :].copy()
        
        # إنشاء مجموعة التوازن (Equilibrium Pool)
        # اختيار أفضل 4 حلول
        sorted_indices = np.argsort(fitness)
        c_eq_list = [X[sorted_indices[i], :].copy() for i in range(min(4, SearchAgents_no))]
        
        # إضافة المتوسط إلى مجموعة التوازن
        c_eq_mean = np.mean(c_eq_list, axis=0)
        c_eq_list.append(c_eq_mean)
        
        # المعادلة 9: حساب معامل t
        t_factor = (1 - t / Max_iter) ** (a2 * t / Max_iter)
        
        # تحديث مواقع العوامل
        for i in range(SearchAgents_no):
            # اختيار عشوائي لمرشح من مجموعة التوازن
            c_eq = c_eq_list[np.random.randint(0, len(c_eq_list))]
            
            # المعادلة 11: حساب معامل f
            lamda = np.random.uniform(0, 1, dim)
            r = np.random.uniform(0, 1, dim)
            f = a1 * np.sign(r - 0.5) * (np.exp(-lamda * t_factor) - 1.0)
            
            # المعادلة 15: حساب معامل gcp
            r1 = np.random.random()
            r2 = np.random.random()
            gcp = 0.5 * r1 * np.ones(dim) * (r2 >= GP)
            
            # المعادلة 14: حساب g0
            g0 = gcp * (c_eq - lamda * X[i, :])
            
            # المعادلة 13: حساب g
            g = g0 * f
            
            # المعادلة 16: تحديث الموقع
            X_new = c_eq + (X[i, :] - c_eq) * f + (g * V / lamda) * (1.0 - f)
            
            # التحقق من الحدود
            X_new = np.clip(X_new, lb, ub)
            
            # تقييم الحل الجديد
            fitness_new = objective_func(X_new)
            
            # تحديث الحل إذا كان أفضل
            if fitness_new < fitness[i]:
                X[i, :] = X_new.copy()
                fitness[i] = fitness_new
                
                # تحديث أفضل حل
                if fitness_new < best_score:
                    best_score = fitness_new
                    best_pos = X_new.copy()
        
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
