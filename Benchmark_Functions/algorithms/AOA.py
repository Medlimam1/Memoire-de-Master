"""
Arithmetic Optimization Algorithm (AOA)

المرجع الأساسي:
Abualigah, L., Diabat, A., Mirjalili, S., Abd Elaziz, M. and Gandomi, A.H., 2021. 
The arithmetic optimization algorithm. Computer methods in applied mechanics and engineering, 376, p.113609.
DOI: https://doi.org/10.1016/j.cma.2020.113609
"""

import numpy as np
import time

def AOA(objective_func, lb, ub, dim, SearchAgents_no, Max_iter, alpha=5, miu=0.5, moa_min=0.2, moa_max=0.9):
    """
    تنفيذ خوارزمية Arithmetic Optimization Algorithm (AOA)
    
    المعلمات:
        objective_func: دالة الهدف المراد تحسينها
        lb: الحد الأدنى لنطاق البحث
        ub: الحد الأعلى لنطاق البحث
        dim: عدد الأبعاد
        SearchAgents_no: عدد عوامل البحث (حجم المجتمع)
        Max_iter: العدد الأقصى للتكرارات
        alpha: معامل ثابت، معامل استغلال حساس، الافتراضي: 5
        miu: معامل ثابت، معامل تحكم لضبط عملية البحث، الافتراضي: 0.5
        moa_min: الحد الأدنى لمعامل Math Optimizer Accelerated، الافتراضي: 0.2
        moa_max: الحد الأعلى لمعامل Math Optimizer Accelerated، الافتراضي: 0.9
        
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
    
    # وقت البدء
    start_time = time.time()
    
    print(f"AOA يعمل الآن على دالة {objective_func.__name__}")
    
    # الحلقة الرئيسية
    for t in range(Max_iter):
        # تحديث معاملات الخوارزمية
        moa = moa_min + t * ((moa_max - moa_min) / Max_iter)  # المعادلة 2
        mop = 1 - ((t+1) ** (1.0 / alpha)) / (Max_iter ** (1.0 / alpha))  # المعادلة 4
        
        for i in range(SearchAgents_no):
            # تقييم الحلول الحالية
            fitness = objective_func(X[i, :])
            
            # تحديث أفضل حل
            if fitness < best_score:
                best_score = fitness
                best_pos = X[i, :].copy()
            
            # تحديث موقع كل عامل
            for j in range(dim):
                r1, r2, r3 = np.random.random(3)
                
                # حساب المعاملات للتحديث - استخدام lb[j] و ub[j] للتعامل مع المصفوفات
                range_val = (ub[j] - lb[j]) * miu + lb[j]
                
                if r1 > moa:  # مرحلة الاستكشاف
                    if r2 < 0.5:
                        X[i, j] = best_pos[j] / (mop + 1e-10) * range_val
                    else:
                        X[i, j] = best_pos[j] * mop * range_val
                else:  # مرحلة الاستغلال
                    if r3 < 0.5:
                        X[i, j] = best_pos[j] - mop * range_val
                    else:
                        X[i, j] = best_pos[j] + mop * range_val
            
            # التحقق من الحدود
            X[i, :] = np.clip(X[i, :], lb, ub)
        
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
