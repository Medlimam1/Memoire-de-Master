"""
Aquila Optimization (AO) Algorithm

المرجع الأساسي:
Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A.A., Al-Qaness, M.A. and Gandomi, A.H., 2021. 
Aquila optimizer: a novel meta-heuristic optimization algorithm. 
Computers & Industrial Engineering, 157, p.107250.
DOI: https://doi.org/10.1016/j.cie.2021.107250
"""

import numpy as np
import time
import math

def AO(objective_func, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    تنفيذ خوارزمية Aquila Optimization (AO)
    
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
    
    # وقت البدء
    start_time = time.time()
    
    print(f"AO يعمل الآن على دالة {objective_func.__name__}")
    
    # دالة مساعدة لحساب خطوة levy flight
    def get_levy_flight_step(beta=1.5, multiplier=1.0, case=-1):
        # sigma = (gamma(1+beta) * sin(pi*beta/2) / (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta)
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = multiplier * u / (abs(v) ** (1 / beta))
        
        if case == 1:
            step = 0.01 * step
        return step
    
    # الحلقة الرئيسية
    for t in range(Max_iter):
        # معاملات الخوارزمية
        alpha = delta = 0.1
        g1 = 2 * np.random.random() - 1  # المعادلة 16
        g2 = 2 * (1 - t / Max_iter)  # المعادلة 17
        
        # معاملات إضافية للخوارزمية
        dim_list = np.array(list(range(1, dim + 1)))
        miu = 0.00565
        r0 = 10
        r = r0 + miu * dim_list
        w = 0.005
        phi0 = 3 * np.pi / 2
        phi = -w * dim_list + phi0
        x = r * np.sin(phi)  # المعادلة 9
        y = r * np.cos(phi)  # المعادلة 10
        
        # دالة الجودة (المعادلة 15) - تعديل لتجنب قسمة على الصفر
        if t == 0:
            QF = 0  # قيمة افتراضية عندما t=0
        else:
            # تجنب رفع الصفر إلى قوة سالبة
            denominator = (1 - Max_iter) ** 2
            if denominator == 0:  # لتجنب القسمة على الصفر
                denominator = 1e-10
            QF = t ** ((2 * np.random.random() - 1) / denominator)
        
        # تقييم الحلول الحالية وتحديث أفضل حل
        for i in range(SearchAgents_no):
            fitness = objective_func(X[i, :])
            if fitness < best_score:
                best_score = fitness
                best_pos = X[i, :].copy()
        
        # حساب متوسط المواقع
        x_mean = np.mean(X, axis=0)
        
        # تحديث مواقع العوامل
        for i in range(SearchAgents_no):
            levy_step = get_levy_flight_step(beta=1.5, multiplier=1.0, case=-1)
            
            if t <= (2 / 3) * Max_iter:  # المعادلات 3، 4
                if np.random.random() < 0.5:
                    X[i, :] = best_pos * (1 - t / Max_iter) + np.random.random() * (x_mean - best_pos)
                else:
                    j = np.random.choice([idx for idx in range(SearchAgents_no) if idx != i])
                    X[i, :] = best_pos * levy_step + X[j, :] + np.random.random() * (y - x)  # المعادلة 5
            else:
                if np.random.random() < 0.5:
                    # المعادلة 13
                    X[i, :] = alpha * (best_pos - x_mean) - np.random.random() * \
                             (np.random.random() * (ub - lb) + lb) * delta
                else:
                    # المعادلة 14
                    X[i, :] = QF * best_pos - (g2 * X[i, :] * np.random.random()) - \
                             g2 * levy_step + np.random.random() * g1
            
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
