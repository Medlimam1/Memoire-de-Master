"""
Geometric Mean Optimizer (GMO) Algorithm

المرجع الأساسي:
Rezaei, F., Safavi, H.R., Abd Elaziz, M. et al. 
GMO: geometric mean optimizer for solving engineering problems. 
Soft Comput (2023). https://doi.org/10.1007/s00500-023-08202-z
"""

import numpy as np
import time
import math

def GMO(objective_func, lb, ub, dim, SearchAgents_no, Max_iter, epsilon=1e-10):
    """
    تنفيذ خوارزمية Geometric Mean Optimizer (GMO)
    
    المعلمات:
        objective_func: دالة الهدف المراد تحسينها
        lb: الحد الأدنى لنطاق البحث
        ub: الحد الأعلى لنطاق البحث
        dim: عدد الأبعاد
        SearchAgents_no: عدد عوامل البحث (حجم المجتمع)
        Max_iter: العدد الأقصى للتكرارات
        epsilon: قيمة صغيرة لتجنب القسمة على صفر، الافتراضي = 1e-10
        
    المخرجات:
        best_score: أفضل قيمة تم العثور عليها
        best_pos: أفضل موقع تم العثور عليه
        convergence_curve: منحنى التقارب
    """
    
    # تهيئة المتغيرات
    pp_pbest = np.zeros((SearchAgents_no, dim))  # أفضل المواقع السابقة
    pp_kbest = np.zeros((SearchAgents_no, dim))  # أفضل المواقع النخبة
    pp_guide = np.zeros((SearchAgents_no, dim))  # مواقع التوجيه
    mutant = np.zeros(dim)  # موقع الطفرة
    stdev2 = np.zeros(dim)  # الانحراف المعياري للمواقع
    index = np.zeros(SearchAgents_no, dtype=int)  # مؤشرات الترتيب
    fit = np.zeros((Max_iter, SearchAgents_no))  # قيم اللياقة
    DFI = np.zeros(SearchAgents_no)  # مؤشرات اللياقة المزدوجة
    optimal_pos = np.zeros((Max_iter, dim))  # أفضل المواقع
    z_pbest = np.zeros(SearchAgents_no)  # أفضل القيم السابقة
    z_kbest = np.zeros(SearchAgents_no)  # أفضل القيم النخبة
    z_optimal = np.ones(Max_iter) * float('inf')  # أفضل القيم المثلى
    pos_final = np.zeros(dim)  # الموقع النهائي
    z_iter = np.zeros(Max_iter)  # منحنى التقارب
    
    # معاملات النخبة
    kbest_max = SearchAgents_no
    kbest_min = 2
    
    # تهيئة حدود السرعة
    velmax = 0.1 * (ub - lb)
    velmin = -velmax
    
    # وقت البدء
    start_time = time.time()
    
    print(f"GMO يعمل الآن على دالة {objective_func.__name__}")
    
    # بدء عملية التحسين
    it = 0
    
    # عملية التهيئة للخوارزمية
    # تهيئة المواقع والسرعات
    pp = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    pv = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (velmax - velmin) + velmin
    
    # تقييم دالة الهدف وتحديد أفضل الحلول والأهداف حتى الآن
    for j in range(SearchAgents_no):
        z = objective_func(pp[j, :])
        z_pbest[j] = z
        pp_pbest[j, :] = pp[j, :].copy()
    
    # حساب الانحراف المعياري للمواقع
    for i in range(dim):
        stdev2[i] = np.std(pp_pbest[:, i])
    max_stdev2 = np.max(stdev2)
    
    # حساب المتوسط والانحراف المعياري لقيم الهدف
    ave = np.mean(z_pbest)
    stdev = np.std(z_pbest)
    
    # تحديد عدد الحلول النخبة
    kbest = kbest_max - (kbest_max - kbest_min) * (it / Max_iter)
    n_best = round(kbest)
    
    # تقييم مؤشرات اللياقة المزدوجة للحلول - المعادلة (3)
    for j in range(SearchAgents_no):
        index[j] = j
    
    for j in range(SearchAgents_no):
        prod = 1
        for jj in range(SearchAgents_no):
            if jj != j:
                prod *= 1 / (1 + math.exp((-4) / (stdev * math.sqrt(math.e)) * (z_pbest[jj] - ave)))
        fit[it, j] = prod
    
    # ترتيب الحلول حسب مؤشرات اللياقة
    for j in range(SearchAgents_no - 1):
        for jj in range(j + 1, SearchAgents_no):
            if fit[it, jj] > fit[it, j]:
                c1 = fit[it, j]
                fit[it, j] = fit[it, jj]
                fit[it, jj] = c1
                c2 = index[j]
                index[j] = index[jj]
                index[jj] = c2
    
    # تعيين الحلول النخبة
    sum1 = 0
    for j in range(n_best):
        z_kbest[j] = z_pbest[index[j]]
        pp_kbest[j, :] = pp_pbest[index[j], :].copy()
        sum1 += fit[it, j]
    
    # حساب حلول التوجيه
    for j in range(SearchAgents_no):
        pp_guide[j, :] = np.zeros(dim)
        for jj in range(n_best):
            if index[jj] != j:
                DFI[jj] = fit[it, jj] / (sum1 + epsilon)
                pp_guide[j, :] += DFI[jj] * pp_kbest[jj, :]  # المعادلة (5)
    
    # تحديد أفضل قيمة هدف وحل
    for j in range(SearchAgents_no):
        if z_pbest[j] < z_optimal[it]:
            z_optimal[it] = z_pbest[j]
            optimal_pos[it, :] = pp_pbest[j, :].copy()
    
    # حفظ أفضل قيمة هدف حتى الآن في التشغيل الحالي
    z_iter[it] = z_optimal[it]
    
    # الحلقة الرئيسية
    while it < Max_iter - 1:
        it += 1
        w = 1 - (it / Max_iter)  # المعادلة (9)
        
        for j in range(SearchAgents_no):
            # طفرة حلول التوجيه - المعادلة (6)
            mutant = pp_guide[j, :] + w * np.random.normal(0, 1, dim) * (max_stdev2 - stdev2)
            
            # تحديث سرعة الحلول - المعادلة (7)
            pv[j, :] = w * pv[j, :] + (1 + (2 * np.random.random(dim) - 1) * w) * (mutant - pp[j, :])
            
            # إعادة السرعة إلى الحدود إذا تجاوزت
            flag4lbv = pv[j, :] < velmin
            flag4ubv = pv[j, :] > velmax
            pv[j, :] = pv[j, :] * (~(flag4lbv | flag4ubv)) + velmin * flag4lbv + velmax * flag4ubv
            
            # تحديث موقع الحلول - المعادلة (8)
            pp[j, :] = pp[j, :] + pv[j, :]
            
            # إعادة الموقع والسرعة إلى الحدود إذا تجاوزت
            flag4lbp = pp[j, :] < lb
            flag4ubp = pp[j, :] > ub
            pp[j, :] = pp[j, :] * (~(flag4lbp | flag4ubp)) + lb * flag4lbp + ub * flag4ubp
            pv[j, :] = pv[j, :] * (1 - 2 * (flag4lbp | flag4ubp))
            
            # تقييم دالة الهدف وتحديد أفضل الحلول والأهداف حتى الآن
            z = objective_func(pp[j, :])
            if z < z_pbest[j]:
                z_pbest[j] = z
                pp_pbest[j, :] = pp[j, :].copy()
        
        # حساب الانحراف المعياري للمواقع
        for i in range(dim):
            stdev2[i] = np.std(pp_pbest[:, i])
        max_stdev2 = np.max(stdev2)
        
        # حساب المتوسط والانحراف المعياري لقيم الهدف
        ave = np.mean(z_pbest)
        stdev = np.std(z_pbest)
        
        # تحديد عدد الحلول النخبة
        kbest = kbest_max - (kbest_max - kbest_min) * (it / Max_iter)
        n_best = round(kbest)
        
        # تقييم مؤشرات اللياقة المزدوجة للحلول - المعادلة (3)
        for j in range(SearchAgents_no):
            index[j] = j
        
        for j in range(SearchAgents_no):
            prod = 1
            for jj in range(SearchAgents_no):
                if jj != j:
                    prod *= 1 / (1 + math.exp((-4) / (stdev * math.sqrt(math.e)) * (z_pbest[jj] - ave)))
            fit[it, j] = prod
        
        # ترتيب الحلول حسب مؤشرات اللياقة
        for j in range(SearchAgents_no - 1):
            for jj in range(j + 1, SearchAgents_no):
                if fit[it, jj] > fit[it, j]:
                    c1 = fit[it, j]
                    fit[it, j] = fit[it, jj]
                    fit[it, jj] = c1
                    c2 = index[j]
                    index[j] = index[jj]
                    index[jj] = c2
        
        # تعيين الحلول النخبة
        sum1 = 0
        for j in range(n_best):
            z_kbest[j] = z_pbest[index[j]]
            pp_kbest[j, :] = pp_pbest[index[j], :].copy()
            sum1 += fit[it, j]
        
        # حساب حلول التوجيه
        for j in range(SearchAgents_no):
            pp_guide[j, :] = np.zeros(dim)
            for jj in range(n_best):
                if index[jj] != j:
                    DFI[jj] = fit[it, jj] / (sum1 + epsilon)
                    pp_guide[j, :] += DFI[jj] * pp_kbest[jj, :]  # المعادلة (5)
        
        # تحديد أفضل قيمة هدف وحل حتى الآن
        z_optimal[it] = z_optimal[it - 1]  # نسخ القيمة السابقة كقيمة افتراضية
        optimal_pos[it, :] = optimal_pos[it - 1, :].copy()
        
        for j in range(SearchAgents_no):
            if z_pbest[j] < z_optimal[it]:
                z_optimal[it] = z_pbest[j]
                optimal_pos[it, :] = pp_pbest[j, :].copy()
        
        # حفظ أفضل قيمة هدف حتى الآن في التشغيل الحالي
        z_iter[it] = z_optimal[it]
        
        if it % 50 == 0:
            print(f"التكرار {it}: أفضل قيمة = {z_optimal[it]}")
    
    # حفظ الحل النهائي الأفضل والهدف المكشوف عند انتهاء عملية التحسين
    z_final = z_iter[Max_iter - 1]
    pos_final = optimal_pos[Max_iter - 1, :].copy()
    
    # وقت الانتهاء
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"انتهى التنفيذ في {execution_time:.2f} ثانية")
    print(f"أفضل قيمة: {z_final}")
    
    return z_final, pos_final, z_iter
