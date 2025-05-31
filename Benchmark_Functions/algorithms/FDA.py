"""
Flow Direction Algorithm (FDA)

المرجع الأساسي:
Hojat Karami, Mahdi Valikhan Anaraki, Saeed Farzin, Seyedali Mirjalili,
Flow Direction Algorithm (FDA): A Novel Optimization Approach for Solving Optimization Problems,
Computers & Industrial Engineering, Volume 156, 2021, 107224, ISSN 0360-8352,
DOI: https://doi.org/10.1016/j.cie.2021.107250
"""

import numpy as np
import time

def FDA(objective_func, lb, ub, dim, SearchAgents_no, Max_iter, beta=5, seed=None):
    """
    تنفيذ خوارزمية Flow Direction Algorithm (FDA)
    
    المعلمات:
        objective_func: دالة الهدف المراد تحسينها
        lb: الحد الأدنى لنطاق البحث
        ub: الحد الأعلى لنطاق البحث
        dim: عدد الأبعاد
        SearchAgents_no: عدد عوامل البحث (حجم المجتمع)
        Max_iter: العدد الأقصى للتكرارات
        beta: حجم الجوار (عدد الجيران)، الافتراضي = 5
        seed: بذرة لمولد الأرقام العشوائية، الافتراضي = None
        
    المخرجات:
        best_score: أفضل قيمة تم العثور عليها
        best_pos: أفضل موقع تم العثور عليه
        convergence_curve: منحنى التقارب
    """
    
    # تهيئة مولد الأرقام العشوائية
    prng = np.random.default_rng(seed)
    
    # دالة مساعدة لحساب المسافة الإقليدية بين متجهين
    def l2(x):
        return np.sum(x**2)**0.5
    
    # دالة مساعدة لتوليد حل عشوائي
    def get_random_flow():
        return lb + prng.uniform(0, 1, size=dim) * (ub - lb)
    
    # تهيئة مجموعة التدفقات الأولية
    Flow_X = [get_random_flow() for _ in range(SearchAgents_no)]
    
    # تهيئة منحنى التقارب
    convergence_curve = np.zeros(Max_iter)
    
    # وقت البدء
    start_time = time.time()
    
    print(f"FDA يعمل الآن على دالة {objective_func.__name__}")
    
    # الحلقة الرئيسية
    for t in range(Max_iter):
        # مجموعة جديدة من التدفقات
        Flow_newX = []
        
        # حساب اللياقة لجميع التدفقات
        Flow_fitness = np.array([objective_func(x) for x in Flow_X])
        
        # أفضل تدفق من التدفقات الحالية
        best_flow_at = np.argmin(Flow_fitness)
        best_flow = Flow_X[best_flow_at].copy()
        best_flow_cost = Flow_fitness[best_flow_at]
        
        # حساب معامل W المطلوب لحساب دلتا لكل تدفق
        rand_bar = prng.uniform(0, 1, size=dim)
        randn = prng.normal(0, 1)
        iter_ratio = t / Max_iter
        W = (1 - iter_ratio) ** (2 * randn) * ((iter_ratio * rand_bar) * rand_bar)
        
        # تحديث كل تدفق
        for i, flow_i in enumerate(Flow_X):
            rand = prng.uniform(0, 1)
            x_rand = get_random_flow()
            delta = rand * (x_rand - flow_i) * l2(best_flow - flow_i) * W
            
            # إنشاء جيران بيتا
            Neighbour_X = [flow_i + prng.normal(0, 1) * delta for _ in range(beta)]
            
            # حساب لياقة الجيران
            Neighbour_fitness = np.array([objective_func(n) for n in Neighbour_X])
            best_neighbour_at = np.argmin(Neighbour_fitness)
            best_neighbour = Neighbour_X[best_neighbour_at]
            best_neighbour_cost = Neighbour_fitness[best_neighbour_at]
            
            # حساب الانحدارات
            s0 = []
            for j in range(beta):
                num = Flow_fitness[i] - Neighbour_fitness[j]
                s0.append(num / np.abs(flow_i - Neighbour_X[j]))
            
            # تحديث التدفق
            if best_neighbour_cost < Flow_fitness[i]:
                V = prng.normal(0, 1) * s0[best_neighbour_at]
                new_flow_i = flow_i + V * ((flow_i - best_neighbour) / l2(flow_i - best_neighbour))
            else:
                V = np.zeros(dim) + (1 / dim**0.5)
                r = i
                while r == i:
                    r = prng.integers(0, len(Flow_X))
                
                if Flow_fitness[r] < Flow_fitness[i]:
                    randn_bar = prng.normal(0, 1, size=dim)
                    new_flow_i = flow_i + randn_bar * (Flow_X[r] - flow_i)
                else:
                    rand_n = prng.uniform(0, 1)
                    new_flow_i = flow_i + 2 * rand_n * (best_flow - flow_i)
            
            # التحقق من الحدود
            new_flow_i = np.clip(new_flow_i, lb, ub)
            Flow_newX.append(new_flow_i)
        
        # تقييم التدفقات الجديدة
        Flow_newfitness = np.array([objective_func(x) for x in Flow_newX])
        
        # تحديث التدفقات إذا كانت أفضل
        for i in range(SearchAgents_no):
            if Flow_newfitness[i] < Flow_fitness[i]:
                Flow_X[i] = Flow_newX[i].copy()
        
        # تحديث منحنى التقارب
        Flow_fitness = np.array([objective_func(x) for x in Flow_X])
        best_flow_at = np.argmin(Flow_fitness)
        best_flow = Flow_X[best_flow_at].copy()
        best_flow_cost = Flow_fitness[best_flow_at]
        convergence_curve[t] = best_flow_cost
        
        if t % 50 == 0:
            print(f"التكرار {t}: أفضل قيمة = {best_flow_cost}")
    
    # وقت الانتهاء
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"انتهى التنفيذ في {execution_time:.2f} ثانية")
    print(f"أفضل قيمة: {best_flow_cost}")
    
    return best_flow_cost, best_flow, convergence_curve
