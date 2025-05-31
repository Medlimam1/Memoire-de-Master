"""
Harris Hawks Optimization (HHO) Algorithm

المرجع الأساسي:
Harris hawks optimization: Algorithm and applications
Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
Future Generation Computer Systems, 
DOI: https://doi.org/10.1016/j.future.2019.02.028
"""

import numpy as np
import math
import time

def HHO(objective_func, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    تنفيذ خوارزمية Harris Hawks Optimization (HHO)
    
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
    
    # تهيئة موقع وطاقة الأرنب
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = float("inf")  # للمشاكل التي تتطلب تصغير القيمة
    
    # تهيئة مواقع صقور هاريس
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    
    # تهيئة منحنى التقارب
    convergence_curve = np.zeros(Max_iter)
    
    # وقت البدء
    start_time = time.time()
    
    print(f"HHO يعمل الآن على دالة {objective_func.__name__}")
    
    # الحلقة الرئيسية
    t = 0  # عداد الحلقة
    while t < Max_iter:
        for i in range(SearchAgents_no):
            # التحقق من الحدود
            X[i, :] = np.clip(X[i, :], lb, ub)
            
            # حساب اللياقة للمواقع
            fitness = objective_func(X[i, :])
            
            # تحديث موقع الأرنب
            if fitness < Rabbit_Energy:  # للمشاكل التي تتطلب تصغير القيمة
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()
        
        # معامل E1 يظهر انخفاض طاقة الأرنب
        E1 = 2 * (1 - (t / Max_iter))
        
        # تحديث مواقع صقور هاريس
        for i in range(SearchAgents_no):
            E0 = 2 * np.random.random() - 1  # -1 < E0 < 1
            Escaping_Energy = E1 * E0  # طاقة هروب الأرنب (المعادلة 3)
            
            # -------- مرحلة الاستكشاف (المعادلة 1) ---------------
            if abs(Escaping_Energy) >= 1:
                # صقور هاريس تجثم عشوائياً بناءً على استراتيجيتين
                q = np.random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * np.random.random())
                X_rand = X[rand_Hawk_index, :]
                
                if q < 0.5:
                    # الجثوم بناءً على أفراد العائلة الآخرين
                    X[i, :] = X_rand - np.random.random() * abs(X_rand - 2 * np.random.random() * X[i, :])
                else:
                    # الجثوم على شجرة عالية عشوائية (موقع عشوائي داخل نطاق المجموعة)
                    X[i, :] = (Rabbit_Location - X.mean(0)) - np.random.random() * ((ub - lb) * np.random.random() + lb)
            
            # -------- مرحلة الاستغلال ---------------
            else:
                # مهاجمة الأرنب باستخدام 4 استراتيجيات تتعلق بسلوك الأرنب
                r = np.random.random()  # احتمالية كل حدث
                
                # المرحلة 1: ----- هجوم مفاجئ (سبع قتلات) ----------
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    # حصار صعب (المعادلة 6)
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs(Rabbit_Location - X[i, :])
                
                if r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    # حصار ناعم (المعادلة 4)
                    Jump_strength = 2 * (1 - np.random.random())  # قوة قفزة عشوائية للأرنب
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                
                # المرحلة 2: -------- تنفيذ غطسات سريعة للفريق (حركات القفز) ------
                if r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    # حصار ناعم (المعادلة 10)
                    # الأرنب يحاول الهروب بحركات متعرجة خادعة
                    Jump_strength = 2 * (1 - np.random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                    
                    if objective_func(X1) < objective_func(X[i, :]):  # تحسن الحركة؟
                        X[i, :] = X1.copy()
                    else:
                        # الصقور تنفذ غطسات سريعة قصيرة بتوزيع ليفي حول الأرنب
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :]) + np.multiply(np.random.randn(dim), Levy(dim))
                        if objective_func(X2) < objective_func(X[i, :]):
                            X[i, :] = X2.copy()
                
                if r < 0.5 and abs(Escaping_Energy) < 0.5:
                    # حصار صعب (المعادلة 11)
                    Jump_strength = 2 * (1 - np.random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(0))
                    
                    if objective_func(X1) < objective_func(X[i, :]):  # تحسن الحركة؟
                        X[i, :] = X1.copy()
                    else:
                        # تنفيذ غطسات سريعة قصيرة بتوزيع ليفي حول الأرنب
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(0)) + np.multiply(np.random.randn(dim), Levy(dim))
                        if objective_func(X2) < objective_func(X[i, :]):
                            X[i, :] = X2.copy()
        
        convergence_curve[t] = Rabbit_Energy
        
        if t % 50 == 0:
            print(f"التكرار {t}: أفضل قيمة = {Rabbit_Energy}")
        
        t += 1
    
    # وقت الانتهاء
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"انتهى التنفيذ في {execution_time:.2f} ثانية")
    print(f"أفضل قيمة: {Rabbit_Energy}")
    
    return Rabbit_Energy, Rabbit_Location, convergence_curve

def Levy(dim):
    """
    حساب توزيع ليفي للبحث
    """
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v), (1 / beta))
    step = np.divide(u, zz)
    return step
