"""
ملف تنفيذ التجارب المصغرة للاختبار

هذا الملف يقوم بتنفيذ تجارب مصغرة على الخوارزميات للتأكد من صحة التنفيذ
قبل تشغيل التجارب الكاملة
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

# استيراد الإعدادات الموحدة
from utils.settings import *

# استيراد دعم اللغة العربية
from utils.arabic_support import setup_arabic_fonts, process_arabic_text, create_arabic_convergence_plot

# استيراد الدوال القياسية
from benchmark_functions.functions import benchmark_functions

# استيراد الخوارزميات
from algorithms.HHO import HHO
from algorithms.AOA import AOA
from algorithms.AO import AO
from algorithms.GBO import GBO
from algorithms.FDA import FDA
from algorithms.EO import EO
from algorithms.GMO import GMO

# إعدادات التجارب المصغرة
MINI_DIMENSIONS = 10  # عدد أبعاد مصغر
MINI_POPULATION_SIZE = 20  # حجم مجتمع مصغر
MINI_MAX_ITERATIONS = 100  # عدد تكرارات مصغر

# قاموس يحتوي على جميع الخوارزميات
algorithms = {
    "HHO": HHO,
    "AOA": AOA,
    "AO": AO,
    "GBO": GBO,
    "FDA": FDA,
    "EO": EO,
    "GMO": GMO
}

def run_mini_test():
    """
    تنفيذ اختبار مصغر للتأكد من عمل جميع الخوارزميات
    """
    # إعداد دعم اللغة العربية
    setup_arabic_fonts()
    
    print("بدء تنفيذ الاختبار المصغر...")
    
    # اختيار دالة واحدة للاختبار
    function_name = "F1"  # Sphere function
    function_info = benchmark_functions[function_name]
    objective_func = function_info["function"]
    lb = function_info["lb"]
    ub = function_info["ub"]
    
    # تهيئة حدود البحث
    lb_array = np.ones(MINI_DIMENSIONS) * lb
    ub_array = np.ones(MINI_DIMENSIONS) * ub
    
    # تنفيذ اختبار لكل خوارزمية
    results = {}
    
    for algorithm_name, algorithm_func in algorithms.items():
        print(f"اختبار الخوارزمية {algorithm_name}...")
        
        # قياس وقت التنفيذ
        start_time = time.time()
        
        # تنفيذ الخوارزمية
        best_score, best_pos, convergence_curve = algorithm_func(
            objective_func, lb_array, ub_array, MINI_DIMENSIONS, MINI_POPULATION_SIZE, MINI_MAX_ITERATIONS
        )
        
        # حساب وقت التنفيذ
        execution_time = time.time() - start_time
        
        # تخزين النتائج
        results[algorithm_name] = {
            "best_score": best_score,
            "execution_time": execution_time,
            "convergence_curve": convergence_curve
        }
        
        print(f"  أفضل قيمة = {best_score}, الوقت = {execution_time:.2f} ثانية")
    
    # إنشاء قاموس لمنحنيات التقارب
    convergence_data = {}
    for algorithm_name, result in results.items():
        convergence_data[algorithm_name] = result["convergence_curve"]
    
    # رسم منحنيات التقارب باستخدام دعم العربية المحسن
    create_arabic_convergence_plot(
        data_dict=convergence_data,
        title=f"منحنيات التقارب للدالة {function_name} (اختبار مصغر)",
        xlabel="التكرار",
        ylabel="أفضل قيمة (مقياس لوغاريتمي)",
        filename="mini_test_convergence_improved.png"
    )
    
    # طباعة ملخص النتائج
    print("\nملخص نتائج الاختبار المصغر:")
    print("--------------------------")
    print(f"{'الخوارزمية':<10} | {'أفضل قيمة':<20} | {'وقت التنفيذ (ثانية)':<20}")
    print("-" * 60)
    
    for algorithm_name, result in results.items():
        print(f"{algorithm_name:<10} | {result['best_score']:<20.6e} | {result['execution_time']:<20.2f}")
    
    return results

if __name__ == "__main__":
    run_mini_test()
