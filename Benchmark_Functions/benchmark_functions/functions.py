"""
دوال الاختبار القياسية (Benchmark Functions)

هذا الملف يحتوي على الدوال القياسية المطلوبة للمقارنة:
F1 (Sphere)، F2 (Rastrigin)، F3 (Zakharov)، F8 (Levy)، F9 (Modified Schwefel)، F10 (Ackley)
"""

import numpy as np
import math

# F1: Sphere Function
def F1(x):
    """
    دالة Sphere
    الحد الأدنى: f(0,...,0) = 0
    النطاق المعتاد: [-100, 100]^d
    """
    return np.sum(x**2)

# F2: Rastrigin Function
def F2(x):
    """
    دالة Rastrigin
    الحد الأدنى: f(0,...,0) = 0
    النطاق المعتاد: [-5.12, 5.12]^d
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# F3: Zakharov Function
def F3(x):
    """
    دالة Zakharov
    الحد الأدنى: f(0,...,0) = 0
    النطاق المعتاد: [-10, 10]^d
    """
    n = len(x)
    j = np.arange(1, n+1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * j * x)
    return sum1 + sum2**2 + sum2**4

# F8: Levy Function
def F8(x):
    """
    دالة Levy
    الحد الأدنى: f(1,...,1) = 0
    النطاق المعتاد: [-10, 10]^d
    """
    n = len(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[n-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[n-1])**2)
    
    sum_term = 0
    for i in range(n-1):
        sum_term += (w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2)
    
    return term1 + sum_term + term3

# F9: Schwefel Function (تصحيح الصيغة بناءً على المرجع الأصلي)
def F9(x):
    """
    دالة Schwefel
    الحد الأدنى: f(420.9687,...,420.9687) = 0
    النطاق المعتاد: [-500, 500]^d
    
    الصيغة الصحيحة: f(x) = 418.9829*d - sum(x_i * sin(sqrt(abs(x_i))))
    """
    n = len(x)
    result = 0
    for i in range(n):
        result += x[i] * np.sin(np.sqrt(abs(x[i])))
    
    return 418.9829 * n - result

# F10: Ackley Function
def F10(x):
    """
    دالة Ackley
    الحد الأدنى: f(0,...,0) = 0
    النطاق المعتاد: [-32, 32]^d
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + 20 + np.exp(1)

# قاموس يحتوي على جميع الدوال مع نطاقاتها
benchmark_functions = {
    "F1": {"function": F1, "lb": -100, "ub": 100, "name": "Sphere"},
    "F2": {"function": F2, "lb": -5.12, "ub": 5.12, "name": "Rastrigin"},
    "F3": {"function": F3, "lb": -10, "ub": 10, "name": "Zakharov"},
    "F8": {"function": F8, "lb": -10, "ub": 10, "name": "Levy"},
    "F9": {"function": F9, "lb": -500, "ub": 500, "name": "Schwefel"},
    "F10": {"function": F10, "lb": -32, "ub": 32, "name": "Ackley"}
}

# دالة للتحقق من صحة الدوال
def verify_functions():
    """
    التحقق من صحة الدوال القياسية
    """
    print("التحقق من صحة الدوال القياسية:")
    
    # F1: Sphere - الحد الأدنى عند (0,...,0) = 0
    x_min_f1 = np.zeros(30)
    print(f"F1 (Sphere) عند النقطة المثلى: {F1(x_min_f1)}")
    
    # F2: Rastrigin - الحد الأدنى عند (0,...,0) = 0
    x_min_f2 = np.zeros(30)
    print(f"F2 (Rastrigin) عند النقطة المثلى: {F2(x_min_f2)}")
    
    # F3: Zakharov - الحد الأدنى عند (0,...,0) = 0
    x_min_f3 = np.zeros(30)
    print(f"F3 (Zakharov) عند النقطة المثلى: {F3(x_min_f3)}")
    
    # F8: Levy - الحد الأدنى عند (1,...,1) = 0
    x_min_f8 = np.ones(30)
    print(f"F8 (Levy) عند النقطة المثلى: {F8(x_min_f8)}")
    
    # F9: Schwefel - الحد الأدنى عند (420.9687,...,420.9687) ≈ 0
    x_min_f9 = np.ones(30) * 420.9687
    print(f"F9 (Schwefel) عند النقطة المثلى: {F9(x_min_f9)}")
    
    # F10: Ackley - الحد الأدنى عند (0,...,0) = 0
    x_min_f10 = np.zeros(30)
    print(f"F10 (Ackley) عند النقطة المثلى: {F10(x_min_f10)}")

if __name__ == "__main__":
    verify_functions()
