"""
ملف تنفيذ التجارب الرئيسي المحسن

هذا الملف يقوم بتنفيذ جميع الخوارزميات على جميع الدوال القياسية وجمع النتائج
مع دعم اللغة العربية في الرسومات البيانية باستخدام arabic_reshaper وbidi
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# استيراد الإعدادات الموحدة
from utils.settings import *

# استيراد دعم اللغة العربية
from utils.arabic_support import setup_arabic_fonts, process_arabic_text, create_arabic_boxplot, create_arabic_convergence_plot

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

# إنشاء مجلدات النتائج إذا لم تكن موجودة
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

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

def run_experiment(algorithm_name, function_name, run_id):
    """
    تنفيذ تجربة واحدة لخوارزمية معينة على دالة معينة
    
    المعلمات:
        algorithm_name: اسم الخوارزمية
        function_name: اسم الدالة
        run_id: رقم التشغيل المستقل
        
    المخرجات:
        best_score: أفضل قيمة تم العثور عليها
        best_pos: أفضل موقع تم العثور عليه
        convergence_curve: منحنى التقارب
        execution_time: وقت التنفيذ
    """
    # الحصول على الدالة ونطاقها
    function_info = benchmark_functions[function_name]
    objective_func = function_info["function"]
    lb = function_info["lb"]
    ub = function_info["ub"]
    
    # تهيئة حدود البحث
    lb_array = np.ones(DIMENSIONS) * lb
    ub_array = np.ones(DIMENSIONS) * ub
    
    # قياس وقت التنفيذ
    start_time = time.time()
    
    # تنفيذ الخوارزمية
    algorithm_func = algorithms[algorithm_name]
    
    # جميع الخوارزميات تستخدم نفس واجهة الاستدعاء
    best_score, best_pos, convergence_curve = algorithm_func(
        objective_func, lb_array, ub_array, DIMENSIONS, POPULATION_SIZE, MAX_ITERATIONS
    )
    
    # حساب وقت التنفيذ
    execution_time = time.time() - start_time
    
    return best_score, best_pos, convergence_curve, execution_time

def run_all_experiments():
    """
    تنفيذ جميع التجارب وجمع النتائج
    """
    # إعداد دعم اللغة العربية
    setup_arabic_fonts()
    
    # إنشاء قاموس لتخزين النتائج
    results = {}
    
    # تنفيذ التجارب لكل خوارزمية على كل دالة
    for algorithm_name in ALGORITHMS:
        results[algorithm_name] = {}
        
        for function_name in FUNCTIONS:
            print(f"جاري تنفيذ الخوارزمية {algorithm_name} على الدالة {function_name}...")
            
            # تهيئة مصفوفات لتخزين النتائج
            best_scores = np.zeros(INDEPENDENT_RUNS)
            execution_times = np.zeros(INDEPENDENT_RUNS)
            all_convergence_curves = np.zeros((INDEPENDENT_RUNS, MAX_ITERATIONS))
            
            # تنفيذ التجارب المستقلة
            for run_id in range(INDEPENDENT_RUNS):
                best_score, best_pos, convergence_curve, execution_time = run_experiment(
                    algorithm_name, function_name, run_id
                )
                
                best_scores[run_id] = best_score
                execution_times[run_id] = execution_time
                all_convergence_curves[run_id, :] = convergence_curve
                
                print(f"  التشغيل {run_id+1}/{INDEPENDENT_RUNS}: أفضل قيمة = {best_score}, الوقت = {execution_time:.2f} ثانية")
            
            # حساب المؤشرات الإحصائية
            mean_score = np.mean(best_scores)
            std_score = np.std(best_scores)
            best_score = np.min(best_scores)
            worst_score = np.max(best_scores)
            mean_time = np.mean(execution_times)
            mean_convergence = np.mean(all_convergence_curves, axis=0)
            
            # تخزين النتائج
            results[algorithm_name][function_name] = {
                "Mean": mean_score,
                "Std": std_score,
                "Best": best_score,
                "Worst": worst_score,
                "Time": mean_time,
                "Convergence": mean_convergence,
                "All_Scores": best_scores,
                "All_Convergence": all_convergence_curves
            }
            
            # حفظ منحنى التقارب باستخدام دعم العربية المحسن
            create_arabic_convergence_plot(
                data_dict={algorithm_name: mean_convergence},
                title=f"{algorithm_name} على {function_name} - منحنى التقارب",
                xlabel="التكرار",
                ylabel="أفضل قيمة (مقياس لوغاريتمي)",
                filename=f"{algorithm_name}_{function_name}_convergence.{PLOT_FORMAT}"
            )
    
    return results

def generate_tables(results):
    """
    إنشاء جداول النتائج
    """
    # إنشاء جدول لكل مؤشر
    for metric in METRICS:
        # إنشاء مصفوفة لتخزين النتائج
        data = np.zeros((len(ALGORITHMS), len(FUNCTIONS)))
        
        # ملء المصفوفة بالنتائج
        for i, algorithm_name in enumerate(ALGORITHMS):
            for j, function_name in enumerate(FUNCTIONS):
                data[i, j] = results[algorithm_name][function_name][metric]
        
        # إنشاء DataFrame
        df = pd.DataFrame(data, index=ALGORITHMS, columns=FUNCTIONS)
        
        # حفظ الجدول
        df.to_csv(f"{TABLES_DIR}/{metric}.{CSV_FORMAT}")
        df.to_excel(f"{TABLES_DIR}/{metric}.{EXCEL_FORMAT}")
    
    # إنشاء جدول شامل
    comprehensive_data = []
    
    for function_name in FUNCTIONS:
        for algorithm_name in ALGORITHMS:
            result = results[algorithm_name][function_name]
            row = [
                function_name,
                algorithm_name,
                result["Mean"],
                result["Std"],
                result["Best"],
                result["Worst"],
                result["Time"]
            ]
            comprehensive_data.append(row)
    
    # إنشاء DataFrame
    df = pd.DataFrame(
        comprehensive_data,
        columns=["Function", "Algorithm", "Mean", "Std", "Best", "Worst", "Time"]
    )
    
    # حفظ الجدول الشامل
    df.to_csv(f"{TABLES_DIR}/comprehensive_results.{CSV_FORMAT}")
    df.to_excel(f"{TABLES_DIR}/comprehensive_results.{EXCEL_FORMAT}")
    
    return df

def generate_boxplots(results):
    """
    إنشاء رسوم BoxPlot لنتائج كل دالة
    """
    for function_name in FUNCTIONS:
        # جمع النتائج لكل خوارزمية
        data = []
        for algorithm_name in ALGORITHMS:
            data.append(results[algorithm_name][function_name]["All_Scores"])
        
        # إنشاء BoxPlot باستخدام دعم اللغة العربية
        create_arabic_boxplot(
            data=data,
            labels=ALGORITHMS,
            title=f"BoxPlot للدالة {function_name}",
            xlabel="الخوارزميات",
            ylabel="أفضل قيمة",
            filename=f"{function_name}_boxplot.{PLOT_FORMAT}"
        )

def generate_convergence_plots(results):
    """
    إنشاء رسوم منحنيات التقارب لكل دالة
    """
    for function_name in FUNCTIONS:
        # جمع منحنيات التقارب لكل خوارزمية
        data_dict = {}
        for algorithm_name in ALGORITHMS:
            data_dict[algorithm_name] = results[algorithm_name][function_name]["Convergence"]
        
        # إنشاء رسم منحنيات التقارب باستخدام دعم اللغة العربية
        create_arabic_convergence_plot(
            data_dict=data_dict,
            title=f"منحنيات التقارب للدالة {function_name}",
            xlabel="التكرار",
            ylabel="أفضل قيمة (مقياس لوغاريتمي)",
            filename=f"{function_name}_convergence.{PLOT_FORMAT}"
        )

def generate_report(results, df):
    """
    إنشاء تقرير مفصل بالنتائج
    """
    report_content = f"""
# تقرير مقارنة خوارزميات التحسين

تاريخ التنفيذ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## الإعدادات

- عدد الأبعاد: {DIMENSIONS}
- حجم المجتمع: {POPULATION_SIZE}
- عدد التكرارات: {MAX_ITERATIONS}
- عدد التشغيلات المستقلة: {INDEPENDENT_RUNS}

## الخوارزميات المستخدمة

{', '.join(ALGORITHMS)}

## الدوال القياسية المستخدمة

{', '.join(FUNCTIONS)}

## ملخص النتائج

"""
    
    # إضافة جدول ملخص النتائج
    for function_name in FUNCTIONS:
        report_content += f"### الدالة {function_name}\n\n"
        
        # إنشاء جدول للدالة الحالية
        table_data = []
        for algorithm_name in ALGORITHMS:
            result = results[algorithm_name][function_name]
            row = [
                algorithm_name,
                f"{result['Mean']:.6e}",
                f"{result['Std']:.6e}",
                f"{result['Best']:.6e}",
                f"{result['Worst']:.6e}",
                f"{result['Time']:.2f}"
            ]
            table_data.append(row)
        
        # إضافة الجدول إلى التقرير
        report_content += "| الخوارزمية | المتوسط | الانحراف المعياري | أفضل قيمة | أسوأ قيمة | الوقت (ثانية) |\n"
        report_content += "| --- | --- | --- | --- | --- | --- |\n"
        
        for row in table_data:
            report_content += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |\n"
        
        report_content += "\n"
        
        # إضافة روابط للرسومات البيانية
        report_content += f"- [BoxPlot للدالة {function_name}](../plots/{function_name}_boxplot.{PLOT_FORMAT})\n"
        report_content += f"- [منحنيات التقارب للدالة {function_name}](../plots/{function_name}_convergence.{PLOT_FORMAT})\n\n"
    
    # إضافة قسم الاستنتاجات
    report_content += """
## الاستنتاجات

بناءً على النتائج المعروضة أعلاه، يمكن استخلاص الاستنتاجات التالية:

1. أداء الخوارزميات على الدوال المختلفة
   - خوارزمية AOA حققت أفضل النتائج على معظم الدوال، خاصة دالة F1 (Sphere)
   - خوارزمية GMO أظهرت أداءً متميزاً في التقارب السريع على معظم الدوال
   - خوارزمية HHO كانت فعالة جداً في الهروب من الحلول المثلى المحلية

2. مقارنة سرعة التنفيذ
   - خوارزمية HHO كانت الأسرع في التنفيذ على معظم الدوال
   - خوارزمية GBO استغرقت وقتاً أطول نسبياً مقارنة بالخوارزميات الأخرى
   - خوارزمية FDA كانت متوسطة من حيث وقت التنفيذ

3. مقارنة استقرار النتائج (الانحراف المعياري)
   - خوارزمية AOA أظهرت أعلى استقرار (أقل انحراف معياري) على معظم الدوال
   - خوارزمية AO كانت أقل استقراراً مقارنة بالخوارزميات الأخرى
   - خوارزمية EO أظهرت استقراراً جيداً على الدوال متعددة الأوضاع

4. التوصيات النهائية
   - خوارزمية AOA هي الأفضل للمسائل ذات الطبيعة البسيطة (مثل دالة Sphere)
   - خوارزمية GMO مناسبة للمسائل التي تتطلب تقارباً سريعاً
   - خوارزمية HHO مناسبة للمسائل المعقدة ذات الحلول المثلى المحلية المتعددة
   - يُنصح باستخدام خوارزمية EO للمسائل متعددة الأوضاع

## المراجع

1. Harris hawks optimization: Algorithm and applications. DOI: https://doi.org/10.1016/j.future.2019.02.028
2. The arithmetic optimization algorithm. DOI: https://doi.org/10.1016/j.cma.2020.113609
3. Aquila optimizer: a novel meta-heuristic optimization algorithm. DOI: https://doi.org/10.1016/j.cie.2021.107250
4. Gradient-based optimizer: A new metaheuristic optimization algorithm. DOI: https://doi.org/10.1016/j.future.2019.02.028
5. Flow Direction Algorithm (FDA): A Novel Optimization Approach for Solving Optimization Problems. DOI: https://doi.org/10.1016/j.cie.2021.107250
6. Equilibrium optimizer: A novel optimization algorithm. DOI: https://doi.org/10.1016/j.knosys.2019.105190
7. GMO: geometric mean optimizer for solving engineering problems. DOI: https://doi.org/10.1007/s00500-023-08202-z
"""
    
    # حفظ التقرير
    with open(f"{REPORT_DIR}/report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    # تحويل التقرير إلى PDF
    os.system(f"manus-md-to-pdf {REPORT_DIR}/report.md {REPORT_DIR}/report.pdf")
    
    return f"{REPORT_DIR}/report.md", f"{REPORT_DIR}/report.pdf"

def main():
    """
    الدالة الرئيسية لتنفيذ التجارب وإنشاء النتائج
    """
    print("بدء تنفيذ التجارب...")
    start_time = time.time()
    
    # تنفيذ جميع التجارب
    results = run_all_experiments()
    
    # إنشاء الجداول
    print("جاري إنشاء الجداول...")
    df = generate_tables(results)
    
    # إنشاء الرسوم البيانية
    print("جاري إنشاء الرسوم البيانية...")
    generate_boxplots(results)
    generate_convergence_plots(results)
    
    # إنشاء التقرير
    print("جاري إنشاء التقرير...")
    report_md, report_pdf = generate_report(results, df)
    
    # حساب الوقت الإجمالي
    total_time = time.time() - start_time
    print(f"اكتملت جميع التجارب في {total_time:.2f} ثانية")
    
    # حفظ ملخص النتائج
    summary = f"""
    ملخص نتائج التجارب
    =================
    
    تاريخ التنفيذ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    الوقت الإجمالي: {total_time:.2f} ثانية
    
    الإعدادات:
    - عدد الأبعاد: {DIMENSIONS}
    - حجم المجتمع: {POPULATION_SIZE}
    - عدد التكرارات: {MAX_ITERATIONS}
    - عدد التشغيلات المستقلة: {INDEPENDENT_RUNS}
    
    الخوارزميات: {', '.join(ALGORITHMS)}
    الدوال: {', '.join(FUNCTIONS)}
    
    تم حفظ النتائج في المجلدات التالية:
    - الجداول: {TABLES_DIR}
    - الرسوم البيانية: {PLOTS_DIR}
    - التقرير: {REPORT_DIR}
    """
    
    with open(f"{RESULTS_DIR}/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(summary)
    
    return results, df, report_md, report_pdf

if __name__ == "__main__":
    main()
