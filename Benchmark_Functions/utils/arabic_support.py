"""
وحدة دعم اللغة العربية في الرسومات البيانية

هذا الملف يحتوي على الدوال والإعدادات اللازمة لدعم اللغة العربية في الرسومات البيانية
باستخدام مكتبة Matplotlib مع دعم arabic-reshaper وpython-bidi
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

def setup_arabic_fonts():
    """
    إعداد الخطوط العربية في Matplotlib
    
    تقوم هذه الدالة بإعداد الخطوط التي تدعم اللغة العربية في مكتبة Matplotlib
    وتعيين الإعدادات المناسبة لعرض النصوص العربية بشكل صحيح
    """
    # قائمة الخطوط التي تدعم اللغة العربية
    arabic_fonts = ['DejaVu Sans', 'Arial', 'Tahoma', 'Times New Roman']
    
    # تعيين الخطوط المفضلة
    plt.rcParams['font.family'] = arabic_fonts
    
    # تعيين حجم الخط الافتراضي
    plt.rcParams['font.size'] = 12
    
    # تمكين دعم اللغة العربية في Matplotlib
    plt.rcParams['axes.unicode_minus'] = False
    
    print("تم إعداد الخطوط العربية بنجاح")
    
    # عرض الخطوط المتاحة التي تدعم العربية
    available_fonts = [f.name for f in fm.fontManager.ttflist if any(af in f.name for af in arabic_fonts)]
    print(f"الخطوط المتاحة التي تدعم العربية: {available_fonts}")
    
    return available_fonts

def process_arabic_text(text):
    """
    معالجة النص العربي لعرضه بشكل صحيح في الرسومات البيانية
    
    المعلمات:
        text: النص العربي المراد معالجته
        
    المخرجات:
        النص بعد المعالجة
    """
    # إعادة تشكيل النص العربي
    reshaped_text = arabic_reshaper.reshape(text)
    
    # تطبيق خوارزمية BIDI لعرض النص بالاتجاه الصحيح
    bidi_text = get_display(reshaped_text)
    
    return bidi_text

def test_arabic_support():
    """
    اختبار دعم اللغة العربية في الرسومات البيانية
    
    تقوم هذه الدالة بإنشاء رسم بياني بسيط مع نصوص عربية للتأكد من دعم اللغة العربية
    """
    # إعداد الخطوط العربية
    setup_arabic_fonts()
    
    # إنشاء رسم بياني بسيط
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # بيانات للرسم
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # رسم البيانات
    ax.plot(x, y)
    
    # إضافة عناوين بالعربية مع معالجة النص
    ax.set_title(process_arabic_text("اختبار دعم اللغة العربية في الرسومات البيانية"), fontsize=14)
    ax.set_xlabel(process_arabic_text("المحور السيني"), fontsize=12)
    ax.set_ylabel(process_arabic_text("المحور الصادي"), fontsize=12)
    
    # إضافة شبكة
    ax.grid(True)
    
    # حفظ الرسم البياني
    output_dir = "/home/ubuntu/project/results/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/arabic_test_improved.png", dpi=300, bbox_inches='tight')
    
    print(f"تم حفظ الرسم البياني في {output_dir}/arabic_test_improved.png")
    
    return f"{output_dir}/arabic_test_improved.png"

def create_arabic_boxplot(data, labels, title, xlabel, ylabel, filename):
    """
    إنشاء رسم صندوقي يدعم اللغة العربية
    
    المعلمات:
        data: قائمة من البيانات للرسم الصندوقي
        labels: قائمة من التسميات للمحور السيني
        title: عنوان الرسم البياني
        xlabel: تسمية المحور السيني
        ylabel: تسمية المحور الصادي
        filename: اسم ملف الحفظ (بدون المسار)
    """
    # إعداد الخطوط العربية
    setup_arabic_fonts()
    
    # إنشاء رسم بياني
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # رسم البيانات
    ax.boxplot(data)
    
    # معالجة النصوص العربية
    processed_title = process_arabic_text(title)
    processed_xlabel = process_arabic_text(xlabel)
    processed_ylabel = process_arabic_text(ylabel)
    processed_labels = [process_arabic_text(label) for label in labels]
    
    # إضافة عناوين بالعربية
    ax.set_title(processed_title, fontsize=14)
    ax.set_xlabel(processed_xlabel, fontsize=12)
    ax.set_ylabel(processed_ylabel, fontsize=12)
    
    # تعيين تسميات المحور السيني
    ax.set_xticklabels(processed_labels, rotation=45)
    
    # إضافة شبكة
    ax.grid(True)
    
    # تحسين التخطيط
    plt.tight_layout()
    
    # حفظ الرسم البياني
    output_dir = "/home/ubuntu/project/results/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    
    print(f"تم حفظ الرسم البياني في {output_dir}/{filename}")
    
    return f"{output_dir}/{filename}"

def create_arabic_convergence_plot(data_dict, title, xlabel, ylabel, filename):
    """
    إنشاء رسم منحنيات التقارب يدعم اللغة العربية
    
    المعلمات:
        data_dict: قاموس يحتوي على البيانات للرسم (المفتاح: اسم الخوارزمية، القيمة: منحنى التقارب)
        title: عنوان الرسم البياني
        xlabel: تسمية المحور السيني
        ylabel: تسمية المحور الصادي
        filename: اسم ملف الحفظ (بدون المسار)
    """
    # إعداد الخطوط العربية
    setup_arabic_fonts()
    
    # إنشاء رسم بياني
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # رسم البيانات
    for label, data in data_dict.items():
        ax.semilogy(data, label=process_arabic_text(label))
    
    # معالجة النصوص العربية
    processed_title = process_arabic_text(title)
    processed_xlabel = process_arabic_text(xlabel)
    processed_ylabel = process_arabic_text(ylabel)
    
    # إضافة عناوين بالعربية
    ax.set_title(processed_title, fontsize=14)
    ax.set_xlabel(processed_xlabel, fontsize=12)
    ax.set_ylabel(processed_ylabel, fontsize=12)
    
    # إضافة وسيلة إيضاح
    ax.legend(loc='best', fontsize=10)
    
    # إضافة شبكة
    ax.grid(True)
    
    # تحسين التخطيط
    plt.tight_layout()
    
    # حفظ الرسم البياني
    output_dir = "/home/ubuntu/project/results/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}", dpi=300, bbox_inches='tight')
    
    print(f"تم حفظ الرسم البياني في {output_dir}/{filename}")
    
    return f"{output_dir}/{filename}"

if __name__ == "__main__":
    # اختبار دعم اللغة العربية
    test_arabic_support()
