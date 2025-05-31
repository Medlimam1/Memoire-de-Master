"""
ملف الإعدادات الموحدة للتجربة

هذا الملف يحتوي على الإعدادات الموحدة المطلوبة لتنفيذ التجارب على جميع الخوارزميات والدوال.
"""

# إعدادات التجربة الموحدة
DIMENSIONS = 30  # عدد الأبعاد D = 30
POPULATION_SIZE = 50  # حجم المجتمع Population Size = 50
MAX_ITERATIONS = 1000  # عدد التكرارات Max Iterations = 1000
INDEPENDENT_RUNS = 30  # عدد التشغيلات المستقلة Runs = 30

# قائمة الخوارزميات المطلوبة
ALGORITHMS = [
    "HHO",  # Harris Hawks Optimization
    "AOA",  # Arithmetic Optimization Algorithm
    "AO",   # Aquila Optimizer
    "GBO",  # Gradient-Based Optimizer
    "FDA",  # Flow Direction Algorithm
    "EO",   # Equilibrium Optimizer
    "GMO"   # Geometric Mean Optimizer
]

# قائمة الدوال القياسية المطلوبة
FUNCTIONS = [
    "F1",  # Sphere
    "F2",  # Rastrigin
    "F3",  # Zakharov
    "F8",  # Levy
    "F9",  # Schwefel
    "F10"  # Ackley
]

# مؤشرات التقييم المطلوبة
METRICS = [
    "Mean",    # المتوسط
    "Std",     # الانحراف المعياري
    "Best",    # أفضل نتيجة
    "Worst"    # أسوأ نتيجة
]

# مسارات حفظ النتائج
RESULTS_DIR = "/home/ubuntu/project/results"
TABLES_DIR = "/home/ubuntu/project/results/tables"
PLOTS_DIR = "/home/ubuntu/project/results/plots"
REPORT_DIR = "/home/ubuntu/project/results/report"

# تنسيقات الملفات
CSV_FORMAT = "csv"
EXCEL_FORMAT = "xlsx"
PLOT_FORMAT = "png"
REPORT_FORMAT = "pdf"

# إعدادات دعم اللغة العربية في الرسومات البيانية
ARABIC_FONT = "DejaVu Sans"  # خط يدعم اللغة العربية
FONT_SIZE = 12  # حجم الخط الافتراضي
TITLE_FONT_SIZE = 14  # حجم خط العناوين
LABEL_FONT_SIZE = 12  # حجم خط التسميات
LEGEND_FONT_SIZE = 10  # حجم خط وسيلة الإيضاح

# إعدادات الرسومات البيانية
FIGURE_WIDTH = 10  # عرض الشكل البياني
FIGURE_HEIGHT = 6  # ارتفاع الشكل البياني
DPI = 300  # دقة الرسم البياني
