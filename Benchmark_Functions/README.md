# 🧪 Benchmark Functions – Geometric Mean Optimizer (GMO)

هذا المجلد يحتوي على الكود الخاص باختبار أداء خوارزمية **Geometric Mean Optimizer (GMO)** على دوال رياضية معيارية (Benchmark Functions)، مع مقارنتها بخوارزميات تحسين أخرى مثل PSO وGA.

---

## 📁 محتويات المجلد

| العنصر | الوصف |
|--------|-------|
| `algorithms/` | يحتوي على تعريف وتنفيذ الخوارزميات (GMO، GA، PSO، AOA...). |
| `benchmark_functions/` | دوال رياضية معيارية مثل Sphere، Rastrigin، Ackley وغيرها. |
| `utils/` | دوال مساعدة مثل إدارة النتائج، الرسم البياني، التهيئة. |
| `mini_test.py` | اختبار سريع لوظائف أساسية للتحقق من السلامة. |
| `mini_test_improved.py` | نسخة محسّنة من الاختبار السريع مع تحسين الرسوميات. |
| `run_experiments.py` | تشغيل التجارب الرسمية على جميع الدوال، مع تسجيل النتائج. |
| `run_experiments_improved.py` | نسخة محسّنة بخصائص إضافية (حفظ النتائج، الرسم التلقائي). |
| `test_arabic_support.py` | اختبار دعم اللغة العربية في الرسوم البيانية أو التقارير (اختياري). |

---

## ⚙️ كيفية التشغيل

### 1. المتطلبات

قبل التشغيل، تأكد من تثبيت المكتبات التالية:

```bash
pip install numpy matplotlib pandas scipy
