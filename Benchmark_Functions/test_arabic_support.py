"""
اختبار دعم اللغة العربية في الرسومات البيانية
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.arabic_support import setup_arabic_fonts, test_arabic_support

if __name__ == "__main__":
    # اختبار دعم اللغة العربية
    print("جاري اختبار دعم اللغة العربية في الرسومات البيانية...")
    image_path = test_arabic_support()
    print(f"تم إنشاء الرسم البياني بنجاح في: {image_path}")
