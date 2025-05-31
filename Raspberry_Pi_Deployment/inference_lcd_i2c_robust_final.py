#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام اكتشاف الالتهاب الرئوي باستخدام زر مادي وشاشة I2C LCD وكاميرا USB على Raspberry Pi
(إصدار نهائي ومحسن v3: يتضمن رسالة ترحيبية، معالجة قوية للزر، وتحسينات استقرار)

الاستخدام:
    sudo python3 inference_lcd_i2c_robust_final.py [--model MODEL_PATH] [--address I2C_ADDRESS]
    (ملاحظة: يفضل تشغيله باستخدام sudo لضمان صلاحيات GPIO الكاملة)

المتطلبات:
    - Raspberry Pi مع GPIO
    - زر مادي متصل بـ GPIO 23 (مع مقاومة سحب خارجية 10k إلى 3.3V)
    - شاشة LCD 16x2 I2C (PCF8574) (متصلة بـ SDA/SCL - العنوان الافتراضي 0x27)
    - مؤشر LED أحمر (GPIO 17)
    - مؤشر LED أخضر (GPIO 27)
    - جرس إنذار (GPIO 22)
    - كاميرا USB متصلة
    - ملف النموذج المدرب (pneumonia_model.joblib)
"""

import os
import time
import numpy as np
import cv2
import joblib
import argparse
import threading
import signal
import sys

# --- تكوين --- 
MODEL_PATH = "pneumonia_model.joblib"
TEMP_IMAGE_PATH = "captured_image.jpg"
FEATURES_EXTRACTION_SIZE = (224, 224)  # حجم الصورة للمعالجة

# تكوين GPIO (حسب الدليل والتوصيلات)
BUTTON_PIN = 23  # GPIO للزر (مقاومة سحب خارجية)
RED_LED_PIN = 17 # GPIO للمؤشر الأحمر (نتيجة إيجابية)
GREEN_LED_PIN = 27 # GPIO للمؤشر الأخضر (نتيجة سلبية)
BUZZER_PIN = 22  # GPIO للجرس (مع النتيجة الإيجابية)

# تكوين I2C LCD
DEFAULT_I2C_ADDRESS = 0x27 # العنوان الشائع لشاشات PCF8574، قد يكون 0x3F
I2C_PORT = 1 # المنفذ الافتراضي على Raspberry Pi
LCD_COLS = 16
LCD_ROWS = 2

# متغيرات عامة
model_data = None
button_pressed_flag = False # علم للإشارة إلى الضغط على الزر
is_processing = False
lcd = None # سيتم تهيئته في setup_lcd
gpio_available = False # لتتبع توفر مكتبة GPIO

# --- دوال المساعدة --- 
def display_message(text_line1="", text_line2="", clear_first=True):
    """عرض رسالة على شاشة LCD (سطرين)"""
    global lcd
    if lcd:
        try:
            if clear_first:
                lcd.clear()
            lcd.cursor_pos = (0, 0)
            lcd.write_string(text_line1[:LCD_COLS]) # اقتصاص النص إذا كان أطول من اللازم
            lcd.cursor_pos = (1, 0)
            lcd.write_string(text_line2[:LCD_COLS])
        except Exception as e:
            print(f"Error displaying on LCD: {str(e)}")
    else:
        # طباعة للمحاكاة
        print(f"LCD L1: {text_line1}")
        print(f"LCD L2: {text_line2}")

def load_model(model_path):
    """تحميل النموذج المدرب"""
    print("Loading model...")
    display_message("Loading Model...")
    try:
        model_data = joblib.load(model_path)
        accuracy = model_data.get("accuracy", "N/A")
        print(f"Model loaded successfully (Accuracy: {accuracy})")
        # لا نعرض الدقة هنا، سنعرض الرسالة الترحيبية بعد إعداد الشاشة
        # display_message("Model Loaded", f"Acc: {accuracy:.2f}" if isinstance(accuracy, float) else f"Acc: {accuracy}")
        # time.sleep(1.5)
        return model_data
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        display_message("Model Load Error", str(e)[:LCD_COLS])
        return None

# --- إعداد العتاد --- 
def setup_gpio():
    """إعداد GPIO للزر والمؤشرات والجرس"""
    global gpio_available
    try:
        import RPi.GPIO as GPIO
        gpio_available = True
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        # محاولة تنظيف أي إعدادات سابقة قد تسبب مشاكل
        try:
            GPIO.cleanup()
            print("Performed GPIO cleanup before setup.")
        except Exception as e:
            print(f"Note: GPIO cleanup before setup failed (maybe first run?): {e}")
            
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP) # استخدام مقاومة السحب الداخلية كبديل/إضافة
        GPIO.setup(RED_LED_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(GREEN_LED_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
        print("GPIO setup successful.")
        return True

    except ImportError:
        print("Warning: RPi.GPIO library not found. Running in simulation mode.")
        gpio_available = False
        return False
    except Exception as e:
        print(f"Error setting up GPIO: {str(e)}")
        gpio_available = False
        return False

def setup_lcd(i2c_address):
    """إعداد شاشة I2C LCD"""
    global lcd
    try:
        from RPLCD.i2c import CharLCD
        # تهيئة الشاشة
        lcd = CharLCD(i2c_expander="PCF8574", address=i2c_address, port=I2C_PORT,
                      cols=LCD_COLS, rows=LCD_ROWS, dotsize=8,
                      charmap="A02",
                      auto_linebreaks=True,
                      backlight_enabled=True)
        lcd.clear()
        print(f"I2C LCD connected successfully at address {hex(i2c_address)}")
        # عرض الرسالة الترحيبية المطلوبة
        display_message("Welcome", "med limame")
        time.sleep(2.5) # عرض الترحيب لثوانٍ
        display_message("System Ready", "Press Button")
        return lcd # إرجاع الكائن للاستخدام
    except ImportError:
        print("Error: RPLCD library not installed. Install with 'pip install RPLCD'")
    except Exception as e:
        print(f"Error setting up I2C LCD at address {hex(i2c_address)}: {str(e)}")
        print("Check I2C connection and address (use 'i2cdetect -y 1')")
        
    # إذا فشل الإعداد، استخدم محاكاة
    print("Warning: Failed to setup I2C LCD. Running in simulation mode.")
    lcd = None # تأكد من أن lcd هو None في حالة الفشل
    return None # إرجاع None للإشارة إلى الفشل

def control_outputs(red_state, green_state, buzzer_state):
    """التحكم في مؤشرات LED والجرس"""
    if not gpio_available: return
    try:
        import RPi.GPIO as GPIO
        GPIO.output(RED_LED_PIN, red_state)
        GPIO.output(GREEN_LED_PIN, green_state)
        GPIO.output(BUZZER_PIN, buzzer_state)
    except Exception as e:
        print(f"Error controlling GPIO outputs: {e}")

def cleanup():
    """تنظيف الموارد عند الخروج"""
    print("Cleaning up resources...")
    if lcd:
        try:
            lcd.backlight_enabled = False
            lcd.clear()
        except Exception as e:
             print(f"Error cleaning up LCD: {e}")
    if gpio_available:
        try:
            import RPi.GPIO as GPIO
            control_outputs(GPIO.LOW, GPIO.LOW, GPIO.LOW)
            GPIO.cleanup()
            print("GPIO cleaned up")
        except Exception as e:
            print(f"Error cleaning up GPIO: {e}")

def signal_handler(sig, frame):
    """معالج إشارات الإنهاء (Ctrl+C)"""
    print("\nTermination signal received...")
    cleanup()
    sys.exit(0)

# --- سير عمل المعالجة --- 
def capture_image(output_path=TEMP_IMAGE_PATH):
    """التقاط صورة من كاميرا USB باستخدام OpenCV"""
    display_message("Capturing Image", "Please wait...")
    cap = None # تهيئة المتغير
    try:
        cap = cv2.VideoCapture(0) # 0 هو عادةً أول كاميرا USB
        if not cap.isOpened():
            print("Error: Cannot access USB camera! Check connection and permissions.")
            display_message("Camera Error!", "Check connection")
            return False
        
        # إعطاء الكاميرا وقتاً للاستعداد
        time.sleep(1) 
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture image from USB! Check camera.")
            display_message("Capture Failed!", "Try again")
            return False
        
        cv2.imwrite(output_path, frame)
        print(f"Image captured and saved to: {output_path}")
        display_message("Image Captured!", "Processing...")
        time.sleep(0.5) # عرض الرسالة للحظة
        return True
    except Exception as e:
        print(f"Error capturing image: {str(e)}")
        display_message("Capture Error!", str(e)[:LCD_COLS])
        return False
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            print("Camera released.")

def extract_features(image_path):
    """استخراج الميزات من الصورة (مطابقة لتدريب النموذج)"""
    display_message("Processing Image", "Extracting...")
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read image from: {image_path}")
            display_message("Read Image Err!", "Check file")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, FEATURES_EXTRACTION_SIZE)
        features = resized.flatten()
        
        # التأكد من أن عدد الميزات هو 512 (حسب الكود الأصلي)
        expected_features = 512
        if len(features) != expected_features:
            print(f"Warning: Feature count mismatch ({len(features)} vs {expected_features}). Adjusting...")
            if len(features) > expected_features:
                features = features[:expected_features]
            else:
                # استخدام padding بحذر، قد يؤثر على الدقة
                features = np.pad(features, (0, expected_features - len(features)), 'constant', constant_values=0)
        
        print("Features extracted")
        display_message("Image Processed!", "Analyzing...")
        time.sleep(0.5)
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        display_message("Feature Ext Err!", str(e)[:LCD_COLS])
        return None

def predict_pneumonia(features, model_data):
    """التنبؤ باستخدام النموذج المدرب"""
    display_message("Analyzing...", "Please wait...")
    try:
        classifier = model_data['classifier']
        selected_features = model_data['selected_features']
        scaler = model_data['scaler']
        
        # التأكد من أن features هو 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        features_scaled = scaler.transform(features)
        features_selected = features_scaled[:, selected_features]
        
        start_time = time.time()
        prediction = classifier.predict(features_selected)[0]
        probability = classifier.predict_proba(features_selected)[0]
        inference_time = time.time() - start_time
        
        pred_probability = probability[prediction]
        
        print("Analysis complete")
        display_message("Analysis Done!", "Showing Result...")
        time.sleep(0.5)
        return prediction, pred_probability, inference_time
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        display_message("Prediction Error!", str(e)[:LCD_COLS])
        return None, None, None

def display_result(prediction, probability, inference_time):
    """عرض نتيجة التنبؤ على شاشة LCD وتشغيل المخرجات"""
    if prediction is None:
        control_outputs(False, False, False)
        display_message("Error Occurred", "Check Logs")
        return
    
    result_text = "Pneumonia" if prediction == 1 else "Normal"
    prob_percentage = probability * 100
    
    # عرض النتيجة على الشاشة
    display_message(f"Result: {result_text}", f"Conf: {prob_percentage:.1f}%")
    
    # طباعة النتيجة في الطرفية
    print("\n" + "="*50)
    print(f"Result: {result_text}")
    print(f"Confidence: {prob_percentage:.1f}%")
    print(f"Inference Time: {inference_time*1000:.2f} ms")
    print("="*50 + "\n")
    
    # التحكم بالمخرجات
    if prediction == 1: # مصاب
        control_outputs(True, False, True) # LED أحمر + جرس
        threading.Timer(2.0, lambda: control_outputs(True, False, False)).start() # إيقاف الجرس بعد ثانيتين
    else: # طبيعي
        control_outputs(False, True, False) # LED أخضر فقط

def process_image_workflow():
    """سير عمل معالجة الصورة الكامل"""
    global is_processing, button_pressed_flag
    
    is_processing = True
    control_outputs(False, False, False) # إطفاء المخرجات أثناء المعالجة
    
    if not capture_image():
        is_processing = False
        button_pressed_flag = False # إعادة تعيين العلم
        control_outputs(False, False, False)
        display_message("Capture Failed", "Press Button")
        # إضاءة المؤشر الأخضر كإشارة للجاهزية مرة أخرى
        control_outputs(False, True, False)
        return
    
    features = extract_features(TEMP_IMAGE_PATH)
    if features is None:
        is_processing = False
        button_pressed_flag = False # إعادة تعيين العلم
        control_outputs(False, False, False)
        display_message("Process Failed", "Press Button")
        # إضاءة المؤشر الأخضر كإشارة للجاهزية مرة أخرى
        control_outputs(False, True, False)
        return
    
    prediction, probability, inference_time = predict_pneumonia(features, model_data)
    
    display_result(prediction, probability, inference_time)
    
    time.sleep(4) # عرض النتيجة لبعض الوقت
    
    is_processing = False
    button_pressed_flag = False # إعادة تعيين العلم
    display_message("System Ready", "Press Button")
    # إبقاء المؤشر الأخضر/الأحمر مضاءً بعد النتيجة
    if prediction == 0: # Normal
         control_outputs(False, True, False)
    elif prediction == 1: # Pneumonia
         control_outputs(True, False, False) # Red LED remains on (buzzer stopped earlier)
    else: # Error case
         control_outputs(False, True, False) # Default to green LED if error

# --- الدالة الرئيسية --- 
def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description='Pneumonia Detection System (Robust Final Version)')
    parser.add_argument('--model', default=MODEL_PATH, help='Path to the trained model file')
    parser.add_argument('--address', type=lambda x: int(x,0), default=DEFAULT_I2C_ADDRESS, help='I2C address of the LCD (e.g., 0x27 or 0x3F)')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode (no GPIO/LCD)')
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # إعداد GPIO أولاً (قبل الشاشة، للتحكم بالمؤشرات)
    global gpio_available
    if not args.simulate:
        gpio_available = setup_gpio()
        if not gpio_available:
            print("Critical Error: GPIO setup failed. Check wiring/permissions.")
            # لا يمكن المتابعة بدون GPIO للتحكم الأساسي
            # يمكنك إضافة عرض رسالة خطأ على الشاشة إذا كانت الشاشة تعمل
            # setup_lcd(args.address) # محاولة إعداد الشاشة لعرض الخطأ
            # display_message("GPIO Init Fail!", "Check wiring")
            # time.sleep(5)
            return # الخروج إذا فشل إعداد GPIO
    else:
        gpio_available = False
        print("Running in Simulation Mode (GPIO disabled).")

    # إعداد شاشة LCD
    global lcd
    if not args.simulate:
        lcd = setup_lcd(args.address)
        if lcd is None:
            print("Warning: LCD setup failed, continuing without display.")
    else:
        lcd = None

    # تحميل النموذج
    global model_data
    model_data = load_model(args.model)
    if model_data is None:
        # تم عرض الخطأ بالفعل في load_model أو على الطرفية
        control_outputs(True, False, False) # إضاءة المؤشر الأحمر للإشارة لخطأ فادح
        time.sleep(5)
        cleanup()
        return # الخروج إذا فشل تحميل النموذج
    
    # إذا نجح كل شيء، إضاءة المؤشر الأخضر
    control_outputs(False, True, False)
    if lcd:
        # التأكد من عرض رسالة الجاهزية بعد تحميل النموذج
        display_message("System Ready", "Press Button")
    else:
        print("System Ready (No LCD). Press Button to capture and analyze an image.")
        
    print("Press Ctrl+C to exit.")
    
    # حلقة المعالجة الرئيسية
    last_button_state = 1 # افتراض أن الزر غير مضغوط (HIGH بسبب PUD_UP)
    button_press_time = 0
    debounce_time = 0.05 # 50ms debounce
    
    while True:
        global button_pressed_flag # استخدام العلم الذي يتم تعيينه في thread
        
        # --- آلية قراءة الزر (Polling) --- 
        # هذه الطريقة تعمل حتى لو فشل كشف الحافة أو لم تكن هناك صلاحيات كافية
        current_button_state = 1 # القيمة الافتراضية
        if gpio_available:
            try:
                import RPi.GPIO as GPIO
                current_button_state = GPIO.input(BUTTON_PIN)
            except Exception as e:
                print(f"Error reading button state: {e}")
                time.sleep(1) # انتظار قبل المحاولة مرة أخرى
                continue

        # Debouncing Logic
        if current_button_state == 0 and last_button_state == 1: # تم الضغط للتو (Falling edge)
            button_press_time = time.time()
        elif current_button_state == 1 and last_button_state == 0: # تم رفع الضغط للتو (Rising edge)
            if time.time() - button_press_time >= debounce_time:
                if not is_processing:
                    print("Button pressed (Polling detected)")
                    button_pressed_flag = True # تعيين العلم لبدء المعالجة
        
        last_button_state = current_button_state
        # --- نهاية آلية قراءة الزر --- 

        # --- وضع المحاكاة --- 
        if args.simulate:
            try:
                # استخدام طريقة غير معطلة للتحقق من الإدخال (إذا أمكن)
                # أو ببساطة طلب الإدخال بشكل دوري
                if not is_processing:
                    user_input = input("Simulate button press? (y/N/q): ").lower()
                    if user_input == 'y':
                        button_pressed_flag = True
                    elif user_input == 'q':
                        break
            except EOFError:
                 break
        # --- نهاية وضع المحاكاة --- 

        # بدء المعالجة إذا تم الضغط على الزر ولم تكن هناك معالجة جارية
        if button_pressed_flag and not is_processing:
            thread = threading.Thread(target=process_image_workflow)
            thread.daemon = True # السماح للبرنامج بالخروج حتى لو كان الـ thread يعمل
            thread.start()
            # button_pressed_flag is reset inside process_image_workflow
        
        time.sleep(0.02) # دورة قصيرة للتحقق من الزر
    
    cleanup()
    print("Program terminated.")

if __name__ == "__main__":
    main()

