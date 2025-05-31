#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تطبيق ويب متكامل لنظام كشف الالتهاب الرئوي باستخدام Flask
يتضمن واجهة ويب للتحكم بالكاميرا وعرض النتائج والتحكم بالمؤشرات

الاستخدام:
    python3 app.py
"""

import os
import time
import threading
import numpy as np
import cv2
import joblib
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
import base64
from datetime import datetime

# --- تكوين --- 
MODEL_PATH = "pneumonia_model.joblib"
TEMP_IMAGE_PATH = "static/captured_image.jpg"
FEATURES_EXTRACTION_SIZE = (224, 224)  # حجم الصورة للمعالجة
UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"

# تكوين GPIO (حسب الدليل والتوصيلات)
BUTTON_PIN = 23  # GPIO للزر (مقاومة سحب خارجية)
RED_LED_PIN = 17 # GPIO للمؤشر الأحمر (نتيجة إيجابية)
GREEN_LED_PIN = 27 # GPIO للمؤشر الأخضر (نتيجة سلبية)
BUZZER_PIN = 22  # GPIO للجرس (مع النتيجة الإيجابية)

# متغيرات عامة
model_data = None
is_processing = False
gpio_available = False # لتتبع توفر مكتبة GPIO
camera = None  # كائن الكاميرا العالمي

# إنشاء المجلدات اللازمة
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# إنشاء تطبيق Flask
app = Flask(__name__)

# --- دوال المساعدة --- 
def load_model(model_path):
    """تحميل النموذج المدرب"""
    print("Loading model...")
    try:
        model_data = joblib.load(model_path)
        accuracy = model_data.get("accuracy", "N/A")
        print(f"Model loaded successfully (Accuracy: {accuracy})")
        return model_data
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

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

def cleanup_gpio():
    """تنظيف الموارد عند الخروج"""
    if gpio_available:
        try:
            import RPi.GPIO as GPIO
            control_outputs(False, False, False)
            GPIO.cleanup()
            print("GPIO cleaned up")
        except Exception as e:
            print(f"Error cleaning up GPIO: {e}")

def capture_image_from_camera():
    """التقاط صورة من كاميرا USB باستخدام OpenCV"""
    global camera
    cap = None
    try:
        # إذا كانت الكاميرا مفتوحة بالفعل، أغلقها وأعد فتحها
        if camera is not None:
            camera.release()
            camera = None
            
        cap = cv2.VideoCapture(0)  # 0 هو عادةً أول كاميرا USB
        if not cap.isOpened():
            print("Error: Cannot access USB camera! Check connection and permissions.")
            return False, "فشل الوصول إلى الكاميرا. تأكد من التوصيل والصلاحيات."
        
        # إعطاء الكاميرا وقتاً للاستعداد
        time.sleep(1) 
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture image from USB! Check camera.")
            return False, "فشل التقاط الصورة. تأكد من عمل الكاميرا."
        
        # إنشاء اسم فريد للصورة
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(UPLOAD_FOLDER, f"captured_{timestamp}.jpg")
        
        cv2.imwrite(image_path, frame)
        print(f"Image captured and saved to: {image_path}")
        
        # نسخ الصورة أيضاً إلى المسار المؤقت للمعالجة
        cv2.imwrite(TEMP_IMAGE_PATH, frame)
        
        return True, image_path
    except Exception as e:
        error_msg = str(e)
        print(f"Error capturing image: {error_msg}")
        return False, f"خطأ في التقاط الصورة: {error_msg}"
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            print("Camera released.")

def extract_features(image_path):
    """استخراج الميزات من الصورة (مطابقة لتدريب النموذج)"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read image from: {image_path}")
            return None, "فشل قراءة الصورة. تأكد من وجود الملف."
        
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
        return features, None
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error extracting features: {error_msg}")
        return None, f"خطأ في استخراج الميزات: {error_msg}"

def predict_pneumonia(features, model_data):
    """التنبؤ باستخدام النموذج المدرب"""
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
        
        # تحديد النتيجة
        result_text = "Pneumonia" if prediction == 1 else "Normal"
        result_text_ar = "التهاب رئوي" if prediction == 1 else "طبيعي"
        prob_percentage = pred_probability * 100
        
        # التحكم بالمخرجات
        if prediction == 1: # مصاب
            control_outputs(True, False, True) # LED أحمر + جرس
            threading.Timer(2.0, lambda: control_outputs(True, False, False)).start() # إيقاف الجرس بعد ثانيتين
        else: # طبيعي
            control_outputs(False, True, False) # LED أخضر فقط
        
        # إنشاء قاموس النتائج
        result = {
            "prediction": int(prediction),
            "result_text": result_text,
            "result_text_ar": result_text_ar,
            "probability": float(pred_probability),
            "probability_percentage": float(prob_percentage),
            "inference_time_ms": float(inference_time * 1000)
        }
        
        return result, None
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error during prediction: {error_msg}")
        return None, f"خطأ في التحليل: {error_msg}"

def process_image_workflow(image_path):
    """سير عمل معالجة الصورة الكامل"""
    global is_processing
    
    is_processing = True
    control_outputs(False, False, False) # إطفاء المخرجات أثناء المعالجة
    
    # استخراج الميزات
    features, error = extract_features(image_path)
    if features is None:
        is_processing = False
        control_outputs(False, True, False) # إعادة المؤشر الأخضر للجاهزية
        return None, error
    
    # التنبؤ
    result, error = predict_pneumonia(features, model_data)
    if result is None:
        is_processing = False
        control_outputs(False, True, False) # إعادة المؤشر الأخضر للجاهزية
        return None, error
    
    # إضافة مسار الصورة للنتيجة
    result["image_path"] = image_path
    
    is_processing = False
    return result, None

def generate_camera_frames():
    """توليد إطارات الكاميرا للعرض المباشر"""
    global camera
    
    # إذا كانت الكاميرا مفتوحة بالفعل، أغلقها وأعد فتحها
    if camera is not None:
        camera.release()
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Cannot access camera for streaming")
        return
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # تحويل الإطار إلى JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                
                # إرسال الإطار كجزء من استجابة متعددة الأجزاء
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Error in camera stream: {e}")
    finally:
        if camera is not None:
            camera.release()

# --- مسارات Flask --- 
@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """مسار لعرض الكاميرا المباشر"""
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    """التقاط صورة من الكاميرا"""
    if is_processing:
        return jsonify({"success": False, "error": "جاري معالجة صورة أخرى. يرجى الانتظار."})
    
    success, result = capture_image_from_camera()
    if not success:
        return jsonify({"success": False, "error": result})
    
    # معالجة الصورة وتحليلها
    analysis_result, error = process_image_workflow(result)
    if analysis_result is None:
        return jsonify({"success": False, "error": error})
    
    # إرجاع النتيجة
    return jsonify({
        "success": True, 
        "result": analysis_result,
        "image_url": "/" + result  # إضافة / للحصول على مسار URL صحيح
    })

@app.route('/upload', methods=['POST'])
def upload():
    """تحميل صورة من المستخدم"""
    if is_processing:
        return jsonify({"success": False, "error": "جاري معالجة صورة أخرى. يرجى الانتظار."})
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "لم يتم تحديد ملف."})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "لم يتم اختيار ملف."})
    
    if file:
        # إنشاء اسم فريد للصورة
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # نسخ الصورة أيضاً إلى المسار المؤقت للمعالجة
        file.seek(0)
        with open(TEMP_IMAGE_PATH, 'wb') as f:
            f.write(file.read())
        
        # معالجة الصورة وتحليلها
        analysis_result, error = process_image_workflow(filepath)
        if analysis_result is None:
            return jsonify({"success": False, "error": error})
        
        # إرجاع النتيجة
        return jsonify({
            "success": True, 
            "result": analysis_result,
            "image_url": "/" + filepath  # إضافة / للحصول على مسار URL صحيح
        })

@app.route('/status')
def status():
    """التحقق من حالة النظام"""
    status_info = {
        "model_loaded": model_data is not None,
        "gpio_available": gpio_available,
        "is_processing": is_processing,
        "camera_connected": False
    }
    
    # التحقق من اتصال الكاميرا
    try:
        cap = cv2.VideoCapture(0)
        status_info["camera_connected"] = cap.isOpened()
        if cap.isOpened():
            cap.release()
    except Exception:
        pass
    
    return jsonify(status_info)

# --- تهيئة التطبيق --- 
def initialize_app():
    """تهيئة التطبيق وتحميل النموذج وإعداد GPIO"""
    global model_data
    
    # تحميل النموذج
    model_data = load_model(MODEL_PATH)
    if model_data is None:
        print("Warning: Failed to load model. Predictions will not work.")
    
    # إعداد GPIO
    setup_gpio()
    
    # إضاءة المؤشر الأخضر للإشارة إلى جاهزية النظام
    control_outputs(False, True, False)

# --- تشغيل التطبيق --- 
if __name__ == '__main__':
    try:
        initialize_app()
        # تشغيل التطبيق على جميع الواجهات (0.0.0.0) ليكون متاحاً من أجهزة أخرى على الشبكة
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("Application terminated by user")
    finally:
        cleanup_gpio()
        if camera is not None:
            camera.release()
