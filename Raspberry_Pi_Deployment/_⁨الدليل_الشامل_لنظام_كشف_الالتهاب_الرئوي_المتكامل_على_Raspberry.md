# الدليل الشامل لنظام كشف الالتهاب الرئوي المتكامل على Raspberry Pi

## مقدمة

أهلاً بك في الدليل الشامل لنظام كشف الالتهاب الرئوي المتكامل على Raspberry Pi. هذا النظام يجمع بين قوة الذكاء الاصطناعي وسهولة الاستخدام، ويوفر طريقتين للتشغيل:

1. **واجهة الويب**: تتيح لك التحكم بالنظام من أي جهاز متصل بالشبكة المحلية
2. **المكونات المادية**: تتيح لك التشغيل المباشر عبر الزر والشاشة والمؤشرات الضوئية

## المكونات المستخدمة

* Raspberry Pi 3 Model B (2015)
* كاميرا مايكروسوفت (USB)
* شاشة LCD 16x2 مع واجهة I2C (PCF8574)
* أسلاك توصيل
* مقاومات (220 أوم لمؤشرات LED و 10 كيلو أوم للزر)
* جرس إنذار (Buzzer)
* مؤشرات ضوئية (LEDs) (أحمر وأخضر)
* زر ضغط (Button)
* بطاقة ذاكرة MicroSD (يفضل 8 جيجابايت أو أكثر، فئة 10)
* مصدر طاقة مناسب للـ Raspberry Pi (5V, 2.5A على الأقل)

## الجزء الأول: إعداد النظام الأساسي

### 1. تجهيز بطاقة الذاكرة (SD Card) ونظام التشغيل

1. **تنزيل Raspberry Pi Imager:**
   * على حاسوبك، اذهب إلى [https://www.raspberrypi.com/software/](https://www.raspberrypi.com/software/)
   * قم بتنزيل وتثبيت Raspberry Pi Imager.

2. **حرق النظام على بطاقة الذاكرة:**
   * أدخل بطاقة MicroSD في قارئ بطاقات.
   * شغل Raspberry Pi Imager.
   * "CHOOSE OS" -> "Raspberry Pi OS (other)" -> "Raspberry Pi OS Lite (64-bit)" أو "(32-bit)".
   * "CHOOSE STORAGE" -> اختر بطاقتك.
   * **انقر أيقونة الإعدادات (الترس):**
     * **Enable SSH:** فعّله واختر "Use password authentication".
     * **Set username and password:** عيّن اسم مستخدم (مثل `pi`) وكلمة مرور قوية. **احفظها جيداً.**
     * **Configure wireless LAN:** أدخل اسم شبكة الواي فاي (SSID) وكلمة المرور.
   * انقر "SAVE".
   * انقر "WRITE" وانتظر اكتمال العملية.

3. **إخراج البطاقة:** أخرج البطاقة بأمان.

### 2. التشغيل الأول والاتصال بالـ Raspberry Pi

1. **إدخال البطاقة وتوصيل الطاقة:**
   * أدخل البطاقة في Raspberry Pi.
   * وصل مصدر الطاقة وانتظر دقيقة أو دقيقتين.

2. **الاتصال عبر SSH من ويندوز:**
   * استخدم PowerShell أو PuTTY.
   * **PowerShell:** `ssh pi@raspberrypi.local` (استبدل `pi` و `raspberrypi.local` إذا غيرتهما).
   * **PuTTY:** أدخل `raspberrypi.local` في Host Name، Port 22، Type SSH، ثم Open.
   * أدخل اسم المستخدم وكلمة المرور عند الطلب.

3. **تحديث النظام:**
   * بمجرد تسجيل الدخول، نفذ:
     ```bash
     sudo apt update
     sudo apt upgrade -y
     ```

### 3. تثبيت الحزم والمكتبات اللازمة

1. **تثبيت الأدوات الأساسية ومكتبات OpenCV:**
   ```bash
   sudo apt install -y python3-pip python3-dev python3-numpy libatlas-base-dev libopenjp2-7 libtiff5 build-essential cmake pkg-config libjpeg-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libcanberra-gtk* libatlas-base-dev gfortran python3-opencv git i2c-tools
   ```

2. **تثبيت مكتبات بايثون الأساسية:**
   ```bash
   pip3 install numpy scikit-learn joblib flask --break-system-packages
   ```

3. **تثبيت مكتبات التحكم بالـ GPIO وشاشة I2C LCD:**
   * **مكتبة GPIO:**
     ```bash
     pip3 install RPi.GPIO --break-system-packages
     ```
   * **مكتبة شاشة I2C LCD (RPLCD):**
     ```bash
     pip3 install RPLCD --break-system-packages
     ```

4. **تفعيل واجهة I2C:**
   * نفذ الأمر:
     ```bash
     sudo raspi-config
     ```
   * اذهب إلى `Interface Options` -> `I2C` -> `<Yes>` -> `<Ok>` -> `<Finish>`.
   * أعد تشغيل الجهاز إذا طُلب منك.
   * بعد إعادة التشغيل والاتصال مجدداً عبر SSH، قم بتوصيل شاشة LCD (كما في الخطوة التالية) وتحقق من الاتصال:
     ```bash
     sudo i2cdetect -y 1
     ```
   * **هام:** يجب أن ترى عنوان الشاشة في الجدول. العنوان الأكثر شيوعاً لشاشات LCD I2C PCF8574 هو `27` (يكتب 0x27). قد يكون أيضاً `3f` (يكتب 0x3F) أو عنواناً آخر حسب الشاشة. **لاحظ العنوان الذي يظهر لديك.**

## الجزء الثاني: توصيل المكونات الإلكترونية

**تحذير:** افصل مصدر الطاقة عن Raspberry Pi قبل التوصيل.

1. **توصيل شاشة LCD I2C (PCF8574):**
   * `VCC` (الشاشة) ← `5V` (Pi - Pin 2 أو Pin 4)  **(ملاحظة: بعض شاشات LCD تعمل بشكل أفضل مع 5V)**
   * `GND` (الشاشة) ← `GND` (Pi - Pin 6 أو أي منفذ أرضي آخر)
   * `SDA` (الشاشة) ← `GPIO 2 (SDA)` (Pi - Pin 3)
   * `SCL` (الشاشة) ← `GPIO 3 (SCL)` (Pi - Pin 5)

2. **توصيل مؤشر LED أحمر (للنتيجة الإيجابية):**
   * الطرف الموجب (Anode) ← مقاومة 220 أوم ← `GPIO 17` (Pi - Pin 11)
   * الطرف السالب (Cathode) ← `GND` (Pi - Pin 9 أو أي منفذ أرضي)

3. **توصيل مؤشر LED أخضر (للنتيجة السلبية):**
   * الطرف الموجب (Anode) ← مقاومة 220 أوم ← `GPIO 27` (Pi - Pin 13)
   * الطرف السالب (Cathode) ← `GND` (Pi - Pin 14 أو أي منفذ أرضي)

4. **توصيل جرس الإنذار (Buzzer):**
   * الطرف الموجب (+) ← `GPIO 22` (Pi - Pin 15)
   * الطرف السالب (-) ← `GND` (Pi - Pin 20 أو أي منفذ أرضي)

5. **توصيل زر الضغط (Button):**
   * أحد أطراف الزر ← `GPIO 23` (Pi - Pin 16)
   * الطرف الآخر للزر ← `GND` (Pi - Pin 25 أو أي منفذ أرضي)
   * **مقاومة السحب Pull-up:** وصل مقاومة 10 كيلو أوم بين `GPIO 23` (Pin 16) و `3.3V` (Pi - Pin 1 أو Pin 17).

6. **توصيل الكاميرا:**
   * وصل كاميرا المايكروسوفت USB بأحد منافذ USB.

7. **المراجعة النهائية للتوصيلات:** تأكد من صحة وأمان جميع التوصيلات.

## الجزء الثالث: تثبيت وتشغيل النظام

### 1. نقل ملفات المشروع

1. **إنشاء مجلد للمشروع على Raspberry Pi:**
   ```bash
   mkdir -p ~/pneumonia_detection
   cd ~/pneumonia_detection
   ```

2. **نقل الملفات:**
   * استخدم `scp` أو WinSCP لنقل الملفات التالية إلى مجلد `~/pneumonia_detection` على Raspberry Pi:
     * `inference_lcd_i2c_robust_final.py` (كود التشغيل المستقل)
     * `pneumonia_model.joblib` (ملف النموذج)

3. **إنشاء مجلد لتطبيق الويب:**
   ```bash
   mkdir -p ~/pneumonia_webapp/templates
   mkdir -p ~/pneumonia_webapp/static
   ```

4. **نقل ملفات تطبيق الويب:**
   * استخدم `scp` أو WinSCP لنقل الملفات التالية:
     * `app.py` إلى مجلد `~/pneumonia_webapp`
     * `index.html` إلى مجلد `~/pneumonia_webapp/templates`
     * `style.css` إلى مجلد `~/pneumonia_webapp/static`
     * انسخ `pneumonia_model.joblib` إلى مجلد `~/pneumonia_webapp`

### 2. تشغيل النظام المستقل (مع الشاشة والزر)

1. **تشغيل الكود المستقل:**
   ```bash
   cd ~/pneumonia_detection
   sudo python3 inference_lcd_i2c_robust_final.py
   ```

2. **ملاحظات هامة حول الشاشة LCD:**
   * **إذا لم تظهر أي كتابة على الشاشة:** قم بتعديل التباين عن طريق تدوير البرغي الأزرق الصغير الموجود على ظهر لوحة I2C الملحقة بالشاشة ببطء شديد أثناء تشغيل الكود. هذه هي المشكلة الأكثر شيوعاً.
   * **إذا كان عنوان I2C للشاشة مختلفاً عن 0x27:** استخدم الخيار `--address` لتحديد العنوان الصحيح:
     ```bash
     sudo python3 inference_lcd_i2c_robust_final.py --address 0x3F
     ```

3. **اختبار النظام المستقل:**
   * عند بدء التشغيل، ستظهر رسالة ترحيبية "Welcome med limame" على الشاشة.
   * بعد ذلك، ستظهر رسالة "System Ready" و "Press Button".
   * اضغط على الزر المادي لالتقاط صورة وتحليلها.
   * ستعرض الشاشة حالة المعالجة والنتيجة النهائية.
   * سيضيء المؤشر الأخضر للنتيجة الطبيعية، أو المؤشر الأحمر مع تشغيل الجرس للنتيجة الإيجابية (التهاب رئوي).

### 3. تشغيل تطبيق الويب

1. **تشغيل خادم الويب:**
   ```bash
   cd ~/pneumonia_webapp
   sudo python3 app.py
   ```

2. **الوصول إلى واجهة الويب:**
   * افتح متصفح الويب على أي جهاز متصل بنفس الشبكة المحلية.
   * أدخل عنوان IP الخاص بـ Raspberry Pi متبوعاً بالمنفذ 5000:
     ```
     http://raspberrypi.local:5000
     ```
     أو
     ```
     http://<عنوان-IP>:5000
     ```
     (يمكنك معرفة عنوان IP باستخدام الأمر `hostname -I` على Raspberry Pi)

3. **استخدام واجهة الويب:**
   * ستظهر واجهة الويب مع عرض مباشر للكاميرا.
   * يمكنك النقر على زر "التقاط صورة" لالتقاط صورة من الكاميرا وتحليلها.
   * يمكنك أيضاً تحميل صورة من جهازك لتحليلها.
   * ستظهر النتيجة مع نسبة الثقة ووقت المعالجة.
   * سيتم أيضاً تشغيل المؤشرات الضوئية والجرس على Raspberry Pi بناءً على النتيجة.

## الجزء الرابع: إعداد التشغيل التلقائي عند بدء التشغيل

### 1. إعداد التشغيل التلقائي للنظام المستقل (مع الشاشة والزر)

1. **إنشاء ملف خدمة Systemd:**
   ```bash
   sudo nano /etc/systemd/system/pneumonia-detector.service
   ```

2. **لصق محتوى الخدمة:**
   ```ini
   [Unit]
   Description=Pneumonia Detection Service (LCD & Button)
   After=network.target multi-user.target
   
   [Service]
   ExecStart=/usr/bin/sudo /usr/bin/python3 /home/pi/pneumonia_detection/inference_lcd_i2c_robust_final.py
   WorkingDirectory=/home/pi/pneumonia_detection
   StandardOutput=inherit
   StandardError=inherit
   Restart=always
   User=pi
   
   [Install]
   WantedBy=multi-user.target
   ```
   * **ملاحظة هامة:** إذا كان عنوان I2C لشاشتك **ليس** `0x27`، أضف `--address 0x3F` (أو العنوان الصحيح) في نهاية سطر `ExecStart`.
   * **ملاحظة:** لاحظ استخدام `sudo` في سطر `ExecStart` لضمان صلاحيات GPIO الكاملة.

3. **حفظ الملف والخروج:**
   * اضغط `Ctrl+X`، ثم `Y`، ثم `Enter`.

### 2. إعداد التشغيل التلقائي لتطبيق الويب

1. **إنشاء ملف خدمة Systemd:**
   ```bash
   sudo nano /etc/systemd/system/pneumonia-webapp.service
   ```

2. **لصق محتوى الخدمة:**
   ```ini
   [Unit]
   Description=Pneumonia Detection Web Application
   After=network.target multi-user.target
   
   [Service]
   ExecStart=/usr/bin/sudo /usr/bin/python3 /home/pi/pneumonia_webapp/app.py
   WorkingDirectory=/home/pi/pneumonia_webapp
   StandardOutput=inherit
   StandardError=inherit
   Restart=always
   User=pi
   
   [Install]
   WantedBy=multi-user.target
   ```

3. **حفظ الملف والخروج:**
   * اضغط `Ctrl+X`، ثم `Y`، ثم `Enter`.

### 3. تفعيل وبدء الخدمات

1. **أعد تحميل إعدادات systemd:**
   ```bash
   sudo systemctl daemon-reload
   ```

2. **اختر أحد الخيارين:**
   * **لتشغيل النظام المستقل فقط (مع الشاشة والزر):**
     ```bash
     sudo systemctl enable pneumonia-detector.service
     sudo systemctl start pneumonia-detector.service
     ```
   * **لتشغيل تطبيق الويب فقط:**
     ```bash
     sudo systemctl enable pneumonia-webapp.service
     sudo systemctl start pneumonia-webapp.service
     ```
   * **لتشغيل كلا النظامين معاً:**
     ```bash
     sudo systemctl enable pneumonia-detector.service pneumonia-webapp.service
     sudo systemctl start pneumonia-detector.service pneumonia-webapp.service
     ```

3. **التحقق من حالة الخدمات:**
   ```bash
   sudo systemctl status pneumonia-detector.service
   sudo systemctl status pneumonia-webapp.service
   ```

## الجزء الخامس: استكشاف الأخطاء وإصلاحها

### 1. مشاكل الشاشة LCD

1. **الشاشة لا تعرض أي شيء:**
   * **الحل الأكثر شيوعاً:** قم بتعديل التباين عن طريق تدوير البرغي الأزرق الصغير الموجود على ظهر لوحة I2C الملحقة بالشاشة ببطء شديد أثناء تشغيل الكود.
   * تأكد من توصيل الشاشة بشكل صحيح (VCC, GND, SDA, SCL).
   * تأكد من توصيل VCC بمنفذ 5V وليس 3.3V.
   * تحقق من عنوان I2C باستخدام `sudo i2cdetect -y 1` وتأكد من استخدام العنوان الصحيح في الكود.

2. **أحرف غريبة على الشاشة:**
   * قد تكون مشكلة في إعدادات الشاشة. جرب استخدام خرائط أحرف مختلفة في الكود (مثل "A00" بدلاً من "A02").

### 2. مشاكل GPIO

1. **المؤشرات الضوئية أو الجرس لا تعمل:**
   * تأكد من توصيل المكونات بشكل صحيح.
   * تأكد من استخدام المقاومات المناسبة.
   * تأكد من تشغيل الكود باستخدام `sudo`.

2. **الزر لا يستجيب:**
   * تأكد من توصيل الزر بشكل صحيح.
   * تأكد من استخدام مقاومة السحب (10 كيلو أوم).
   * تأكد من تشغيل الكود باستخدام `sudo`.

### 3. مشاكل الكاميرا

1. **الكاميرا لا تعمل:**
   * تأكد من توصيل الكاميرا بشكل صحيح بمنفذ USB.
   * تحقق من أن الكاميرا معترف بها في النظام باستخدام `lsusb`.
   * جرب منفذ USB آخر.
   * تأكد من تثبيت `python3-opencv`.

### 4. مشاكل تطبيق الويب

1. **لا يمكن الوصول إلى واجهة الويب:**
   * تأكد من أن تطبيق الويب يعمل باستخدام `sudo systemctl status pneumonia-webapp.service`.
   * تأكد من استخدام عنوان IP الصحيح والمنفذ 5000.
   * تأكد من أن جهازك متصل بنفس الشبكة المحلية مع Raspberry Pi.
   * تحقق من إعدادات جدار الحماية باستخدام `sudo ufw status` وتأكد من أن المنفذ 5000 مفتوح.

2. **بث الكاميرا لا يظهر في واجهة الويب:**
   * تأكد من أن الكاميرا متصلة ومعترف بها.
   * أعد تشغيل تطبيق الويب باستخدام `sudo systemctl restart pneumonia-webapp.service`.

## الجزء السادس: الاستخدام اليومي

### 1. استخدام النظام المستقل (مع الشاشة والزر)

1. **بدء التشغيل:**
   * عند توصيل الطاقة، سيبدأ النظام تلقائياً إذا تم إعداد التشغيل التلقائي.
   * ستظهر رسالة ترحيبية "Welcome med limame" على الشاشة.
   * بعد ذلك، ستظهر رسالة "System Ready" و "Press Button".

2. **التقاط وتحليل صورة:**
   * ضع الصورة المراد تحليلها أمام الكاميرا.
   * اضغط على الزر المادي.
   * ستعرض الشاشة حالة المعالجة والنتيجة النهائية.
   * سيضيء المؤشر الأخضر للنتيجة الطبيعية، أو المؤشر الأحمر مع تشغيل الجرس للنتيجة الإيجابية (التهاب رئوي).

### 2. استخدام واجهة الويب

1. **الوصول إلى واجهة الويب:**
   * افتح متصفح الويب على أي جهاز متصل بنفس الشبكة المحلية.
   * أدخل عنوان IP الخاص بـ Raspberry Pi متبوعاً بالمنفذ 5000:
     ```
     http://raspberrypi.local:5000
     ```
     أو
     ```
     http://<عنوان-IP>:5000
     ```

2. **التقاط صورة من الكاميرا:**
   * انقر على زر "التقاط صورة".
   * انتظر حتى تكتمل المعالجة.
   * ستظهر النتيجة مع نسبة الثقة ووقت المعالجة.

3. **تحميل صورة من جهازك:**
   * انقر على "اختيار ملف" واختر صورة من جهازك.
   * انتظر حتى تكتمل المعالجة.
   * ستظهر النتيجة مع نسبة الثقة ووقت المعالجة.

## الخاتمة

تهانينا! لقد قمت الآن بإعداد نظام كشف الالتهاب الرئوي المتكامل على Raspberry Pi. يمكنك استخدام النظام إما عبر المكونات المادية (الشاشة والزر) أو عبر واجهة الويب من أي جهاز متصل بالشبكة المحلية.

إذا واجهت أي مشاكل، يرجى الرجوع إلى قسم "استكشاف الأخطاء وإصلاحها" أو التواصل للحصول على المساعدة.
