# 🍓 سراج على الراسبيري باي - التعليمات السريعة

## 📦 **ما تم إرساله:**

✅ **الملفات الأساسية:**
- `full_inegration.py` - الكود الرئيسي المحدث
- `requirements_raspberry_pi.txt` - المكتبات المطلوبة
- `setup_pi.sh` - سكريبت الإعداد الآلي
- `arabic_enhancement.py` - معالج النصوص العربية

✅ **ملفات البيانات:**
- `*.csv` - بيانات المطاعم والمحطات
- `*.json` - قواعد البيانات والأسئلة الشائعة
- `*.pdf` - شروط وأحكام بطاقة درب
- `*.geojson` - خرائط المترو

✅ **ملفات النظام:**
- `haarcascade_frontalface_default.xml` - كشف الوجوه

## 🚀 **التشغيل السريع:**

### **1. الإعداد الآلي (مُوصى به):**
```bash
# في الراسبيري باي
cd ~/siraj_gemini
chmod +x setup_pi.sh
./setup_pi.sh
```

### **2. أو الإعداد اليدوي:**
```bash
# إنشاء البيئة الافتراضية
python3 -m venv venv
source venv/bin/activate

# تثبيت المكتبات
pip install -r requirements_raspberry_pi.txt

# إعداد المتغيرات البيئية
echo "GEMINI_API_KEY=your_actual_key_here" > .env
echo "SIRAJ_HEADLESS=true" >> .env

# تشغيل سراج
python3 full_inegration.py
```

## ⚙️ **المميزات الجديدة:**

### **🔄 الكشف التلقائي للوضع:**
- **كشف تلقائي** للراسبيري باي
- **كشف session SSH** - ينتقل للوضع headless تلقائياً
- **كشف عدم توفر DISPLAY** - يحل مشكلة tkinter
- **وضع headless قسري** عبر `SIRAJ_HEADLESS=true`

### **🎛️ تحكم GPIO (اختياري):**
- **GPIO 18** - زر التفعيل/الإلغاء
- **GPIO 24** - LED للحالة
- **GPIO 23** - Buzzer للتنبيهات

### **🔊 صوت محسن:**
- **buffer sizes مخصصة** للراسبيري باي
- **كشف أجهزة الصوت** التلقائي
- **تحسينات PyAudio** للأداء

## 🎮 **أوامر التشغيل:**

```bash
# التشغيل العادي (يكتشف الوضع تلقائياً)
python3 full_inegration.py

# فرض الوضع بدون GUI
python3 full_inegration.py --headless

# عرض أوامر الإعداد
python3 full_inegration.py --pi-setup

# إنشاء خدمة النظام
python3 full_inegration.py --create-service
```

## 🛠️ **إعداد خدمة النظام:**

```bash
# إنشاء وتثبيت الخدمة
python3 full_inegration.py --create-service
sudo cp siraj.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable siraj.service
sudo systemctl start siraj.service

# مراقبة الخدمة
sudo systemctl status siraj.service
sudo journalctl -u siraj.service -f
```

## 🔧 **اختبار النظام:**

```bash
# اختبار الصوت
./test_audio.sh

# اختبار GPIO
./test_gpio.sh

# تشغيل سراج
./run_siraj.sh
```

## 📊 **مراقبة الأداء:**

```bash
# مراقبة السجلات
sudo journalctl -u siraj.service -f

# مراقبة الموارد
htop

# درجة الحرارة
vcgencmd measure_temp
```

## ⚠️ **حل المشاكل:**

### **مشكلة DISPLAY (محلولة):**
```
_tkinter.TclError: no display name and no $DISPLAY environment variable
```
**الحل:** الكود يكتشف هذه المشكلة تلقائياً وينتقل للوضع headless

### **مشكلة الصوت:**
```bash
# تحقق من أجهزة الصوت
aplay -l
arecord -l

# اختبار الميكروفون
arecord -f cd -t wav -d 5 test.wav
aplay test.wav
```

### **مشكلة GPIO:**
```bash
# تحقق من GPIO
gpio readall

# اختبار LED
echo "24" > /sys/class/gpio/export
echo "out" > /sys/class/gpio/gpio24/direction
echo "1" > /sys/class/gpio/gpio24/value
```

## 📱 **التفاعل مع سراج:**

- **📢 تحدث مباشرة** - سراج يسمع دائماً
- **🔘 اضغط الزر المادي** للتحكم (إذا متوفر)
- **💡 راقب LED** لمعرفة الحالة
- **🎤 جرب قول:** "أريد الذهاب إلى مطعم ماكدونالدز"

## 🎯 **نصائح الأداء:**

- ✅ **استخدم كارت ذاكرة سريع** (Class 10+)
- ✅ **تأكد من التبريد الجيد**
- ✅ **مصدر طاقة 5V/3A** على الأقل
- ✅ **أغلق التطبيقات غير الضرورية**

---

## 🎉 **جاهز للتشغيل!**

الآن سراج محسن ومجهز للعمل على الراسبيري باي مع حل جميع مشاكل العرض والصوت! 🚀 