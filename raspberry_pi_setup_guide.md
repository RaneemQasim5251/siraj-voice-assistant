# دليل إعداد سراج على الراسبيري باي
## Siraj Raspberry Pi Setup Guide

### 🎯 المتطلبات الأساسية / Prerequisites

#### العتاد المطلوب / Hardware Requirements:
- **Raspberry Pi 4 أو 5** (4GB RAM أو أكثر مُفضل)
- **كاميرا USB** (للكشف عن الوجوه)
- **ميكروفون USB** (للتفاعل الصوتي)
- **سماعات أو مكبرات صوت**
- **أزرار GPIO** (اختياري للتحكم اليدوي)
- **LED** (اختياري للحالة البصرية)
- **Buzzer** (اختياري للتنبيهات الصوتية)

#### نظام التشغيل / Operating System:
```bash
# تحديث النظام
sudo apt update && sudo apt upgrade -y

# تثبيت Python وأدوات التطوير
sudo apt install python3 python3-pip python3-venv git -y
```

### 🔧 إعداد البيئة / Environment Setup

#### 1. إنشاء المجلد وتفعيل البيئة الافتراضية:
```bash
cd ~/
git clone https://github.com/YourRepo/siraj_gemini.git
cd siraj_gemini

# إنشاء البيئة الافتراضية
python3 -m venv venv_pi
source venv_pi/bin/activate
```

#### 2. تثبيت المكتبات المطلوبة:
```bash
# مكتبات النظام
sudo apt install -y libportaudio2 libportaudio-dev libasound2-dev
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libsndfile1 libsndfile1-dev

# مكتبات Python
pip install --upgrade pip
pip install -r requirements_pi.txt
```

#### 3. إنشاء ملف requirements_pi.txt:
```bash
# إنشاء ملف المتطلبات للراسبيري باي
cat > requirements_pi.txt << 'EOF'
# Core dependencies
google-genai>=0.10.0
python-dotenv>=1.0.0
asyncio-mqtt>=0.11.0

# Audio processing
pyaudio>=0.2.11
soundfile>=0.12.1
sounddevice>=0.4.6
pygame>=2.5.0

# Computer vision
opencv-python>=4.8.0
numpy>=1.24.3

# Data processing
pandas>=2.0.3
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.21.1

# Arabic text processing
arabic-reshaper>=3.0.0
python-bidi>=0.4.2

# GPIO for Raspberry Pi
RPi.GPIO>=0.7.1

# Logging
loguru>=0.7.0
EOF
```

### 🎛️ إعداد GPIO / GPIO Configuration

#### مخطط التوصيل / Wiring Diagram:
```
GPIO Pin 18 → Push Button (مع مقاومة Pull-up)
GPIO Pin 24 → LED (مع مقاومة 330Ω)
GPIO Pin 23 → Buzzer (اختياري)
GND → Ground لجميع المكونات
```

#### تفعيل واجهات النظام:
```bash
# تفعيل الكاميرا وواجهات GPIO
sudo raspi-config
# اختر: Interface Options → Camera → Enable
# اختر: Interface Options → I2C → Enable
# اختر: Interface Options → SPI → Enable
```

### 🔑 إعداد مفاتيح API / API Keys Setup

#### إنشاء ملف المتغيرات البيئية:
```bash
# إنشاء ملف .env
cat > .env << 'EOF'
# Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Audio settings for Pi
PI_AUDIO_DEVICE=default
PI_MIC_DEVICE=default

# GPIO Pin configurations
BUTTON_PIN=18
LED_PIN=24
BUZZER_PIN=23
EOF
```

### 📁 نسخ ملفات البيانات / Data Files

```bash
# التأكد من وجود ملفات البيانات
ls -la *.csv *.json *.pdf

# إذا لم تكن موجودة، قم بنسخها من الجهاز الأساسي
scp user@main_computer:/path/to/data/*.csv ./
scp user@main_computer:/path/to/data/*.json ./
scp user@main_computer:/path/to/data/*.pdf ./
```

### 🚀 تشغيل سراج / Running Siraj

#### التشغيل اليدوي:
```bash
# تفعيل البيئة الافتراضية
source venv_pi/bin/activate

# تشغيل سراج
python3 siraj_pi.py
```

#### التشغيل كخدمة نظام / System Service:
```bash
# إنشاء ملف الخدمة
sudo cat > /etc/systemd/system/siraj.service << 'EOF'
[Unit]
Description=Siraj Arabic Voice Assistant for Riyadh Metro
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/siraj_gemini
Environment=PATH=/home/pi/siraj_gemini/venv_pi/bin
ExecStart=/home/pi/siraj_gemini/venv_pi/bin/python /home/pi/siraj_gemini/siraj_pi.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# تفعيل وبدء الخدمة
sudo systemctl daemon-reload
sudo systemctl enable siraj.service
sudo systemctl start siraj.service

# مراقبة الخدمة
sudo systemctl status siraj.service
sudo journalctl -u siraj.service -f
```

### 🔧 إعداد الصوت / Audio Configuration

#### ضبط أجهزة الصوت:
```bash
# عرض أجهزة الصوت المتاحة
aplay -l
arecord -l

# ضبط جهاز الصوت الافتراضي
sudo cat > /etc/asound.conf << 'EOF'
pcm.!default {
    type asym
    playback.pcm "plughw:1,0"
    capture.pcm "plughw:1,0"
}
EOF

# إعادة تشغيل خدمة الصوت
sudo systemctl restart alsa-state
```

#### اختبار الصوت:
```bash
# تسجيل اختبار (10 ثوان)
arecord -D plughw:1,0 -f cd -t wav -d 10 test.wav

# تشغيل الاختبار
aplay test.wav

# إزالة ملف الاختبار
rm test.wav
```

### 📊 مراقبة الأداء / Performance Monitoring

#### مراقبة استخدام الموارد:
```bash
# مراقبة CPU والذاكرة
htop

# مراقبة سجلات سراج
tail -f /tmp/siraj_pi.log

# مراقبة خدمة النظام
sudo journalctl -u siraj.service --since "5 minutes ago"
```

#### أوامر التحكم في الخدمة:
```bash
# إيقاف الخدمة
sudo systemctl stop siraj.service

# بدء الخدمة
sudo systemctl start siraj.service

# إعادة تشغيل الخدمة
sudo systemctl restart siraj.service

# تعطيل الخدمة من البدء التلقائي
sudo systemctl disable siraj.service
```

### 🛠️ استكشاف الأخطاء / Troubleshooting

#### مشاكل الصوت الشائعة:
```bash
# التحقق من أجهزة الصوت
lsusb | grep -i audio
cat /proc/asound/cards

# إعادة ضبط ALSA
sudo alsa force-reload

# اختبار الميكروفون
arecord -f cd -t wav -d 5 -v mic_test.wav
```

#### مشاكل الكاميرا:
```bash
# التحقق من الكاميرا
lsusb | grep -i camera
v4l2-ctl --list-devices

# اختبار الكاميرا
ffmpeg -f v4l2 -i /dev/video0 -t 3 test.mp4
```

#### مشاكل GPIO:
```bash
# التحقق من حالة GPIO
gpio readall

# اختبار LED
echo "24" > /sys/class/gpio/export
echo "out" > /sys/class/gpio/gpio24/direction
echo "1" > /sys/class/gpio/gpio24/value
echo "0" > /sys/class/gpio/gpio24/value
```

### 🔄 التحديث والصيانة / Updates & Maintenance

#### تحديث سراج:
```bash
# سحب آخر التحديثات
cd ~/siraj_gemini
git pull origin main

# تحديث المكتبات
source venv_pi/bin/activate
pip install --upgrade -r requirements_pi.txt

# إعادة تشغيل الخدمة
sudo systemctl restart siraj.service
```

#### نسخ احتياطي للبيانات:
```bash
# إنشاء نسخة احتياطية
tar -czf siraj_backup_$(date +%Y%m%d).tar.gz \
    ~/siraj_gemini/*.csv \
    ~/siraj_gemini/*.json \
    ~/siraj_gemini/*.pdf \
    ~/siraj_gemini/.env

# استعادة النسخة الاحتياطية
tar -xzf siraj_backup_YYYYMMDD.tar.gz -C ~/siraj_gemini/
```

### 🎮 استخدام سراج / Using Siraj

#### التفاعل الصوتي:
1. **التشغيل التلقائي**: سراج يبدأ الاستماع تلقائياً عند اكتشاف وجه
2. **التحكم اليدوي**: اضغط على الزر المادي للتفعيل/الإلغاء
3. **الحالة البصرية**: LED يظهر حالة النظام
   - **مضيء**: جاهز للاستماع
   - **وامض**: يستمع حالياً
   - **مطفأ**: معطل

#### أمثلة على الاستعلامات:
- "أريد الذهاب إلى مطعم ماكدونالدز"
- "ما هي محطات الخط الأزرق؟"
- "كيف أجدد بطاقة درب؟"
- "أفضل مطاعم بيتزا قريبة من المترو"

### 📞 الدعم الفني / Technical Support

#### سجلات مفيدة للتشخيص:
```bash
# سجل سراج
cat /tmp/siraj_pi.log

# سجل النظام
sudo journalctl -u siraj.service --no-pager

# حالة النظام
systemctl status siraj.service
```

#### معلومات النظام:
```bash
# معلومات الراسبيري باي
cat /proc/cpuinfo | grep "Revision"
vcgencmd measure_temp
vcgencmd get_mem arm && vcgencmd get_mem gpu
```

---

### 🎯 نصائح للأداء الأمثل / Performance Tips

1. **استخدم كارت ذاكرة سريع** (Class 10 أو أفضل)
2. **تأكد من التبريد الجيد** للراسبيري باي
3. **استخدم مصدر طاقة 5V/3A** على الأقل
4. **أغلق التطبيقات غير الضرورية** لتوفير الذاكرة
5. **راقب درجة الحرارة** باستمرار

```bash
# مراقبة درجة الحرارة
watch -n 2 vcgencmd measure_temp
```

هذا الدليل يوفر إعداد كامل لسراج على الراسبيري باي مع جميع الميزات المحسنة! 🚀 