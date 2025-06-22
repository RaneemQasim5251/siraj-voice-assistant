# 🎙️ Siraj - Arabic Voice Assistant for Riyadh Metro

**سراج** هو مساعد صوتي ذكي باللغة العربية مخصص لمترو الرياض، يستخدم تقنيات الذكاء الاصطناعي المتقدمة لتقديم خدمات التنقل والمطاعم.

## ✨ المميزات

### 🎤 **التفاعل الصوتي المتقدم**
- **Gemini Live API**: تفاعل صوتي حي ومباشر
- **كشف الوجوه**: ترحيب تلقائي عند اكتشاف المستخدم
- **متعدد اللغات**: دعم العربية، الإنجليزية، الأردية، والمزيد

### 🗺️ **خدمات التنقل**
- **مسارات المترو**: إرشاد دقيق لجميع خطوط مترو الرياض
- **خرائط تفاعلية**: عرض المسارات بصرياً مع Folium
- **معلومات المحطات**: بيانات شاملة لجميع محطات المترو

### 🍽️ **دليل المطاعم**
- **قاعدة بيانات ضخمة**: أكثر من 2600 مطعم في الرياض
- **البحث الذكي**: بحث متقدم بالاسم، النوع، والموقع
- **التقييمات**: عرض تقييمات المطاعم وآراء المستخدمين
- **المسارات**: إرشاد من المترو إلى المطعم المطلوب

### 📱 **واجهات متعددة**
- **GUI Mode**: واجهة رسومية مع فيديو تفاعلي
- **Headless Mode**: وضع صوتي للخوادم والراسبيري باي
- **تبديل تلقائي**: بين فيديو الكلام والصمت

### 🤖 **تقنيات ذكية**
- **RAG System**: استرجاع معلومات من ملفات PDF
- **Face Detection**: OpenCV + YOLO للكشف المتقدم
- **Arabic NLP**: معالجة النصوص العربية المحسنة

## 🛠️ متطلبات النظام

### **للاستخدام العادي:**
```
Python 3.8+
Microphone & Speakers
Internet Connection
Gemini API Key
```

### **للراسبيري باي:**
```
Raspberry Pi 4 (موصى به)
Camera Module (اختياري)
USB Microphone
Speakers أو سماعات
```

## 📦 التثبيت

### **1. استنساخ المشروع:**
```bash
git clone https://github.com/your-username/siraj-voice-assistant.git
cd siraj-voice-assistant
```

### **2. إنشاء البيئة الافتراضية:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# أو
venv\Scripts\activate     # Windows
```

### **3. تثبيت المتطلبات:**
```bash
# للاستخدام العادي
pip install -r requirements.txt

# للراسبيري باي
pip install -r requirements_raspberry_pi.txt
```

### **4. إعداد مفتاح API:**
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### **5. تحميل البيانات:**
قم بتحميل ملفات البيانات المطلوبة:
- `restaurants2_neighborhood_stations_paths_15.csv`
- `metro-stations-in-riyadh-*.geojson`
- `metro-lines-in-riyadh-*.geojson`
- `Darb card terms and conditions.pdf`

## 🚀 الاستخدام

### **الوضع العادي (مع GUI):**
```bash
python3 full_inegration.py
```

### **الوضع الصوتي (بدون GUI):**
```bash
export SIRAJ_HEADLESS=true
python3 full_inegration.py
```

### **للراسبيري باي:**
```bash
# تشغيل مع شاشة
export DISPLAY=:0
python3 full_inegration.py

# تشغيل بدون شاشة
export SIRAJ_HEADLESS=true
python3 full_inegration.py
```

## 🎯 أمثلة الاستخدام

### **البحث عن مطعم:**
```
المستخدم: "أبغى أروح مطعم ماكدونالدز"
سراج: "المسار إلى ماكدونالدز: اركب المترو من المحطة الحالية..."
```

### **معلومات بطاقة درب:**
```
المستخدم: "متى تنتهي صلاحية بطاقة درب؟"
سراج: "بطاقة درب صالحة لمدة خمس سنوات من تاريخ الإصدار..."
```

### **استعلامات متعددة اللغات:**
```
User: "When does the Darb card expire?"
Siraj: "The Darb card is valid for five years from the date of issuance..."
```

## 📁 هيكل المشروع

```
siraj-voice-assistant/
├── full_inegration.py          # الملف الرئيسي
├── arabic_enhancement.py       # معالجة النصوص العربية
├── requirements.txt            # متطلبات عامة
├── requirements_raspberry_pi.txt # متطلبات الراسبيري باي
├── .env.example               # مثال على ملف البيئة
├── data/                      # ملفات البيانات
│   ├── faq.json
│   ├── metro_data.json
│   └── rules.json
├── templates/                 # قوالب الويب
│   └── index.html
├── models/                    # نماذج YOLO
│   └── yolov8n.pt
├── docs/                      # الوثائق
│   ├── RASPBERRY_PI_SETUP.md
│   ├── API_USAGE.md
│   └── TROUBLESHOOTING.md
└── scripts/                   # سكريبتات المساعدة
    ├── setup_pi.sh
    └── install_dependencies.sh
```

## 🎛️ الإعدادات

### **متغيرات البيئة:**
```bash
GEMINI_API_KEY=your_gemini_api_key
SIRAJ_HEADLESS=false          # true للوضع الصوتي
DISPLAY=:0                    # للراسبيري باي مع شاشة
```

### **إعدادات الصوت:**
```python
# في full_inegration.py
SEND_SAMPLE_RATE = 16000      # معدل الميكروفون
RECEIVE_SAMPLE_RATE = 24000   # معدل المكبر
CHUNK_SIZE = 1024             # حجم البيانات (Pi)
CHUNK_SIZE = 2048             # حجم البيانات (Desktop)
```

## 🔧 استكشاف الأخطاء

### **مشاكل شائعة:**

#### **PySimpleGUI خطأ:**
```bash
pip uninstall PySimpleGUI -y
pip install --force-reinstall --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
```

#### **مشاكل الصوت في الراسبيري باي:**
```bash
sudo apt install alsa-utils pulseaudio
```

#### **مشكلة DISPLAY:**
```bash
export DISPLAY=:0
# أو للوضع الصوتي
export SIRAJ_HEADLESS=true
```

## 🤝 المساهمة

1. Fork المشروع
2. إنشاء branch جديد (`git checkout -b feature/amazing-feature`)
3. Commit التغييرات (`git commit -m 'Add amazing feature'`)
4. Push إلى Branch (`git push origin feature/amazing-feature`)
5. فتح Pull Request

## 📄 الرخصة

هذا المشروع مرخص تحت رخصة MIT - انظر ملف [LICENSE](LICENSE) للتفاصيل.

## 🙏 الشكر والتقدير

- **Google Gemini AI** - للذكاء الاصطناعي المتقدم
- **OpenCV** - لتقنيات رؤية الكمبيوتر
- **PySimpleGUI** - للواجهة الرسومية
- **مترو الرياض** - للبيانات والإلهام

## 📞 الدعم والتواصل

- **Issues**: [GitHub Issues](https://github.com/your-username/siraj-voice-assistant/issues)
- **Email**: your-email@example.com
- **Twitter**: [@your-twitter]

## 🌟 النسخ المستقبلية

- [ ] دعم المزيد من اللغات
- [ ] تكامل مع تطبيقات التوصيل
- [ ] ميزات الحجز المباشر
- [ ] تطبيق موبايل
- [ ] API للمطورين

---

<div align="center">
  <b>صُنع بـ ❤️ في الرياض، المملكة العربية السعودية</b>
</div> 