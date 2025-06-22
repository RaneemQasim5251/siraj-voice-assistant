# 🖥️ Siraj GUI Setup Guide for Raspberry Pi

## المشكلة: GUI لا يظهر
عندما تحاول تشغيل Siraj على الراسبيري باي عبر SSH، قد لا يظهر الـ GUI بسبب عدم تعيين متغير `DISPLAY`.

## 🎯 الحلول المختلفة:

### **1. التشغيل المباشر على الراسبيري باي**
إذا كان عندك شاشة ولوحة مفاتيح متصلة مباشرة بالراسبيري باي:

```bash
# من terminal الراسبيري باي مباشرة
cd ~/siraj_gemini
export DISPLAY=:0
python3 full_inegration.py
```

### **2. استخدام السكريبت الجاهز**
تم إنشاء سكريبت يعرض الـ GUI على شاشة الراسبيري باي:

```bash
# تشغيل السكريبت
ssh siraj@192.168.100.10
~/run_siraj_gui.sh
```

### **3. التشغيل بدون GUI (Headless Mode)**
الأفضل للاستخدام العادي - صوت فقط بدون فيديو:

```bash
ssh siraj@192.168.100.10
cd ~/siraj_gemini
export SIRAJ_HEADLESS=true
python3 full_inegration.py
```

### **4. SSH مع X11 Forwarding**
لعرض الـ GUI على جهاز الكمبيوتر الخاص بك:

#### على Windows:
1. قم بتثبيت [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
2. شغل XLaunch واختر "Multiple windows"
3. ثم نفذ:
```bash
ssh -X siraj@192.168.100.10
cd ~/siraj_gemini
python3 full_inegration.py
```

#### على Linux/Mac:
```bash
ssh -X siraj@192.168.100.10
cd ~/siraj_gemini
python3 full_inegration.py
```

### **5. إعداد متغير DISPLAY بشكل دائم**
لتعيين DISPLAY تلقائياً في كل مرة:

```bash
ssh siraj@192.168.100.10
echo 'export DISPLAY=:0' >> ~/.bashrc
source ~/.bashrc
```

## 🚀 **التوصية الأفضل:**

### للاستخدام اليومي العادي:
```bash
ssh siraj@192.168.100.10
cd ~/siraj_gemini
export SIRAJ_HEADLESS=true
python3 full_inegration.py
```

### للعرض التقديمي أو التجريب:
```bash
ssh siraj@192.168.100.10
~/run_siraj_gui.sh
```

## 🔧 **حل المشاكل:**

### إذا ظهرت رسالة خطأ "No display":
```bash
export DISPLAY=:0
```

### إذا لم تعمل شاشة الراسبيري باي:
```bash
# تأكد من تشغيل X11
sudo systemctl status display-manager
```

### للتشغيل التلقائي عند بدء الراسبيري باي:
```bash
# إضافة إلى startup
echo "@/home/siraj/run_siraj_gui.sh" >> ~/.config/lxsession/LXDE-pi/autostart
```

## 📝 **ملاحظات مهمة:**
- الـ GUI يحتاج شاشة متصلة بالراسبيري باي
- Headless mode أسرع وأقل استهلاكاً للذاكرة
- يمكن التبديل بين الوضعين حسب الحاجة 