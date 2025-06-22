# 🤝 المساهمة في مشروع سراج

نرحب بمساهماتكم في تطوير **سراج**! هذا الدليل سيساعدكم على فهم كيفية المساهمة بفعالية.

## 📋 جدول المحتويات

- [كيفية المساهمة](#كيفية-المساهمة)
- [إعداد البيئة التطويرية](#إعداد-البيئة-التطويرية)
- [إرشادات الكود](#إرشادات-الكود)
- [تشغيل الاختبارات](#تشغيل-الاختبارات)
- [إرسال Pull Request](#إرسال-pull-request)
- [الإبلاغ عن المشاكل](#الإبلاغ-عن-المشاكل)

## 🚀 كيفية المساهمة

### أنواع المساهمات المرحب بها:

- 🐛 **إصلاح الأخطاء** - حل المشاكل الموجودة
- ✨ **ميزات جديدة** - إضافة وظائف مفيدة
- 📚 **تحسين الوثائق** - تطوير الشرح والأمثلة
- 🌍 **دعم لغات جديدة** - إضافة لغات أخرى
- 🎨 **تحسين الواجهة** - تطوير تجربة المستخدم
- 🔧 **تحسين الأداء** - تسريع وتحسين الكود

## 🛠️ إعداد البيئة التطويرية

### 1. Fork والاستنساخ:
```bash
# Fork المشروع من GitHub
git clone https://github.com/your-username/siraj-voice-assistant.git
cd siraj-voice-assistant
```

### 2. إعداد البيئة:
```bash
# تشغيل سكريبت الإعداد
chmod +x setup.sh
./setup.sh

# أو الإعداد اليدوي
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. إعداد Git:
```bash
# إضافة المستودع الأصلي
git remote add upstream https://github.com/original-owner/siraj-voice-assistant.git

# إنشاء branch جديد للتطوير
git checkout -b feature/your-feature-name
```

### 4. تجربة التشغيل:
```bash
# إضافة API key
echo "GEMINI_API_KEY=your_key_here" > .env

# تشغيل التطبيق
python3 full_inegration.py
```

## 📝 إرشادات الكود

### معايير البرمجة:

#### 1. **Python Style:**
```python
# استخدم Black للتنسيق
black full_inegration.py

# استخدم flake8 للتحقق
flake8 full_inegration.py --max-line-length=100
```

#### 2. **التعليقات والوثائق:**
```python
def process_arabic_text(text: str) -> str:
    """
    معالجة النص العربي وتحسين النطق
    
    Args:
        text (str): النص العربي المدخل
        
    Returns:
        str: النص المحسن للنطق
        
    Example:
        >>> process_arabic_text("مرحباً بك")
        'مرحبا بك'
    """
    # كود المعالجة هنا
    return enhanced_text
```

#### 3. **معالجة الأخطاء:**
```python
import logging

logger = logging.getLogger(__name__)

try:
    # الكود الرئيسي
    result = dangerous_operation()
except SpecificException as e:
    logger.error(f"خطأ محدد: {e}")
    # معالجة مناسبة
except Exception as e:
    logger.error(f"خطأ غير متوقع: {e}")
    # معالجة عامة
```

#### 4. **الثوابت والإعدادات:**
```python
# في أعلى الملف
DEFAULT_SAMPLE_RATE = 16000
SUPPORTED_LANGUAGES = ["ar", "en", "ur", "zh"]
AUDIO_CHUNK_SIZE = 1024

# استخدام الثوابت
def initialize_audio(sample_rate: int = DEFAULT_SAMPLE_RATE):
    pass
```

### إرشادات خاصة بسراج:

#### 1. **معالجة اللغة العربية:**
```python
# دائماً استخدم UTF-8
# -*- coding: utf-8 -*-

# اختبار النصوص العربية
arabic_text = "السلام عليكم"
assert len(arabic_text) > 0
```

#### 2. **التعامل مع الصوت:**
```python
# استخدم try/except للعمليات الصوتية
try:
    audio_data = record_audio()
    if audio_data is None:
        logger.warning("لم يتم تسجيل صوت")
        return
except AudioException as e:
    logger.error(f"خطأ في الصوت: {e}")
```

#### 3. **دعم الراسبيري باي:**
```python
# فحص المنصة
import platform

def is_raspberry_pi():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            return 'BCM' in f.read()
    except:
        return False

# إعدادات مختلفة للراسبيري باي
if is_raspberry_pi():
    CHUNK_SIZE = 1024  # أصغر للراسبيري باي
else:
    CHUNK_SIZE = 2048  # أكبر للأجهزة القوية
```

## 🧪 تشغيل الاختبارات

### 1. **الاختبارات الأساسية:**
```bash
# تشغيل جميع الاختبارات
python -m pytest tests/

# اختبار ملف محدد
python -m pytest tests/test_audio.py -v

# اختبار مع التغطية
python -m pytest --cov=. tests/
```

### 2. **اختبارات يدوية:**
```bash
# اختبار الصوت
python3 -c "from full_inegration import test_audio; test_audio()"

# اختبار الراسبيري باي
export SIRAJ_HEADLESS=true
python3 full_inegration.py --test-mode
```

### 3. **كتابة اختبارات جديدة:**
```python
# tests/test_new_feature.py
import pytest
from full_inegration import new_function

def test_new_function_with_arabic():
    """اختبار الوظيفة الجديدة مع النص العربي"""
    result = new_function("مرحباً")
    assert result is not None
    assert "مرحبا" in result

def test_new_function_edge_cases():
    """اختبار الحالات الاستثنائية"""
    assert new_function("") == ""
    assert new_function(None) is None
```

## 📤 إرسال Pull Request

### خطوات الإرسال:

1. **تأكد من جودة الكود:**
```bash
# تنسيق الكود
black .
flake8 . --max-line-length=100

# تشغيل الاختبارات
pytest tests/
```

2. **كتابة رسالة commit واضحة:**
```bash
git add .
git commit -m "✨ إضافة دعم اللغة الفرنسية

- إضافة ترجمات فرنسية للواجهة
- تحديث نظام النطق للفرنسية
- اختبارات جديدة للغة الفرنسية
- تحديث الوثائق"
```

3. **رفع التغييرات:**
```bash
git push origin feature/your-feature-name
```

4. **إنشاء Pull Request:**
- اذهب إلى GitHub
- اضغط "New Pull Request"
- اكتب عنوان ووصف واضح
- اربط بـ Issue إذا كان موجود

### قالب Pull Request:

```markdown
## 📋 وصف التغييرات
وصف مختصر للتغييرات المضافة

## 🔗 مرتبط بـ Issue
Fixes #123

## ✅ التحقق من الجودة
- [ ] تم تشغيل الاختبارات
- [ ] تم تنسيق الكود
- [ ] تم تحديث الوثائق
- [ ] تم اختبار الراسبيري باي (إذا كان مطلوباً)

## 🖼️ لقطات الشاشة
(إذا كانت التغييرات تتعلق بالواجهة)

## 📝 ملاحظات إضافية
أي معلومات إضافية للمراجعين
```

## 🐛 الإبلاغ عن المشاكل

### قبل الإبلاغ:
1. تأكد من وجود المشكلة في أحدث إصدار
2. ابحث في Issues الموجودة
3. جرب الحلول في دليل استكشاف الأخطاء

### قالب الإبلاغ:
```markdown
## 🐛 وصف المشكلة
وصف واضح ومختصر للمشكلة

## 🔄 خطوات إعادة الإنتاج
1. اذهب إلى '...'
2. اضغط على '...'
3. شاهد الخطأ

## ✅ السلوك المتوقع
ما كان يجب أن يحدث

## 📱 معلومات البيئة
- نظام التشغيل: [مثلاً Ubuntu 20.04]
- Python: [مثلاً 3.9.7]
- الجهاز: [مثلاً Raspberry Pi 4]

## 📋 رسائل الأخطاء
```
ضع رسائل الأخطاء هنا
```

## 📸 لقطات الشاشة
إذا كان ذلك مناسباً
```

## 🏷️ تصنيف المساهمات

استخدم هذه التصنيفات في commits و PRs:

- 🐛 `:bug:` - إصلاح خطأ
- ✨ `:sparkles:` - ميزة جديدة  
- 📚 `:books:` - تحديث الوثائق
- 🎨 `:art:` - تحسين الكود/الواجهة
- ⚡ `:zap:` - تحسين الأداء
- 🔧 `:wrench:` - تغيير الإعدادات
- 🌍 `:earth_africa:` - تدويل/لغات جديدة
- 🧪 `:test_tube:` - إضافة اختبارات

## 📞 الحصول على المساعدة

- **GitHub Issues**: للأسئلة التقنية
- **GitHub Discussions**: للنقاشات العامة
- **Email**: للاستفسارات الخاصة

## 🙏 شكر المساهمين

نشكر جميع من ساهم في تطوير سراج:
- المطورين
- المختبرين  
- كاتبي الوثائق
- مقدمي الاقتراحات

---

**مع الشكر لكل من يساهم في جعل سراج أفضل! 🎙️** 