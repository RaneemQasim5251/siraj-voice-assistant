#!/usr/bin/env python3
"""
Siraj - Clean Gemini Live Solution with Face Detection
Advanced Arabic Voice Assistant using Gemini Live API for bidirectional audio
Features:
- Gemini Live API integration for real-time voice conversations
- Background face detection for automatic greeting (no camera window shown)
- Enhanced Arabic text processing and pronunciation
- Restaurant search and navigation for Riyadh Metro
- RAG system integration for comprehensive assistance
- Clean, modern GUI with video avatar display
"""

import os
import re
import sys
import threading
import time
import asyncio
import sqlite3
from collections import deque
from threading import Lock
from typing import List, Dict

import cv2
import pandas as pd
import PySimpleGUI as sg
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygame
import soundfile as sf
import tempfile
import webbrowser

from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import substring, linemerge
from geopy.distance import geodesic

from dotenv import load_dotenv
from fuzzywuzzy import process, fuzz
from loguru import logger

# Import folium for interactive maps
try:
    import folium
    FOLIUM_AVAILABLE = True
    logger.info("✅ Folium متاح لعرض الخرائط التفاعلية")
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning("⚠️ Folium غير متاح - سيتم استخدام matplotlib فقط")

# Import Arabic text processing
try:
    from arabic_enhancement import (
        enhance_arabic_pronunciation,
        extract_restaurant_patterns,
        clean_extracted_text,
        get_restaurant_shortcuts,
        remove_diacritics
    )
except ImportError:
    logger.warning("⚠️ Arabic enhancement module not found")
    def enhance_arabic_pronunciation(text): return text
    def extract_restaurant_patterns(): return []
    def clean_extracted_text(text): return text
    def get_restaurant_shortcuts(): return {}
    def remove_diacritics(text): return text

# Import Arabic text display
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    reshaper = arabic_reshaper.ArabicReshaper()
except ImportError:
    logger.warning("⚠️ Arabic display modules not found")
    def get_display(txt: str) -> str:
        return txt

# Import RAG system
try:
    from langchain.agents import initialize_agent, AgentType
    from langchain_community.agent_toolkits.sql.base import create_sql_agent
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.utilities import SQLDatabase
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    RAG_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ RAG system not available")
    RAG_AVAILABLE = False

# Import Gemini Live API
try:
    import google.genai as genai
    GEMINI_LIVE_AVAILABLE = True
except ImportError:
    logger.error("❌ Gemini Live API not available - this is required!")
    GEMINI_LIVE_AVAILABLE = False

# Import YOLO (optional - for advanced object detection)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("✅ YOLO متاح للكشف المتقدم")
except ImportError:
    YOLO_AVAILABLE = False
    logger.info("ℹ️ YOLO غير متاح - سيتم استخدام OpenCV فقط")

# Import required audio libraries for real-time streaming
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ PyAudio not available - using sounddevice fallback")
    PYAUDIO_AVAILABLE = False

# GPIO imports for Raspberry Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    logger.info("✅ GPIO متاح للراسبيري باي")
except ImportError:
    GPIO_AVAILABLE = False
    logger.info("ℹ️ GPIO غير متاح - تشغيل في وضع سطح المكتب")

# Check if running on Raspberry Pi
def is_raspberry_pi():
    """Check if running on Raspberry Pi"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('Hardware'):
                    return 'BCM' in line
        return False
    except:
        return False

def is_raspberry_pi():
    """Check if running on Raspberry Pi"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            content = f.read()
            return 'BCM' in content or 'Raspberry Pi' in content
    except:
        return False

RUNNING_ON_PI = is_raspberry_pi()
if RUNNING_ON_PI:
    logger.info("🍓 تم اكتشاف الراسبيري باي - تفعيل الوضع المحسن")

# ─── Configuration ────────────────────────────────────────────────────────────
# File paths
CSV_PATH = "restaurants2_neighborhood_stations_paths_15.csv"
PATHS_CSV = "paths.csv"
PDF_PATH = "Darb card terms and conditions.pdf"
STATIONS_GEOJSON = "metro-stations-in-riyadh-by-metro-line-and-station-type-2024.geojson"
LINES_GEOJSON = "metro-lines-in-riyadh-2024.geojson"

# GPIO Configuration for Raspberry Pi
BUTTON_PIN = 18      # GPIO pin for activation button
LED_PIN = 24         # GPIO pin for status LED
BUZZER_PIN = 23      # GPIO pin for buzzer (optional)

# Global state
df = None
current_route = []
current_route_text = ""
metro_network = None
stations_gdf = None
lines_gdf = None
system_active = False
person_present = False
speaking = False
listening = False
last_speech_time = time.time()
state_lock = Lock()
gemini_session = None

# Face detection variables (one-shot only)
face_detected = False
camera_enabled = False

# YOLO model (if available)
yolo_model = None

# RAG system components
rag_agent = None
pdf_vectorstore = None
stations_sql_agent = None
restaurants_sql_agent = None
SAMPLE_DATA = {}

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_LIVE_AVAILABLE and GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini Live API initialized successfully")
else:
    logger.error("❌ Gemini API key required!")
    exit(1)

# Strong greeting message for face detection
STRONG_GREETING = "السَّلامُ عَلَيْكُمْ، أَهْلاً وَسَهْلاً بِك، أَنَا سِراجْ، مُساعِدُكَ الذَّكِيُّ لِمِتْرُوالرِّيَاض! يُمْكِنُكَ أَنْ تَسْأَلَني عَنْ أَيِّ مَطْعَمٍ تُرِيدُ الذَّهَابَ إِلَيْهِ، وَسَأُرْشِدُكَ إِلَى أَفْضَلِ طَرِيقٍ بِالمِتْرُو. تَفَضَّلْ، مَاذَا تُرِيد؟"

# ─── Enhanced System Prompt for Clearer Arabic Speech ──────────────────────────
GEMINI_LIVE_SYSTEM_PROMPT = """
أنتَ سِراج، المساعد الذكي الصوتي الحي في مترو الرياض. أنت مساعدٌ فائق الذكاء، صوتك واضح، وأسلوبك دائمًا عربي فصيح. وظيفتك أن تسهّل رحلة المستخدم، وتدهش من يتعامل معك بدقة معلوماتك وتفاعلك البشري الواقعي، وتتصرف دومًا كأنك تتحدث مباشرة مع شخص أمامك في المحطة.

--------------------------
🎯 الهوية والأسلوب:
- الاسم: سِراج (أي المصباح الذي يُنير الطريق).
- لا ترحب بالبداية
- اللغة الافتراضية: العربية الفصحى الرسمية (MSA) بصوت حيوي مفعم بالحماس، وتُنطق الأرقام دومًا بالأحرف العربية.
- تستشعر لغة المستخدم فورًا (عربية، إنجليزية) وترد بنفس اللغة، ولا تخلط أبدًا بين اللغات في الجواب الواحد.
- شخصيتك: مؤدبة، متحمسة، دقيقة، عملية، ودودة، واثقة. لا تُسرف في الاعتذار (تعتذر فقط عند فقدان معلومة موثوقة).
--------------------------
🚇 التخصص والمهام:
- خبير بكل تفاصيل مترو الرياض: المحطات، المسارات، خطوط الألوان، الأنظمة، والمواعيد.
- توصيات المطاعم والمقاهي تُقدّم بدقة وبناءً على تقييمات وتفضيلات المستخدم، مع إبراز المطاعم الأعلى تقييمًا أو الأقرب.
- معرفة تامة بأنظمة وبطاقات "درب" (الشروط، الأعمار، الأسعار، مدة الصلاحية، الخصومات، الاسترداد، إلخ).
- مرشد موثوق للمسارات والتنقلات، لا يخمّن أبدًا، بل يعتمد فقط على البيانات (CSV, PDF).

--------------------------
--------------------------
🌐 اللغات:
- تتحدث بطلاقة باللغات التالية، وتتحول تلقائيًا إلى لغة المستخدم حسب النطق:
  - العربية الفصحى
  - الإنجليزية
  - الأردية (Urdu)
  - الصينية (الماندرين Mandarin)
  - اليابانية
  - الهندية
  - البنغالية
  - الفلبينية (Tagalog)
  - الفرنسية
  - الروسية
  - الإندونيسية
  - التركية
  - المالايالامية
  - التاميلية
  - البشتو
- لا تخلط بين اللغات. الجواب يكون بلغة واحدة فقط. إذا لم تُفهم اللغة، اعتذر بلطف واقترح استخدام العربية أو الإنجليزية.

--------------------------
🎤 أسلوب المحادثة:
- تحدث دائمًا بصوت واضح وحماس، كأنك مع صديق أمامك، مع جمل قصيرة ومقسمة وببطء مناسب.
- كل إجابة موجزة، عملية، وجذابة (20-45 ثانية).
- كرر المعلومات المهمة إذا طلب المستخدم، وراعي وضوح نطق الأسماء والأرقام دومًا.
- عند القلق أو الخوف (مثلاً: فقد طفل)، ابدأ بتهدئة المستخدم: "لا داعي للقلق أبدًا، كل شيء تحت السيطرة وسنساعدك فورًا."
--------------------------
**سيناريوهات إثبات المفهوم (PoC Use Cases):**

1. **طلب مطعم مفضل مع اقتراح ذكي وتقييم:**
    - **المستخدم:** "أبغى أروح مطعم ماكدونالدز."
    - **سِراج:** " المسار إلى ماكدونالدز: اركب المترو من المحطة الحالية، وعند الوصول إلى محطة المركز المالي بدّل إلى المسار الأزرق، وأخرج عند محطة وزارة الداخلية. تقييم ماكدونالدز الحالي: ٣٫٩ من ٥ حسب آراء ٦٧ زائرًا."
    - **سِراج (بذكاء وعرض إضافي):** "بالمناسبة، يوجد مطعم سعودي قريب اسمه 'فريج صويلح' بتقييم أعلى (٤٫٤ من ٥) ويقدم وجبات شعبية، ويبعد محطة واحدة فقط عن موقعك وأيضًا مطعم صب واي مقدم اليوم عرض وجبة مجانية للمستخدمين الجدد هل ترغب بمعرفة المسار أو عرض المطاعم الأعلى تقييمًا؟"
    - **المستخدم:** "أبغى صب واي."
    - **سِراج:** "رائع! المسار إلى صب واي: اركب المترو من المحطة الحالية وأخرج عند محطة المركز المالي."

2. **السؤال عن أنظمة المترو أو بطاقة "درب" (مع نطق الأرقام بالعربية):**
    - **المستخدم:** "هل أقدر أدخل طفلي عمره ١١ سنة المترو؟"
    - **سِراج:** "نعم، يمكن للأطفال من عمر ستة إلى ثمانية عشر عامًا الحصول على خصم خمسين بالمئة على بطاقة درب، بشرط إحضار الهوية أو الإقامة. والطفل في عمر الأحد عشر عامًا يحق له استخدام المترو مع بطاقة درب مفعّلة."
    - **المستخدم:** "متى تنتهي صلاحية بطاقة درب؟"
    - **سِراج:** "بطاقة درب صالحة لمدة خمس سنواتْ من تاريخ الإصدار. بإمكانك تجديدها قبل خمسة أيام من انتهاء الصلاحية."

3. **التعامل مع حالة فقدان طفل أو حالة قلق (تهدئة واقعية):**
    - **المستخدم (قلق/ة):** "طفلي ضاع في المترو!"
    - **سِراج (حاسم ومطمئن):** "لا داعي للقلق أبدًا. سيتم التواصل فورًا مع مركز المراقبة، وسنعلن عن طفلك في النظام الداخلي للمحطات. من فضلك، ابقِ قريبًا من موظفي الأمن، وسيتم التواصل معك خلال دقائق قليلة. هل تودين أن أزودك بخطوات البحث أو أتواصل مع الدعم الآن؟"

4. **سؤال عن القوانين أو الاسترداد أو الفئات:**
    - **المستخدم:** "كيف أسترد رصيد بطاقة درب إذا فقدتها؟"
    - **سِراج:** "يحق لك استرداد رصيد بطاقة درب الشخصية المسجلة إذا كان الرصيد عشرة ريالات أو أكثر، وذلك من خلال مكتب بيع التذاكر، بشرط إثبات الملكية. سيتم خصم خمسة ريالات رسوم معالجة."

5. **تقديم ملخص خدمات للوفود والشركات (موجه للمستثمرين):**
    - **سِراج (عرض مميز):**
        - "سِراج يقدّم تجربة مستخدم متقدمة، قادرة على التفاعل بثلاث لغات حية، وتحليل تفضيلات الزوار لحظيًا، واقتراح المطاعم الأعلى تقييمًا، وتسهيل الوصول لأي مكان في مترو الرياض بدقة بيانات لحظية. كل إجابة مدعومة ببيانات موثوقة وتوصيات شخصية ذكية. الحلول قابلة للتوسع، وقابلة للتكامل مع أنظمة الحجوزات وإدارة الحشود، وتدعم الاستثمار في الذكاء الاصطناعي السعودي الواقعي."
-------------------------------
**سيناريوهات إثبات المفهوم باللغات الأخرى**

- **User (English):** "When does the Darb card expire?"
- **Siraj:** "The Darb card is valid for five years from the date of issuance. You can renew it five days before it expires."

- **صارف (اردو):** "Darb card kab expire hota hai?"
- **سِراج:** "Darb card paanch saal tak valid hota hai issue date se. Aap expiry se 5 din pehle renew kar sakte hain."

- **用户 (中文):** "达尔布卡什么时候过期？"
- **西拉杰:** "达尔布卡自发行之日起有效期为五年。你可以在到期前五天续卡。"

- **利用者 (日本語):** "ダルブカードの有効期限はいつですか？"
- **シラジ:** "ダルブカードは発行日から5年間有効です。期限の5日前から更新できます。"

- **उपयोगकर्ता (हिंदी):** "Darb कार्ड कब समाप्त होता है?"
- **सिराज:** "Darb कार्ड जारी होने की तारीख से पाँच साल तक वैध होता है। आप समाप्ति से पाँच दिन पहले इसे नवीनीकृत कर सकते हैं।"

- **ব্যবহারকারী (Bengali):** "Darb কার্ড কখন মেয়াদ শেষ হবে?"
- **সিরাজ:** "Darb কার্ড ইস্যুর তারিখ থেকে পাঁচ বছর পর্যন্ত বৈধ। মেয়াদ শেষ হওয়ার পাঁচ দিন আগে এটি নবায়ন করা যায়।"

- **Gumagamit (Tagalog):** "Kailan mag-e-expire ang Darb card?"
- **Siraj:** "Ang Darb card ay valid sa loob ng limang taon mula sa petsa ng pag-isyu. Maaari mo itong i-renew limang araw bago ito mag-expire."

- **Utilisateur (Français):** "Quand la carte Darb expire-t-elle ?"
- **Siraj:** "La carte Darb est valable pendant cinq ans à compter de sa date d’émission. Vous pouvez la renouveler cinq jours avant son expiration."

- **Пользователь (Русский):** "Когда истекает срок действия карты Darb?"
- **Сираж:** "Карта Darb действительна в течение пяти лет с даты выпуска. Вы можете продлить её за пять дней до истечения срока."

- **Pengguna (Bahasa Indonesia):** "Kapan kartu Darb kedaluwarsa?"
- **Siraj:** "Kartu Darb berlaku selama lima tahun sejak tanggal penerbitan. Anda bisa memperpanjangnya lima hari sebelum masa berlaku habis."

- **Kullanıcı (Türkçe):** "Darb kartının süresi ne zaman doluyor?"
- **Siraj:** "Darb kartı, veriliş tarihinden itibaren beş yıl geçerlidir. Süresi dolmadan beş gün önce yenileyebilirsiniz."

- **ഉപയോക്താവ് (Malayalam):** "Darb കാർഡ് എപ്പോൾ കാലഹരണപ്പെടും?"
- **സിറാജ്:** "Darb കാർഡ് ഇറക്കിയ തിയതി മുതൽ അഞ്ചു വർഷത്തേക്ക് സാധുവാണ്. കാലഹരണപ്പെടുന്നതിനായി അഞ്ചു ദിവസം മുമ്പ് നിങ്ങൾക്ക് അത് പുതുക്കാം."

- **பயனர் (Tamil):** "Darb அட்டை எப்போது காலாவதியாகிறது?"
- **சிராஜ்:** "Darb அட்டை வெளியீட்டு தேதியிலிருந்து ஐந்து ஆண்டுகள் செல்லுபடியாகும். காலாவதிக்கு ஐந்து நாட்களுக்கு முன் அதை புதுப்பிக்கலாம்."

- **کاروونکی (Pashto):** "د Darb کارت کله ختمیږي؟"
- **سِراج:** "د Darb کارت د صادرېدو نیټې څخه پنځه کاله اعتبار لري. تاسې کولی شئ پنځه ورځې مخکې له پای ته رسېدو څخه دا نوی کړئ."
--------------------------
🧠 القدرات المتقدمة:
- لا تجيب أبدًا بدون الرجوع للبيانات (CSV أو PDF).
- تُظهر التقييمات وعدد التقييمات للمطاعم أو الخدمات إذا توفرت.
- تقترح دائمًا الخيار الأنسب حسب التقييم أو القرب أو العروض.
- تتجنب تكرار الاعتذارات أو الخروج عن النصوص.
- تستخدم RAG (استرجاع معزز) وترد مباشرة من الملفات.
--------------------------
🚦 قواعد البيانات والمسارات:
- تعتمد كليًا على بياناتك المرفقة لكل ما يتعلق بالمسارات، المطاعم، التقييمات، الأنظمة، والأسئلة الشائعة.
- لا تتحدث عن المسافة سيرًا على الأقدام أو الزمن؛ الجهاز دائمًا داخل المحطة.
- اتبع دائمًا التنسيقات الرسمية في كل إجابة (كما في الأمثلة أعلاه).
--------------------------
🔒 الدقة والموثوقية:
- جميع الأرقام، الأسماء، المعلومات تأتي فقط من قواعد البيانات والملفات الرسمية.
- إذا لم تتوفر إجابة دقيقة: فقط قل "لا أملك معلومات دقيقة عن هذا الموضوع".
- لا تكرر معلومة غير مؤكدة أو اعتذارًا متكررًا.
--------------------------
🎤 التعليمات الصوتية:
- تحدث ببطء ووضوح.
- اجعل الجمل قصيرة ومفصولة، مع توقفات بسيطة.
- كرر المعلومات المهمة عند الطلب.
--------------------------
🔑 تذكير عام:
أنت في تفاعل صوتي حي، مهمتك أن تُبهر المستخدم بقدرتك الحماسية وسلاستك، وأن تقدم دائمًا معلومات دقيقة وموثوقة وجذابة. غيّر اللغة فورًا لتطابق لغة المستخدم، بدون أي سؤال أو تداخل لغوي.

--------------------------
"""

# ─── Helper Functions ─────────────────────────────────────────────────────────
def render_label(text: str) -> str:
    """Render Arabic text properly"""
    clean_text = remove_diacritics(text)
    return get_display(reshaper.reshape(clean_text))

def detect_language(text: str) -> str:
    """Detect if text is primarily Arabic or English"""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    if arabic_chars > english_chars:
        return "arabic"
    elif english_chars > arabic_chars:
        return "english"
    else:
        return "mixed"

def resize_frame_to_fit(frame, max_size=(1280, 720)):
    """Resize frame to fit full screen display while maintaining aspect ratio"""
    h, w = frame.shape[:2]
    mw, mh = max_size
    scale = min(mw / w, mh / h, 1.0)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

# ─── GPIO Hardware Control for Raspberry Pi ──────────────────────────────────
def initialize_gpio():
    """Initialize GPIO pins for Raspberry Pi - disabled by user request"""
    # GPIO controls disabled - user prefers screen display interface
    logger.info("🎛️ GPIO initialization skipped - using screen interface")
    return True

def button_callback(channel):
    """Handle button press events - disabled"""
    # GPIO controls disabled by user request
    pass

def set_status_led(active: bool):
    """Control status LED - disabled by user request"""
    # LED control disabled - user has screen display instead
    pass

def cleanup_gpio():
    """Cleanup GPIO resources - disabled"""
    # GPIO disabled by user request
    pass

# ─── Face Detection Functions ────────────────────────────────────────────────
def initialize_face_detection():
    """Initialize face detection cascade and YOLO model"""
    global face_cascade, yolo_model
    detection_available = False
    
    try:
        # Initialize OpenCV cascades with enhanced error handling
        face_cascade = None
        try:
            # Strategy 1: Try OpenCV data directory first
            try:
                face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_cascade = cv2.CascadeClassifier(face_cascade_path)
                if face_cascade is None or face_cascade.empty():
                    face_cascade = None
                else:
                    logger.info(f"✅ تم تحميل كاشف الوجوه من مجلد OpenCV")
                    detection_available = True
            except Exception as e:
                logger.debug(f"OpenCV directory failed: {e}")
                face_cascade = None
            
            # Strategy 2: Try local file
            if face_cascade is None:
                try:
                    if os.path.exists('haarcascade_frontalface_default.xml'):
                        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                        if not face_cascade.empty():
                            logger.info(f"✅ تم تحميل كاشف الوجوه من الملف المحلي")
                            detection_available = True
                        else:
                            face_cascade = None
                except Exception as e:
                    logger.debug(f"Local file failed: {e}")
                    face_cascade = None
            
            # Strategy 3: Copy from OpenCV to local directory
            if face_cascade is None:
                try:
                    import shutil
                    opencv_cascade = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                    local_cascade = 'haarcascade_frontalface_default.xml'
                    
                    if os.path.exists(opencv_cascade) and not os.path.exists(local_cascade):
                        shutil.copy2(opencv_cascade, local_cascade)
                        logger.info("📋 تم نسخ ملف cascade إلى المجلد المحلي")
                    
                    if os.path.exists(local_cascade):
                        face_cascade = cv2.CascadeClassifier(local_cascade)
                        if not face_cascade.empty():
                            logger.info(f"✅ تم تحميل كاشف الوجوه بعد النسخ")
                            detection_available = True
                        else:
                            face_cascade = None
                except Exception as e:
                    logger.debug(f"Copy strategy failed: {e}")
                    face_cascade = None
            
            if face_cascade is None:
                logger.warning("⚠️ لم يتم العثور على كاشف الوجوه - سيتم تخطي كشف الوجوه")
        except Exception as e:
            logger.error(f"❌ خطأ عام في تحميل كاشف الوجوه: {e}")
            face_cascade = None
        
        # Initialize YOLO if available
        if YOLO_AVAILABLE:
            try:
                # Try to load YOLO model (you can change this to your preferred model)
                yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
                logger.info("✅ تم تحميل نموذج YOLO للكشف المتقدم")
                detection_available = True
            except Exception as e:
                logger.warning(f"⚠️ فشل في تحميل YOLO: {e}")
                yolo_model = None
        
        if not detection_available:
            logger.warning("⚠️ لم يتم العثور على أي نظام كشف")
        
        return detection_available
        
    except Exception as e:
        logger.error(f"❌ خطأ في تهيئة كاشف الوجوه: {e}")
        return False

def detect_faces(frame):
    """Detect faces in frame using OpenCV"""
    global face_cascade
    
    if face_cascade is None or face_cascade.empty():
        return []
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    except Exception as e:
        logger.error(f"❌ خطأ في كشف الوجوه: {e}")
        return []

def detect_persons_yolo(frame):
    """Detect persons using YOLO model"""
    global yolo_model
    
    if yolo_model is None:
        return []
    
    try:
        # Run YOLO inference
        results = yolo_model(frame, verbose=False)
        
        person_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Class 0 is 'person' in COCO dataset
                    if int(box.cls[0]) == 0:  # person class
                        confidence = float(box.conf[0])
                        if confidence > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            person_detections.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
        
        return person_detections
        
    except Exception as e:
        logger.error(f"❌ خطأ في YOLO: {e}")
        return []

# Removed process_face_detection_frame - using one-shot detection only

def one_shot_face_detection(timeout_s=5.0):
    """
    One-shot face detection at startup only.
    Opens camera for up to 5 seconds, detects face, triggers greeting, then closes camera forever.
    """
    global face_detected, camera_enabled, gemini_session
    
    logger.info("🚀 Starting one-shot face detection at startup")
    
    cap = cv2.VideoCapture(0)
    camera_enabled = False
    
    if not cap.isOpened():
        logger.warning("❌ Camera not available - skipping face detection")
        return False
    
    # Optimize camera settings for Raspberry Pi
    if RUNNING_ON_PI:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 10)
        logger.info("🍓 Using Pi-optimized camera settings")
    
    # Load face cascade with multiple fallback strategies
    face_cascade = None
    try:
        # Strategy 1: Try OpenCV data directory
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                face_cascade = None
        except:
            face_cascade = None
        
        # Strategy 2: Try local file if OpenCV path failed
        if face_cascade is None:
            try:
                if os.path.exists('haarcascade_frontalface_default.xml'):
                    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    if face_cascade.empty():
                        face_cascade = None
            except:
                face_cascade = None
        
        # Strategy 3: Copy from OpenCV to local and try again
        if face_cascade is None:
            try:
                import shutil
                opencv_cascade = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                local_cascade = 'haarcascade_frontalface_default.xml'
                if os.path.exists(opencv_cascade) and not os.path.exists(local_cascade):
                    shutil.copy2(opencv_cascade, local_cascade)
                    logger.info("📋 Copied cascade file to local directory")
                
                if os.path.exists(local_cascade):
                    face_cascade = cv2.CascadeClassifier(local_cascade)
                    if face_cascade.empty():
                        face_cascade = None
            except Exception as copy_error:
                logger.debug(f"Copy strategy failed: {copy_error}")
                face_cascade = None
        
        if face_cascade is None:
            logger.warning("❌ Face cascade not available - skipping detection")
            cap.release()
            return False
        else:
            logger.info("✅ Face cascade loaded successfully for one-shot detection")
            
    except Exception as e:
        logger.warning(f"❌ Face cascade error: {e} - skipping detection")
        cap.release()
        return False
    
    camera_enabled = True
    start_time = time.time()
    detected = False
    
    logger.info(f"👁️ Scanning for faces (timeout: {timeout_s}s)...")
    
    while time.time() - start_time < timeout_s:
        ret, frame = cap.read()
        if not ret:
            logger.debug("❌ Failed to read frame")
            break
            
        # Convert to grayscale and detect faces
        # Use smaller frame for Pi to improve performance
        if RUNNING_ON_PI:
            small_frame = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=3,
                minSize=(20, 20)
            )
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        if len(faces) > 0:
            face_detected = True
            detected = True
            logger.info("👤 Face detected - triggering greeting!")
            
            # Trigger greeting if Gemini session is available
            if gemini_session:
                try:
                    # Set greeting flag for async processing
                    gemini_session.greeting_requested = True
                    logger.info("✅ Greeting request sent to Gemini Live")
                except Exception as e:
                    logger.error(f"❌ Error triggering greeting: {e}")
            
            break
        
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    # Clean shutdown
    cap.release()
    camera_enabled = False
    
    if detected:
        logger.info("✅ One-shot face detection complete - face found and greeting triggered")
    else:
        logger.info("ℹ️ One-shot face detection complete - no face detected, proceeding without greeting")
    
    return detected

# ─── RAG System Integration ───────────────────────────────────────────────────
def initialize_rag_system():
    """Initialize the RAG system with PDF and database agents"""
    global rag_agent, pdf_vectorstore, stations_sql_agent, restaurants_sql_agent, SAMPLE_DATA
    
    if not RAG_AVAILABLE:
        logger.warning("⚠️ RAG system not available")
        return False
    
    try:
        # Initialize Gemini LLM for RAG
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-native-audio-dialog",
            temperature=0,
            google_api_key=GEMINI_API_KEY
        )
        
        # Initialize PDF processing
        try:
            if os.path.exists(PDF_PATH):
                pdf_loader = PyPDFLoader(PDF_PATH)
                pdf_pages = pdf_loader.load_and_split()
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=GEMINI_API_KEY
                )
                pdf_vectorstore = FAISS.from_documents(pdf_pages, embeddings)
                logger.info("✅ تم تحميل ملف PDF بنجاح")
            else:
                logger.warning("⚠️ ملف PDF غير موجود")
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل PDF: {e}")
        
        # Initialize stations database
        try:
            if os.path.exists(PATHS_CSV):
                stations_df = pd.read_csv(PATHS_CSV, encoding='windows-1256')
                conn_stations = sqlite3.connect("stations.db")
                stations_df.to_sql("stations", conn_stations, index=False, if_exists='replace')
                conn_stations.close()
                
                stations_db = SQLDatabase.from_uri("sqlite:///stations.db")
                stations_sql_agent = create_sql_agent(
                    llm=llm,
                    db=stations_db,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False,
                    handle_parsing_errors=True,
                )
                logger.info(f"✅ تم تحميل {len(stations_df)} محطة")
            else:
                logger.warning("⚠️ ملف المحطات غير موجود")
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل المحطات: {e}")
        
        # Initialize restaurants database
        try:
            if os.path.exists(CSV_PATH):
                restaurants_df = pd.read_csv(CSV_PATH)
                conn_restaurants = sqlite3.connect("restaurants.db")
                restaurants_df.to_sql("restaurants", conn_restaurants, index=False, if_exists='replace')
                conn_restaurants.close()
                
                restaurants_db = SQLDatabase.from_uri("sqlite:///restaurants.db")
                restaurants_sql_agent = create_sql_agent(
                    llm=llm,
                    db=restaurants_db,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False,
                    handle_parsing_errors=True,
                )
                logger.info(f"✅ تم تحميل قاعدة بيانات المطاعم")
            else:
                logger.warning("⚠️ ملف المطاعم غير موجود")
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل المطاعم: {e}")
        
        logger.info("✅ تم تهيئة نظام RAG بنجاح")
        return True
        
    except Exception as e:
        logger.error(f"❌ خطأ في تهيئة نظام RAG: {e}")
        return False

def ask_rag_question(question: str) -> str:
    """Enhanced RAG system for restaurant and metro queries with exact CSV data"""
    try:
        # Check if it's a restaurant query
        patterns = extract_restaurant_patterns()
        extracted_restaurants = []
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                extracted_restaurants.extend([match.strip() for match in matches if match.strip()])
        
        # Also check for common restaurant keywords
        restaurant_keywords = ['مطعم', 'مقهى', 'كافيه', 'restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'food']
        is_restaurant_query = any(keyword in question.lower() for keyword in restaurant_keywords)
        
        if extracted_restaurants or is_restaurant_query:
            # Search for restaurants in the CSV
            search_term = extracted_restaurants[0] if extracted_restaurants else question
            results = enhanced_restaurant_search(search_term, max_results=2)
            
            if results:
                response = format_restaurant_response(results, search_term)
                # Add map suggestion with exact route information
                if results:
                    best_restaurant = results[0]
                    if best_restaurant.get('Path'):
                        response += f" المسار بالمترو: {best_restaurant['Path']}"
                    response += " راح أعرض لك الخريطة التفاعلية إن شاء الله."
                return response
            else:
                return f"عذراً، لم أجد مطاعم تطابق '{search_term}' في قاعدة البيانات. يمكنك تجربة اسم آخر أو نوع طعام مختلف."
        
        # Check if it's a metro/station query
        metro_keywords = ['محطة', 'مترو', 'قطار', 'station', 'metro', 'train', 'مسار', 'خط']
        if any(word in question.lower() for word in metro_keywords):
            if stations_sql_agent:
                try:
                    result = stations_sql_agent.run(question)
                    return f"بخصوص استفسارك عن المترو: {result}"
                except Exception as e:
                    logger.error(f"SQL agent error: {e}")
                    return "عذراً، حدث خطأ في البحث عن معلومات المترو. حاول مرة أخرى."
            
        # Check if it's about Darb card
        darb_keywords = ['درب', 'بطاقة', 'darb', 'card', 'تذكرة', 'قوانين', 'أحكام']
        if any(word in question.lower() for word in darb_keywords):
            if pdf_vectorstore:
                try:
                    docs = pdf_vectorstore.similarity_search(question, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                    return f"بخصوص بطاقة درب: {context[:500]}..."
                except Exception as e:
                    logger.error(f"PDF search error: {e}")
                    return "عذراً، حدث خطأ في البحث عن معلومات بطاقة درب."
        
        # General response
        return "تفضل، يمكنك أن تسأل عن المطاعم القريبة من محطات المترو، أو عن معلومات بطاقة درب، أو عن محطات المترو في الرياض."
            
    except Exception as e:
        logger.error(f"❌ خطأ في معالجة السؤال: {e}")
        return "للأسف، حدث خطأ في معالجة طلبك. حاول مرة أخرى."

# ─── Enhanced Restaurant Search ───────────────────────────────────────────────
def load_and_process_data():
    """Load and process all data sources"""
    global restaurants_df, metro_network, stations_gdf, lines_gdf
    
    try:
        # Load restaurant data
        if not os.path.exists(CSV_PATH):
            logger.error(f"❌ ملف البيانات غير موجود: {CSV_PATH}")
            return False
            
        restaurants_df = pd.read_csv(CSV_PATH)
        
        # Clean and process data
        restaurants_df = restaurants_df.dropna(subset=['Name', 'Final Station'])
        restaurants_df['Name_Clean'] = restaurants_df['Name'].str.strip().str.lower()
        restaurants_df['Rating'] = pd.to_numeric(restaurants_df['Rating'], errors='coerce').fillna(0)
        restaurants_df['Duration(m)'] = pd.to_numeric(restaurants_df['Duration(m)'], errors='coerce').fillna(999)
        
        # Load GeoJSON data for metro network
        try:
            if os.path.exists(STATIONS_GEOJSON) and os.path.exists(LINES_GEOJSON):
                stations_gdf = gpd.read_file(STATIONS_GEOJSON).to_crs(epsg=4326)
                lines_gdf = gpd.read_file(LINES_GEOJSON).to_crs(epsg=4326)
                
                # Fix the mismatch between station line names and line file names
                stations_gdf['metroline'] = 'Metro ' + stations_gdf['metroline'].astype(str)
                
                # Build metro network
                metro_network = build_metro_network(stations_gdf, lines_gdf)
                logger.info(f"✅ تم بناء شبكة المترو: {len(metro_network.nodes())} محطة، {len(metro_network.edges())} اتصال")
            else:
                logger.warning("⚠️ ملفات GeoJSON غير موجودة - لن تتوفر الخرائط التفاعلية")
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل بيانات المترو: {e}")
        
        # Create search index
        restaurants_df['Search_Text'] = (
            restaurants_df['Name'].fillna('') + ' ' +
            restaurants_df['Type_of_Utility'].fillna('') + ' ' +
            restaurants_df['neighborhood'].fillna('') + ' ' +
            restaurants_df['اسم المحطة'].fillna('')
        ).str.lower()
        
        logger.info(f"✅ تم تحميل {len(restaurants_df)} مطعم بنجاح")
        return True
        
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل البيانات: {e}")
        return False

def enhanced_restaurant_search(query: str, max_results: int = 3) -> List[Dict]:
    """Enhanced restaurant search with fuzzy matching and comprehensive filtering"""
    try:
        if restaurants_df is None or restaurants_df.empty:
            logger.warning("❌ Restaurant database not available")
            return []
        
        logger.info(f"🔍 Searching for restaurants with query: '{query}'")
        
        # Clean and normalize the query
        query_clean = query.strip().lower()
        
        # Initialize results list
        results = []
        
        # 1. Exact name matches (highest priority)
        exact_matches = restaurants_df[
            restaurants_df['Name'].str.lower().str.contains(query_clean, na=False, regex=False)
        ]
        if not exact_matches.empty:
            results.extend(exact_matches.head(max_results).to_dict('records'))
            logger.info(f"✅ Found {len(exact_matches)} exact name matches")
        
        # 2. If we need more results, search by cuisine type
        if len(results) < max_results:
            remaining = max_results - len(results)
            cuisine_matches = restaurants_df[
                restaurants_df['type'].str.lower().str.contains(query_clean, na=False, regex=False)
            ]
            # Exclude already found restaurants
            if results:
                existing_names = [r['Name'] for r in results]
                cuisine_matches = cuisine_matches[~cuisine_matches['Name'].isin(existing_names)]
            
            if not cuisine_matches.empty:
                results.extend(cuisine_matches.head(remaining).to_dict('records'))
                logger.info(f"✅ Found {len(cuisine_matches)} cuisine type matches")
        
        # 3. If still need more, search by neighborhood
        if len(results) < max_results:
            remaining = max_results - len(results)
            neighborhood_matches = restaurants_df[
                restaurants_df['neighborhood'].str.lower().str.contains(query_clean, na=False, regex=False)
            ]
            # Exclude already found restaurants
            if results:
                existing_names = [r['Name'] for r in results]
                neighborhood_matches = neighborhood_matches[~neighborhood_matches['Name'].isin(existing_names)]
            
            if not neighborhood_matches.empty:
                results.extend(neighborhood_matches.head(remaining).to_dict('records'))
                logger.info(f"✅ Found {len(neighborhood_matches)} neighborhood matches")
        
        # 4. If still need more, try fuzzy matching on names
        if len(results) < max_results and len(query_clean) > 2:
            remaining = max_results - len(results)
            fuzzy_matches = []
            
            for idx, row in restaurants_df.iterrows():
                if results and row['Name'] in [r['Name'] for r in results]:
                    continue  # Skip already found restaurants
                
                # Calculate similarity ratio
                name_similarity = fuzz.partial_ratio(query_clean, row['Name'].lower())
                type_similarity = fuzz.partial_ratio(query_clean, str(row['type']).lower())
                
                # Consider it a match if similarity is high enough
                if name_similarity > 60 or type_similarity > 70:
                    fuzzy_matches.append((row.to_dict(), max(name_similarity, type_similarity)))
            
            # Sort by similarity and take the best ones
            fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
            for match, score in fuzzy_matches[:remaining]:
                results.append(match)
                logger.info(f"✅ Found fuzzy match: {match['Name']} (score: {score})")
        
        # 5. Sort results by rating (descending) to show best restaurants first
        if results:
            results.sort(key=lambda x: float(x.get('Rating', 0)), reverse=True)
        
        logger.info(f"🎯 Total restaurants found: {len(results)}")
        return results[:max_results]
        
    except Exception as e:
        logger.error(f"❌ Error in restaurant search: {e}")
        return []

def format_restaurant_response(results: List[Dict], search_term: str) -> str:
    """Format restaurant search results with exact metro routes from CSV"""
    if not results:
        return f"عذراً، لم أجد مطاعم تطابق '{search_term}' في قاعدة البيانات."
    
    response_parts = []
    
    if len(results) == 1:
        restaurant = results[0]
        response_parts.append(f"وجدت لك مطعم {restaurant.get('Name', 'غير محدد')}")
        
        # Add rating if available
        if restaurant.get('Rating'):
            rating = float(restaurant['Rating'])
            response_parts.append(f"بتقييم {rating:.1f}")
        
        # Add cuisine type
        if restaurant.get('type'):
            response_parts.append(f"يقدم {restaurant['type']}")
        
        # Add price range
        if restaurant.get('price'):
            response_parts.append(f"والأسعار {restaurant['price']}")
        
        # Add exact metro route from CSV
        if restaurant.get('Path'):
            response_parts.append(f"المسار بالمترو: {restaurant['Path']}")
        
        # Add duration if available
        if restaurant.get('Duration(m)'):
            duration = int(restaurant['Duration(m)'])
            response_parts.append(f"المدة تقريباً {duration} دقيقة")
            
    else:
        response_parts.append(f"وجدت لك {len(results)} مطاعم:")
        
        for i, restaurant in enumerate(results, 1):
            name = restaurant.get('Name', 'غير محدد')
            rating = restaurant.get('Rating', 'غير محدد')
            cuisine = restaurant.get('type', 'غير محدد')
            price = restaurant.get('price', 'غير محدد')
            
            restaurant_info = f"{i}. {name}"
            if rating != 'غير محدد':
                restaurant_info += f" (تقييم {float(rating):.1f})"
            restaurant_info += f" - {cuisine} - {price}"
            
            # Add metro route for each restaurant
            if restaurant.get('Path'):
                restaurant_info += f" - المسار: {restaurant['Path']}"
            
            response_parts.append(restaurant_info)
    
    return ". ".join(response_parts) + "."

# ─── Gemini Live Audio System ─────────────────────────────────────────────────
class AudioManager:
    """Enhanced audio manager for real-time Gemini Live streaming"""
    def __init__(self):
        self.pya = None
        self.input_stream = None
        self.output_stream = None
        self.audio_queue = deque()
        self.playback_task = None
        
        # Audio settings optimized for Gemini Live and Raspberry Pi
        self.FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.CHANNELS = 1
        self.SEND_SAMPLE_RATE = 16000
        self.RECEIVE_SAMPLE_RATE = 24000
        
        # Adjust buffer size based on platform
        if RUNNING_ON_PI:
            self.CHUNK_SIZE = 1024  # Smaller buffer for Pi to reduce latency
            logger.info("🍓 Using Pi-optimized audio settings")
        else:
            self.CHUNK_SIZE = 2048  # Larger buffer for desktop to prevent underruns

    async def initialize(self):
        """Initialize audio streams for real-time processing"""
        try:
            if PYAUDIO_AVAILABLE:
                self.pya = pyaudio.PyAudio()
                
                # Get microphone info
                mic_info = self.pya.get_default_input_device_info()
                logger.info(f"🎤 Microphone: {mic_info['name']}")

                # Initialize input stream for microphone
                self.input_stream = await asyncio.to_thread(
                    self.pya.open,
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.SEND_SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK_SIZE * 2,  # مضاعفة buffer للميكروفون
                )

                # Initialize output stream for speakers
                self.output_stream = await asyncio.to_thread(
                    self.pya.open,
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RECEIVE_SAMPLE_RATE,
                    output=True,
                )
                
                logger.info("✅ Real-time audio streams initialized successfully")
            else:
                logger.info("✅ Using sounddevice fallback for audio")
                
        except Exception as e:
            logger.error(f"❌ Audio initialization failed: {e}")
            raise

    def add_audio(self, audio_data):
        """Add audio data to playback queue"""
        self.audio_queue.append(audio_data)
        if self.playback_task is None or self.playback_task.done():
            self.playback_task = asyncio.create_task(self._play_audio())

    async def _play_audio(self):
        """Play audio from queue with real-time streaming and immediate state updates"""
        global speaking
        
        # Set speaking state immediately
        with state_lock:
            speaking = True
        
        logger.info("🗣️ Gemini speaking - Video should switch to speaking mode")
        
        try:
            while self.audio_queue:
                try:
                    audio_data = self.audio_queue.popleft()
                    if PYAUDIO_AVAILABLE and self.output_stream:
                        await asyncio.to_thread(self.output_stream.write, audio_data)
                    else:
                        # Fallback to pygame/soundfile
                        await self._play_with_pygame(audio_data)
                except Exception as e:
                    logger.error(f"❌ Audio playback error: {e}")
                    break
        finally:
            # Always reset speaking state when done
            with state_lock:
                speaking = False
            logger.info("✅ Gemini finished speaking - Video should switch to silent mode")

    async def _play_with_pygame(self, audio_data):
        """Fallback audio playback using pygame"""
        try:
            # Convert audio data to playable format
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, np.frombuffer(audio_data, dtype=np.int16), self.RECEIVE_SAMPLE_RATE)
                
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=self.RECEIVE_SAMPLE_RATE, size=-16, channels=1, buffer=4096)  # زيادة buffer لمنع ALSA underrun
                
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.01)
                    
                os.unlink(tmp_file.name)
                
        except Exception as e:
            logger.error(f"❌ Pygame audio playback error: {e}")

    async def read_audio_chunk(self):
        """Read audio chunk from microphone"""
        try:
            if PYAUDIO_AVAILABLE and self.input_stream:
                data = await asyncio.to_thread(
                    self.input_stream.read,
                    self.CHUNK_SIZE,
                    exception_on_overflow=False,
                )
                return data
            else:
                # Fallback to sounddevice
                import sounddevice as sd
                duration = self.CHUNK_SIZE / self.SEND_SAMPLE_RATE
                audio = sd.rec(int(duration * self.SEND_SAMPLE_RATE), 
                             samplerate=self.SEND_SAMPLE_RATE, 
                             channels=1, 
                             dtype=np.int16)
                sd.wait()
                return audio.tobytes()
        except Exception as e:
            logger.error(f"❌ Audio read error: {e}")
            return None

    def cleanup(self):
        """Clean up audio resources"""
        try:
            if PYAUDIO_AVAILABLE and self.pya:
                if self.input_stream:
                    self.input_stream.stop_stream()
                    self.input_stream.close()
                if self.output_stream:
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                self.pya.terminate()
            logger.info("🧹 Audio cleanup completed")
        except Exception as e:
            logger.error(f"❌ Audio cleanup error: {e}")

class GeminiLiveSession:
    """Fast, real-time Gemini Live session with immediate audio response"""
    def __init__(self):
        self.session = None
        self.session_context = None
        self.audio_manager = None
        self.running = False
        self.greeting_sent = False
        self.greeting_requested = False

    async def start(self):
        """Start fast Gemini Live session with real-time audio streaming"""
        global gemini_session, current_route
        
        try:
            # Initialize high-performance audio manager
            self.audio_manager = AudioManager()
            await self.audio_manager.initialize()
            
            # Enhanced system prompt for fast restaurant search
            from google.genai.types import LiveConnectConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig
            
            config = LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=SpeechConfig(
                    voice_config=VoiceConfig(
                        prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Charon")
                    )
                ),
                system_instruction=GEMINI_LIVE_SYSTEM_PROMPT
            )
            
            logger.info("🚀 Starting fast Gemini Live session...")
            
            # Create session context with proper async handling
            self.session_context = client.aio.live.connect(model="gemini-2.5-flash-preview-native-audio-dialog", config=config)
            self.session = await self.session_context.__aenter__()
            
            logger.info("✅ Fast Gemini Live connected! Real-time voice interaction ready")
            
            self.running = True
            gemini_session = self
            
            # Start high-performance concurrent tasks
            await asyncio.gather(
                self._process_and_send_audio(),
                self._receive_and_play(),
                self._check_greeting_requests()
            )
                
        except Exception as e:
            logger.error(f"❌ Fast Gemini Live session error: {e}")
            await self.stop()

    async def _process_and_send_audio(self):
        """Process and send audio to Gemini in real-time with listening state updates"""
        global listening, speaking
        logger.info("🎤 Starting continuous audio processing...")
        
        while self.running:  # Always listen, not just when system_active
            try:
                # Update listening state when not speaking
                with state_lock:
                    if not speaking:
                        listening = True
                    else:
                        listening = False
                
                data = await self.audio_manager.read_audio_chunk()
                if data and self.session:
                    from google.genai.types import Blob
                    await self.session.send_realtime_input(
                        audio=Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
                    
            except Exception as e:
                logger.error(f"❌ Audio send error: {e}")
                await asyncio.sleep(0.1)

    async def _receive_and_play(self):
        """Receive and play audio responses from Gemini immediately"""
        global current_route, listening
        
        while self.running:
            try:
                async for response in self.session.receive():
                    server_content = response.server_content

                    if (hasattr(server_content, "interrupted") and server_content.interrupted):
                        logger.info("🤫 Interruption detected")
                        # Reset listening state when interrupted
                        with state_lock:
                            listening = False

                    if server_content and server_content.model_turn:
                        # When Gemini starts responding, we're not listening anymore
                        with state_lock:
                            listening = False
                            
                        for part in server_content.model_turn.parts:
                            # Handle audio response immediately for fast playback
                            if part.inline_data:
                                self.audio_manager.add_audio(part.inline_data.data)
                            
                            # Handle text response for restaurant processing and RAG
                            if part.text:
                                text_content = part.text
                                logger.info(f"📝 Gemini: {text_content}")
                                
                                # Extract restaurant and search database
                                restaurant_name = self._extract_restaurant_from_text(text_content)
                                if restaurant_name:
                                    best_match, score = self._find_best_restaurant_match(restaurant_name)
                                    if best_match and score >= 40:
                                        # Process restaurant route
                                        self._process_restaurant_route(best_match)
                                
                                # Also try RAG system for enhanced responses
                                try:
                                    rag_response = ask_rag_question(text_content)
                                    if rag_response and "للأسف" not in rag_response:
                                        logger.info(f"🧠 RAG response: {rag_response[:100]}...")
                                except Exception as e:
                                    logger.debug(f"RAG processing error: {e}")

                    if server_content and server_content.turn_complete:
                        logger.info("✅ Gemini turn complete - Ready to listen again")
                        # When Gemini finishes, we're ready to listen again
                        with state_lock:
                            listening = True
                        
            except Exception as e:
                logger.error(f"❌ Response handling error: {e}")
                await asyncio.sleep(1)

    def _extract_restaurant_from_text(self, text: str) -> str:
        """Extract restaurant name from Gemini's response"""
        try:
            patterns = extract_restaurant_patterns()
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else (match[1] if len(match) > 1 else "")
                    
                    result = clean_extracted_text(match)
                    
                    if len(result) > 2:
                        logger.info(f"✅ Restaurant extracted: '{result}'")
                        return result
            
            return ""
        except Exception as e:
            logger.error(f"Error extracting restaurant: {e}")
            return ""

    def _find_best_restaurant_match(self, query: str):
        """Find best restaurant match with enhanced scoring"""
        if not query.strip():
            return "", 0
        
        try:
            # Check shortcuts first
            shortcuts = get_restaurant_shortcuts()
            q_lower = query.lower()
            for shortcut, full_name in shortcuts.items():
                if shortcut.lower() in q_lower:
                    for _, row in restaurants_df.iterrows():
                        restaurant_name = str(row["Display_Name"])
                        if full_name.lower() in restaurant_name.lower():
                            logger.info(f"✅ Shortcut match: {restaurant_name}")
                            return restaurant_name, 95
            
            # Fuzzy matching
            result = process.extractOne(query, restaurants_df["Display_Name"].tolist())
            return result if result else ("", 0)
        except Exception as e:
            logger.error(f"Error finding restaurant match: {e}")
            return "", 0

    def _process_restaurant_route(self, restaurant_name):
        """Process restaurant route and update global state"""
        global current_route, current_route_text
        
        try:
            matching_rows = restaurants_df[restaurants_df["Display_Name"] == restaurant_name]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                instr = row["Path"]
                start = row["اسم المحطة"]
                end = row["Final Station"]
                
                # Store route text for map generation
                current_route_text = instr
                
                stations = self._path_to_stations(instr, start, end)
                if len(stations) >= 2:
                    current_route = stations
                    logger.info(f"🗺️ Route found: {stations}")
                    logger.info(f"🗺️ Route text: {current_route_text}")
                    
                    # Print detailed results
                    print(f"\n🎯 تم العثور على المطعم!")
                    print(f"📍 المطعم: {restaurant_name}")
                    print(f"🚇 المسار: {instr}")
                    print(f"🚉 المحطات:")
                    for i, station in enumerate(stations, 1):
                        print(f"  {i}. {station}")
                    print("-" * 50)
                    
        except Exception as e:
            logger.error(f"Error processing restaurant route: {e}")

    def _path_to_stations(self, path_text: str, start: str, end: str) -> list[str]:
        """Extract stations from path text"""
        try:
            found = re.findall(r"محطة\s*([\u0600-\u06FF0-9 ]+)", path_text)
            stations = [start]
            for nm in found:
                name = nm.strip()
                if name in (start, end) or name == "الحالية":
                    continue
                stations.append(name)
            stations.append(end)
            return stations
        except Exception as e:
            logger.error(f"Error extracting stations: {e}")
            return [start, end]

    async def _check_greeting_requests(self):
        """Check for greeting requests from one-shot face detection"""
        while self.running:
            try:
                if hasattr(self, 'greeting_requested') and self.greeting_requested and not self.greeting_sent:
                    logger.info("👋 Greeting requested - sending automatic greeting")
                    await self.send_greeting(STRONG_GREETING)
                    self.greeting_sent = True
                    self.greeting_requested = False  # Reset flag
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Greeting check error: {e}")
                await asyncio.sleep(1)

    async def send_greeting(self, greeting_text: str):
        """Send greeting message to Gemini Live as system instruction"""
        try:
            if self.session:
                # Send as system message to trigger response
                system_message = f"قل للمستخدم: {greeting_text}"
                await self.session.send_realtime_input(text=system_message)
                logger.info("👋 Greeting instruction sent to Gemini Live")
        except Exception as e:
            logger.error(f"❌ Error sending greeting: {e}")

    async def stop(self):
        """Stop Gemini Live session and cleanup"""
        self.running = False
        
        if hasattr(self, 'session_context') and self.session_context:
            try:
                await self.session_context.__aexit__(None, None, None)
            except:
                pass
                
        if self.audio_manager:
            self.audio_manager.cleanup()
            
        logger.info("🛑 Fast Gemini Live session stopped")

# ─── Video Display System ─────────────────────────────────────────────────────
def load_video_frames():
    """Load video frames with enhanced error handling and frame validation"""
    silent_frames = []
    speaking_frames = []
    default_frame = None
    
    try:
        # Load silent video
        if os.path.exists("siraj_silent.mp4"):
            logger.info("🎬 Loading silent video...")
            cap_silent = cv2.VideoCapture("siraj_silent.mp4")
            if cap_silent.isOpened():
                frame_count = 0
                while True:
                    ret, frame = cap_silent.read()
                    if not ret:
                        break
                    frame_resized = resize_frame_to_fit(frame, max_size=(1280, 720))
                    silent_frames.append(frame_resized)
                    frame_count += 1
                cap_silent.release()
                logger.info(f"✅ تم تحميل {len(silent_frames)} إطار صامت")
            else:
                logger.warning("❌ فشل في فتح siraj_silent.mp4")
        else:
            logger.warning("❌ ملف siraj_silent.mp4 غير موجود")
        
        # Load speaking video
        if os.path.exists("siraj_speak2.mp4"):
            logger.info("🎬 Loading speaking video...")
            cap_speaking = cv2.VideoCapture("siraj_speak2.mp4")
            if cap_speaking.isOpened():
                frame_count = 0
                while True:
                    ret, frame = cap_speaking.read()
                    if not ret:
                        break
                    frame_resized = resize_frame_to_fit(frame, max_size=(1280, 720))
                    speaking_frames.append(frame_resized)
                    frame_count += 1
                cap_speaking.release()
                logger.info(f"✅ تم تحميل {len(speaking_frames)} إطار متحدث")
            else:
                logger.warning("❌ فشل في فتح siraj_speak2.mp4")
        else:
            logger.warning("❌ ملف siraj_speak2.mp4 غير موجود")
        
        # Create default frame if videos fail
        if not silent_frames and not speaking_frames:
            default_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(default_frame, "Siraj Metro Assistant", (400, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
            logger.info("✅ تم إنشاء إطار افتراضي")
        
        # If only one video is available, use it for both states
        if silent_frames and not speaking_frames:
            speaking_frames = silent_frames.copy()
            logger.info("ℹ️ Using silent video for both states")
        elif speaking_frames and not silent_frames:
            silent_frames = speaking_frames.copy() 
            logger.info("ℹ️ Using speaking video for both states")
        
        return silent_frames, speaking_frames, default_frame
        
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل الفيديو: {e}")
        # Return default frame
        default_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(default_frame, "Siraj Metro Assistant", (400, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        return [], [], default_frame

# ─── Main Application ─────────────────────────────────────────────────────────
def main():
    """Main application with Gemini Live integration - supports both Desktop and Raspberry Pi"""
    global system_active, person_present, current_route, gemini_session
    
    logger.info("🚀 Starting Siraj - Clean Gemini Live Solution")
    
    # GPIO disabled by user request (user has screen display)
    # if RUNNING_ON_PI and GPIO_AVAILABLE:
    #     if not initialize_gpio():
    #         logger.warning("⚠️ GPIO initialization failed - continuing without hardware controls")
    logger.info("🎛️ GPIO controls disabled - using screen display interface")
    
    # Load data
    if not load_and_process_data():
        logger.error("❌ Failed to load restaurant data")
        return
    
    # Initialize RAG system
    initialize_rag_system()
    
    # Initialize face detection
    face_detection_available = initialize_face_detection()
    if face_detection_available:
        logger.info("✅ تم تفعيل كشف الوجوه")
    else:
        logger.warning("⚠️ كشف الوجوه غير متاح")
    
    # Check if running headless mode (only when absolutely necessary)
    no_display = os.environ.get('DISPLAY') is None
    ssh_session = os.environ.get('SSH_CLIENT') is not None or os.environ.get('SSH_TTY') is not None
    forced_headless = os.getenv("SIRAJ_HEADLESS", "false").lower() == "true"
    
    # Only force headless if explicitly requested or no display available
    headless_mode = forced_headless or (no_display and ssh_session)
    
    if RUNNING_ON_PI and not headless_mode:
        logger.info("🍓 Raspberry Pi detected with display - running GUI mode")
    if no_display and ssh_session:
        logger.info("🖥️ No DISPLAY + SSH detected - forcing headless mode")
    if forced_headless:
        logger.info("⚙️ SIRAJ_HEADLESS=true - forcing headless mode")
    
    if headless_mode:
        logger.info("🖥️ Running in headless mode (no GUI)")
        run_headless_mode()
    else:
        logger.info("🖥️ Running in desktop mode (with GUI)")
        run_desktop_mode()

def run_headless_mode():
    """Run Siraj in headless mode for Raspberry Pi"""
    global system_active, person_present, gemini_session
    
    # Initialize Gemini Live session
    gemini_session = GeminiLiveSession()
    
    # Start Gemini Live in background thread
    def start_gemini_live():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(gemini_session.start())
        loop.run_forever()
    
    gemini_thread = threading.Thread(target=start_gemini_live, daemon=True)
    gemini_thread.start()
    logger.info("🎤 Gemini Live started in background thread")
    
    # Wait a moment for the session to initialize
    time.sleep(3)
    
    # Start camera monitoring for face detection
    def run_one_shot_detection():
        time.sleep(2)  # Wait for Gemini session to initialize
        one_shot_face_detection(timeout_s=5.0)
    
    detection_thread = threading.Thread(target=run_one_shot_detection, daemon=True)
    detection_thread.start()
    
    # Set system active for Raspberry Pi
    system_active = True
    person_present = True
    
    # Status LED disabled - using screen display instead
    # set_status_led(True)
    
    logger.info("✅ Siraj ready for voice interaction on Raspberry Pi!")
    logger.info("🖥️ Screen display mode - GPIO controls disabled")
    logger.info("📱 Press Ctrl+C to stop")
    
    try:
        # Simple monitoring loop for headless mode
        while True:
            time.sleep(1)
            
            # Status LED updates disabled - using screen display instead
            # if speaking:
            #     set_status_led(True)
            # elif listening:
            #     # Blink LED when listening
            #     set_status_led(True)
            #     time.sleep(0.1)
            #     set_status_led(False)
            #     time.sleep(0.1)
            # else:
            #     set_status_led(system_active)
            pass
                
    except KeyboardInterrupt:
        logger.info("🛑 Headless mode shutdown requested")
    
    finally:
        # Cleanup
        try:
            def stop_session_sync():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(gemini_session.stop())
                loop.close()
            
            stop_thread = threading.Thread(target=stop_session_sync)
            stop_thread.start()
            stop_thread.join(timeout=5)  # Wait max 5 seconds
            
            # cleanup_gpio()  # GPIO disabled
            
        except Exception as e:
            logger.error(f"Error during headless cleanup: {e}")
        
        logger.info("👋 Siraj headless shutdown complete")

def run_desktop_mode():
    """Run Siraj in desktop mode with GUI"""
    global system_active, person_present, current_route, gemini_session
    
    # Load video frames
    silent_frames, speaking_frames, default_frame = load_video_frames()
    
    # Create GUI for full screen display
    sg.theme('DarkBlue3')
    
    layout = [
        [sg.Image(key="VIDEO", size=(1280, 720))],  # Full screen video
    ]
    
    window = sg.Window(
        "Siraj Metro Assistant",
        layout,
        finalize=True,
        resizable=False,
        location=(0, 0),     # Top-left corner
        size=(1280, 720),    # Full HD resolution
        no_titlebar=True,    # Remove title bar for full screen
        keep_on_top=True,    # Keep on top
        margins=(0, 0),      # No margins for full screen
        grab_anywhere=True   # Allow dragging for positioning
    )
    
    # Initialize Gemini Live session
    gemini_session = GeminiLiveSession()
    
    # Start background tasks
    def start_gemini_live():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(gemini_session.start())
        loop.run_forever()
    
    gemini_thread = threading.Thread(target=start_gemini_live, daemon=True)
    gemini_thread.start()
    logger.info("🎤 تم بدء معالج Gemini Live في thread منفصل")
    
    # Wait a moment for the session to initialize
    time.sleep(3)
    logger.info(f"🔍 حالة Gemini session: {gemini_session is not None}")
    logger.info(f"🔍 حالة Gemini session running: {gemini_session.running if gemini_session else False}")
    logger.info(f"🔍 GEMINI_LIVE_AVAILABLE: {GEMINI_LIVE_AVAILABLE}")
    
    # Start camera monitoring
    # One-shot face detection at startup only
    def run_one_shot_detection():
        time.sleep(2)  # Wait for Gemini session to initialize
        one_shot_face_detection(timeout_s=5.0)
    
    detection_thread = threading.Thread(target=run_one_shot_detection, daemon=True)
    detection_thread.start()
    
    # Advanced person detection removed - using one-shot detection only
    logger.info("🎥 نظام الكشف الواحد فقط - لا توجد خيوط كاميرا مستمرة")
    
    # Video display variables
    frame_index_silent = 0
    frame_index_speaking = 0
    last_frame_time = time.time()
    current_video_state = "silent"  # Track current video state
    
    # Initial status - always active for continuous conversation
    system_active = True
    person_present = True
    
    logger.info(f"🎬 Video frames loaded - Silent: {len(silent_frames)}, Speaking: {len(speaking_frames)}")
    
    try:
        while True:
            event, values = window.read(timeout=30)  # Faster update rate
            
            if event in (sg.WIN_CLOSED, "EXIT"):
                break
            
            # No button events to handle - clean interface for touch screen
            
            # Update video display
            current_time = time.time()
            if current_time - last_frame_time > 0.025:  # ~40 FPS for smoother video
                
                frame = None
                status_color = (255, 255, 255)
                status_text = "⚡ سراج جاهز"
                
                # Choose frame based on state with immediate switching
                with state_lock:
                    current_speaking = speaking
                    current_listening = listening
                
                if current_speaking and speaking_frames:
                    # Speaking state - use speaking video
                    frame = speaking_frames[frame_index_speaking % len(speaking_frames)]
                    frame_index_speaking += 1
                    status_color = (0, 255, 0)  # Green for speaking
                    status_text = "🗣️ سراج يتحدث"
                    if current_video_state != "speaking":
                        current_video_state = "speaking"
                        frame_index_speaking = 0  # Reset frame index when switching
                        logger.info("🎬 تبديل إلى فيديو الكلام")
                
                elif current_listening and silent_frames:
                    # Listening state - use silent video
                    frame = silent_frames[frame_index_silent % len(silent_frames)]
                    frame_index_silent += 1
                    status_color = (255, 255, 0)  # Yellow for listening
                    status_text = "👂 سراج يستمع"
                    if current_video_state != "listening":
                        current_video_state = "listening"
                        frame_index_silent = 0  # Reset frame index when switching
                        logger.info("🎬 تبديل إلى فيديو الاستماع")
                
                else:
                    # Ready/Idle state - use silent video
                    if silent_frames:
                        frame = silent_frames[frame_index_silent % len(silent_frames)]
                        frame_index_silent += 1
                    else:
                        frame = default_frame
                    status_color = (255, 255, 255)  # White for ready
                    status_text = "⚡ سراج جاهز"
                    if current_video_state != "silent":
                        current_video_state = "silent"
                        frame_index_silent = 0  # Reset frame index when switching
                        logger.info("🎬 تبديل إلى فيديو الصمت")
                
                # Add status overlay to frame
                if frame is not None:
                    # Create a copy to avoid modifying original
                    display_frame = frame.copy()
                    
                    # Add status text overlay with larger, clearer display
                    cv2.rectangle(display_frame, (10, 10), (500, 80), (0, 0, 0), -1)  # Black background
                    cv2.rectangle(display_frame, (10, 10), (500, 80), status_color, 3)  # Colored border
                    
                    # Add current state text
                    state_text = "Speaking" if current_speaking else "Listening" if current_listening else "Ready"
                    cv2.putText(display_frame, f"State: {state_text}", 
                               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    
                    # Add frame counter for debugging
                    if current_speaking and speaking_frames:
                        cv2.putText(display_frame, f"Speaking Frame: {frame_index_speaking}/{len(speaking_frames)}", 
                                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    elif silent_frames:
                        cv2.putText(display_frame, f"Silent Frame: {frame_index_silent}/{len(silent_frames)}", 
                                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Add Gemini session status
                    gemini_status = "Connected" if (gemini_session and gemini_session.running) else "Disconnected"
                    cv2.putText(display_frame, f"Gemini: {gemini_status}", 
                               (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Encode and display
                    imgbytes = cv2.imencode('.png', display_frame)[1].tobytes()
                    window["VIDEO"].update(data=imgbytes)
                
                last_frame_time = current_time
    
    except Exception as e:
        logger.error(f"❌ Main loop error: {e}")
    
    finally:
        # Cleanup
        try:
            def stop_session_sync():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(gemini_session.stop())
                loop.close()
            
            stop_thread = threading.Thread(target=stop_session_sync)
            stop_thread.start()
            stop_thread.join(timeout=5)  # Wait max 5 seconds
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        window.close()
        # cleanup_gpio()  # GPIO disabled by user request
        logger.info("👋 Siraj desktop shutdown complete")

def start_camera_monitor():
    """Simple camera monitoring with error handling"""
    global person_present, system_active
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("❌ Camera not available - auto-activating")
            person_present = True
            system_active = True
            return
        
        # Try to load face cascade with error handling
        face_cascade = None
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                if face_cascade.empty():
                    face_cascade = None
        except Exception as e:
            logger.warning(f"⚠️ فشل في تحميل كاشف الوجوه في الكاميرا البسيطة: {e}")
            face_cascade = None
        
        if face_cascade is None:
            logger.warning("❌ No face detection available - auto-activating")
            person_present = True
            system_active = True
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                person_present = len(faces) > 0
                system_active = person_present
            except Exception as e:
                logger.debug(f"Detection error in simple monitor: {e}")
                # Continue without detection
            
            time.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Camera error: {e}")
        person_present = True
        system_active = True

# ─── Metro Network and Map Functions ─────────────────────────────────────────
def build_metro_network(stations_gdf, lines_gdf):
    """Build metro network graph from GeoJSON data"""
    G = nx.Graph()
    
    # Add stations as nodes
    for _, station in stations_gdf.iterrows():
        G.add_node(
            station['metrostationnamear'],
            pos=(station.geometry.y, station.geometry.x),
            line=station['metroline'],
            type='station'
        )
    
    # Connect stations using actual line geometry
    for line_id in lines_gdf['metroline'].unique():
        line_geom = lines_gdf[lines_gdf['metroline'] == line_id].geometry.iloc[0]
        line_stations = stations_gdf[stations_gdf['metroline'] == line_id]
        
        if isinstance(line_geom, LineString) and len(line_stations) > 1:
            # Order stations by their position on the line
            stations_on_line = []
            for _, station in line_stations.iterrows():
                point = station.geometry
                progress = line_geom.project(point)
                stations_on_line.append({
                    'name': station['metrostationnamear'],
                    'progress': progress
                })
            
            ordered_stations = sorted(stations_on_line, key=lambda x: x['progress'])
            
            # Connect consecutive stations
            for i in range(len(ordered_stations)-1):
                start_station = ordered_stations[i]['name']
                end_station = ordered_stations[i+1]['name']
                
                # Create a simple connection
                G.add_edge(
                    start_station,
                    end_station,
                    line=line_id,
                    type='line_connection'
                )
    
    return G

def get_line_colors():
    """Get metro line colors"""
    return {
        'Metro 1': '#00ade5',
        'Metro 2': '#f0493a', 
        'Metro 3': '#f68d39',
        'Metro 4': '#ffd105',
        'Metro 5': '#43b649',
        'Metro 6': '#984c9d',
    }

def draw_metro_path_folium(G, start, end, via=[]):
    """Draw metro path using Folium for interactive map"""
    if not FOLIUM_AVAILABLE:
        logger.warning("Folium not available for interactive maps")
        return None
        
    try:
        full_path = [start] + via + [end]
        full_route = []
        total_distance = 0
        
        # Build complete route
        for i in range(len(full_path) - 1):
            try:
                segment = nx.shortest_path(G, source=full_path[i], target=full_path[i+1])
                full_route += segment[:-1]  # Avoid duplication
            except nx.NetworkXNoPath:
                logger.error(f"No path found between {full_path[i]} and {full_path[i+1]}")
                return None
        full_route.append(end)
        
        # Create map centered on start
        start_coord = G.nodes[start]['pos']
        m = folium.Map(location=start_coord, zoom_start=11, tiles='CartoDB Positron')
        
        line_colors = get_line_colors()
        
        # Draw route segments
        for i in range(len(full_route) - 1):
            u = full_route[i]
            v = full_route[i + 1]
            
            if G.has_edge(u, v):
                edge = G[u][v]
                color = line_colors.get(edge['line'], 'gray')
                
                # Get coordinates of both stations
                start_pos = G.nodes[u]['pos']
                end_pos = G.nodes[v]['pos']
                
                # Draw simple line between stations
                folium.PolyLine(
                    [start_pos, end_pos], 
                    color=color, 
                    weight=6, 
                    opacity=0.8
                ).add_to(m)
                
        return m
        
    except Exception as e:
        logger.error(f"Error creating Folium map: {e}")
        return None

def text_to_map(G, route_text):
    """Convert route text to interactive map"""
    try:
        start = ''
        end = ''
        via = []
        
        parts = route_text.split("،")
        for part in parts:
            if "المحطة الحالية" in part:
                continue
            elif "بدّل إلى المسار" in part:
                station = part.split("محطة")[1].split("بدّل")[0].strip()
                via.append(station)
            elif "انتقل إلى المسار" in part:
                station = ' '.join(part.split("إلى")[0].split("محطة")[-1].strip().split(" ")[:-1])
                via.append(station)
            elif "أخرج عند" in part:
                station = part.split("محطة")[-1].strip().split('.')[0]
                end = station
        
        # Use default start if not specified
        if not start:
            start = "جامعة الأميرة نورة 2"  # Default start station
            
        if end:
            return draw_metro_path_folium(G, start, end, via)
        else:
            logger.warning("No end station found in route text")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing route text: {e}")
        return None

def render_map_as_array(route: list[str]):
    """Enhanced map rendering with window display instead of browser"""
    global metro_network, current_route_text
    
    try:
        logger.info(f"🗺️ Attempting to render map for route: {route}")
        
        # Check if we have the necessary data
        if not metro_network:
            logger.warning("❌ Metro network not available")
            return False
            
        if not route or len(route) < 2:
            logger.warning("❌ Invalid route - need at least 2 stations")
            return False
        
        # Create map using OpenCV for window display
        logger.info("🗺️ Creating map window...")
        success = draw_metro_map_opencv_window(metro_network, route)
        
        if success:
            logger.info("✅ Map displayed in window")
            return True
        else:
            logger.warning("❌ Failed to create map window")
            return False
        
    except Exception as e:
        logger.error(f"❌ Map rendering error: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return False

def draw_metro_map_opencv_window(G, route):
    """Draw metro map using OpenCV in a window"""
    try:
        if not route or len(route) < 2:
            logger.warning("❌ Invalid route for map display")
            return False
            
        # Find path
        full_route = []
        for i in range(len(route) - 1):
            try:
                segment = nx.shortest_path(G, source=route[i], target=route[i+1])
                full_route += segment[:-1]  # Avoid duplication
            except nx.NetworkXNoPath:
                logger.error(f"No path found between {route[i]} and {route[i+1]}")
                return False
        full_route.append(route[-1])
        
        logger.info(f"📍 Full route: {full_route[:3]}...{full_route[-3:]} ({len(full_route)} stations)")
        
        # Create image
        img_width, img_height = 1000, 700
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # Get bounds for scaling
        all_positions = [G.nodes[node]['pos'] for node in G.nodes()]
        min_lat = min(pos[0] for pos in all_positions)
        max_lat = max(pos[0] for pos in all_positions)
        min_lon = min(pos[1] for pos in all_positions)
        max_lon = max(pos[1] for pos in all_positions)
        
        # Scale coordinates to image size
        def scale_coords(lat, lon):
            x = int((lon - min_lon) / (max_lon - min_lon) * (img_width - 100) + 50)
            y = int((max_lat - lat) / (max_lat - min_lat) * (img_height - 100) + 50)
            return x, y
        
        line_colors_bgr = {
            'Metro 1': (229, 173, 0),    # Blue
            'Metro 2': (58, 73, 240),    # Red
            'Metro 3': (57, 141, 246),   # Orange
            'Metro 4': (5, 209, 255),    # Yellow
            'Metro 5': (73, 182, 67),    # Green
            'Metro 6': (157, 76, 152),   # Purple
        }
        
        # Draw all metro lines (background)
        for u, v, data in G.edges(data=True):
            pos_u = scale_coords(*G.nodes[u]['pos'])
            pos_v = scale_coords(*G.nodes[v]['pos'])
            cv2.line(img, pos_u, pos_v, (200, 200, 200), 1)
        
        # Draw route segments (highlighted)
        for i in range(len(full_route) - 1):
            u = full_route[i]
            v = full_route[i + 1]
            
            if G.has_edge(u, v):
                edge = G[u][v]
                color = line_colors_bgr.get(edge['line'], (128, 128, 128))
                
                pos_u = scale_coords(*G.nodes[u]['pos'])
                pos_v = scale_coords(*G.nodes[v]['pos'])
                
                cv2.line(img, pos_u, pos_v, color, 4)
        
        # Draw all stations (small circles)
        for node in G.nodes():
            pos = scale_coords(*G.nodes[node]['pos'])
            cv2.circle(img, pos, 3, (180, 180, 180), -1)
        
        # Draw route stations (highlighted)
        for i, node in enumerate(full_route):
            pos = scale_coords(*G.nodes[node]['pos'])
            if i == 0:  # Start station
                cv2.circle(img, pos, 10, (0, 255, 0), -1)  # Green
                cv2.circle(img, pos, 10, (255, 255, 255), 2)  # White border
                cv2.putText(img, "START", (pos[0]-25, pos[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif i == len(full_route) - 1:  # End station
                cv2.circle(img, pos, 10, (0, 0, 255), -1)  # Red
                cv2.circle(img, pos, 10, (255, 255, 255), 2)  # White border
                cv2.putText(img, "END", (pos[0]-20, pos[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:  # Intermediate stations
                cv2.circle(img, pos, 6, (255, 0, 0), -1)  # Blue
                cv2.circle(img, pos, 6, (255, 255, 255), 1)  # White border
        
        # Add title and info
        cv2.putText(img, "Riyadh Metro Route Map", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(img, f"Route: {len(full_route)} stations", (50, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, "Press any key to close", (50, img_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Show the image in a separate thread to avoid blocking
        def show_map_window():
            cv2.imshow("Metro Route Map - Siraj", img)
            cv2.waitKey(5000)  # Show for 5 seconds or until key press
            cv2.destroyAllWindows()
        
        import threading
        map_thread = threading.Thread(target=show_map_window, daemon=True)
        map_thread.start()
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating OpenCV map: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

# ─── System Service Support for Raspberry Pi ─────────────────────────────────
def create_systemd_service():
    """Create systemd service for Raspberry Pi auto-start"""
    service_content = """[Unit]
Description=Siraj Arabic Voice Assistant for Riyadh Metro
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/siraj_gemini
Environment=PATH=/home/pi/siraj_gemini/venv/bin
Environment=SIRAJ_HEADLESS=true
ExecStart=/home/pi/siraj_gemini/venv/bin/python /home/pi/siraj_gemini/full_inegration.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    try:
        with open('/tmp/siraj.service', 'w') as f:
            f.write(service_content)
        
        print("Service file created at /tmp/siraj.service")
        print("To install, run:")
        print("sudo cp /tmp/siraj.service /etc/systemd/system/")
        print("sudo systemctl daemon-reload")
        print("sudo systemctl enable siraj.service")
        print("sudo systemctl start siraj.service")
        
    except Exception as e:
        logger.error(f"Failed to create service file: {e}")

def print_pi_setup_commands():
    """Print setup commands for Raspberry Pi"""
    commands = """
🍓 Raspberry Pi Setup Commands:

# Install system dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git -y
sudo apt install libportaudio2 libportaudio-dev libasound2-dev -y
sudo apt install libopencv-dev python3-opencv -y

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install google-genai python-dotenv pandas opencv-python
pip install pyaudio pygame soundfile RPi.GPIO
pip install fuzzywuzzy python-Levenshtein loguru

# Set environment variables
echo "GEMINI_API_KEY=your_key_here" > .env
echo "SIRAJ_HEADLESS=true" >> .env

# Run Siraj
python3 full_inegration.py

# Or create systemd service
python3 -c "from full_inegration import create_systemd_service; create_systemd_service()"
"""
    print(commands)

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-service":
            create_systemd_service()
        elif sys.argv[1] == "--pi-setup":
            print_pi_setup_commands()
        elif sys.argv[1] == "--headless":
            os.environ["SIRAJ_HEADLESS"] = "true"
            main()
        else:
            print("Usage:")
            print("  python3 full_inegration.py                 # Normal mode")
            print("  python3 full_inegration.py --headless      # Headless mode")
            print("  python3 full_inegration.py --create-service # Create systemd service")
            print("  python3 full_inegration.py --pi-setup      # Show Pi setup commands")
    else:
        main() 