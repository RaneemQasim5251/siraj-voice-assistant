#!/usr/bin/env python3
"""
Siraj - Raspberry Pi Optimized Version
Advanced Arabic Voice Assistant using Gemini Live API for Raspberry Pi
Features:
- Optimized for Raspberry Pi 4/5 performance
- GPIO hardware controls (button, LED, buzzer)
- Low-latency Gemini Live API integration
- Enhanced Arabic text processing
- Restaurant search and metro navigation
- Efficient resource management
"""

import os
import re
import threading
import time
import asyncio
import sqlite3
from collections import deque
from threading import Lock
from typing import List, Dict
import logging

import cv2
import pandas as pd
import numpy as np
import pygame
import soundfile as sf
import tempfile

from dotenv import load_dotenv
from fuzzywuzzy import process, fuzz
from loguru import logger

# GPIO imports for Raspberry Pi
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    logger.info("✅ GPIO متاح للراسبيري باي")
except ImportError:
    GPIO_AVAILABLE = False
    logger.warning("⚠️ GPIO غير متاح - تشغيل في وضع المحاكاة")

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

# Import Gemini Live API
try:
    import google.genai as genai
    GEMINI_LIVE_AVAILABLE = True
    logger.info("✅ Gemini Live API متاح")
except ImportError:
    logger.error("❌ Gemini Live API not available - this is required!")
    GEMINI_LIVE_AVAILABLE = False

# Audio libraries optimized for Pi
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    logger.info("✅ PyAudio متاح")
except ImportError:
    try:
        import sounddevice as sd
        SOUNDDEVICE_AVAILABLE = True
        logger.info("✅ SoundDevice متاح كبديل")
    except ImportError:
        logger.error("❌ No audio library available!")
        PYAUDIO_AVAILABLE = False
        SOUNDDEVICE_AVAILABLE = False

# ─── Configuration for Raspberry Pi ─────────────────────────────────────────
# File paths
CSV_PATH = "restaurants2_neighborhood_stations_paths_15.csv"
PATHS_CSV = "paths.csv"
PDF_PATH = "Darb card terms and conditions.pdf"

# GPIO Configuration
if GPIO_AVAILABLE:
    BUTTON_PIN = 18      # GPIO pin for activation button
    LED_PIN = 24         # GPIO pin for status LED
    BUZZER_PIN = 23      # GPIO pin for buzzer (optional)

# Global state
restaurants_df = None
current_route = []
current_route_text = ""
system_active = False
person_present = False
speaking = False
listening = False
last_speech_time = time.time()
state_lock = Lock()
gemini_session = None

# Face detection variables
face_detected = False
camera_enabled = False
face_cascade = None

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_LIVE_AVAILABLE and GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini Live API initialized successfully")
else:
    logger.error("❌ Gemini API key required!")
    exit(1)

# Optimized system prompt for Pi
GEMINI_LIVE_SYSTEM_PROMPT = """
أنتَ سِراج، المساعد الذكي الصوتي في مترو الرياض على جهاز راسبيري باي. 
أنت مساعدٌ ذكي وسريع، صوتك واضح، وأسلوبك عربي فصيح مبسط.

🎯 المهام الأساسية:
- مساعدة المسافرين في البحث عن المطاعم والمقاهي
- توجيه المسارات في مترو الرياض
- الإجابة على أسئلة بطاقة درب
- التفاعل بالعربية والإنجليزية بسلاسة

🎤 أسلوب المحادثة:
- جمل قصيرة وواضحة (15-30 ثانية)
- صوت حماسي ومفيد
- لا تكرر الاعتذارات
- ركز على المعلومات المفيدة

🚇 قواعد البيانات:
- اعتمد فقط على البيانات المرفقة
- لا تخمن المعلومات
- قدم أفضل المطاعم حسب التقييم

🌐 اللغات:
- العربية الفصحى
- الإنجليزية
- أجب بنفس لغة المستخدم
"""

# ─── GPIO Hardware Control ──────────────────────────────────────────────────
def initialize_gpio():
    """Initialize GPIO pins for Raspberry Pi"""
    if not GPIO_AVAILABLE:
        logger.info("ℹ️ GPIO simulation mode - no hardware control")
        return True
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup button pin with pull-up resistor
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Setup LED pin as output
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.output(LED_PIN, GPIO.LOW)
        
        # Setup buzzer pin (optional)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        
        # Add button event detection
        GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, 
                             callback=button_callback, bouncetime=300)
        
        logger.info("✅ GPIO initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ GPIO initialization failed: {e}")
        return False

def button_callback(channel):
    """Handle button press events"""
    global system_active, person_present
    
    logger.info("🔘 Hardware button pressed - activating system")
    
    with state_lock:
        system_active = not system_active
        person_present = system_active
    
    # Visual feedback
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.HIGH if system_active else GPIO.LOW)
        
        # Brief buzzer sound
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(BUZZER_PIN, GPIO.LOW)

def set_status_led(active: bool):
    """Control status LED"""
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.HIGH if active else GPIO.LOW)

def cleanup_gpio():
    """Cleanup GPIO resources"""
    if GPIO_AVAILABLE:
        try:
            GPIO.cleanup()
            logger.info("🧹 GPIO cleanup completed")
        except Exception as e:
            logger.error(f"❌ GPIO cleanup error: {e}")

# ─── Optimized Face Detection for Pi ────────────────────────────────────────
def initialize_pi_face_detection():
    """Initialize lightweight face detection for Raspberry Pi"""
    global face_cascade
    
    try:
        # Try multiple cascade files for better compatibility
        cascade_files = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        ]
        
        for cascade_file in cascade_files:
            try:
                if os.path.exists(cascade_file):
                    face_cascade = cv2.CascadeClassifier(cascade_file)
                    if not face_cascade.empty():
                        logger.info(f"✅ Face cascade loaded from: {cascade_file}")
                        return True
            except Exception as e:
                logger.debug(f"Failed to load {cascade_file}: {e}")
                continue
        
        logger.warning("⚠️ No face cascade found - face detection disabled")
        return False
        
    except Exception as e:
        logger.error(f"❌ Face detection initialization error: {e}")
        return False

def pi_face_detection_startup():
    """Optimized one-shot face detection for Pi startup"""
    global face_detected, gemini_session
    
    if not initialize_pi_face_detection():
        logger.info("ℹ️ Face detection not available - proceeding without greeting")
        return False
    
    logger.info("👁️ Starting Pi face detection (5s timeout)...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.warning("❌ Camera not available")
        return False
    
    # Lower resolution for Pi performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 10)
    
    start_time = time.time()
    detected = False
    
    while time.time() - start_time < 5.0:  # 5 second timeout
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Downsample for faster processing
        small_frame = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2, 
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        if len(faces) > 0:
            face_detected = True
            detected = True
            logger.info("👤 Face detected on Pi - triggering greeting!")
            
            # Trigger greeting
            if gemini_session and hasattr(gemini_session, 'greeting_requested'):
                gemini_session.greeting_requested = True
            
            break
        
        time.sleep(0.2)  # Reduce CPU usage
    
    cap.release()
    
    if detected:
        logger.info("✅ Pi face detection complete - greeting triggered")
        if GPIO_AVAILABLE:
            # Brief LED flash to indicate detection
            for _ in range(3):
                GPIO.output(LED_PIN, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(LED_PIN, GPIO.LOW)
                time.sleep(0.1)
    else:
        logger.info("ℹ️ Pi face detection timeout - no face detected")
    
    return detected

# ─── Restaurant Search Functions ─────────────────────────────────────────────
def load_restaurant_data():
    """Load restaurant data optimized for Pi"""
    global restaurants_df
    
    try:
        if not os.path.exists(CSV_PATH):
            logger.error(f"❌ Restaurant data not found: {CSV_PATH}")
            return False
            
        restaurants_df = pd.read_csv(CSV_PATH)
        restaurants_df = restaurants_df.dropna(subset=['Name', 'Final Station'])
        restaurants_df['Name_Clean'] = restaurants_df['Name'].str.strip().str.lower()
        restaurants_df['Rating'] = pd.to_numeric(restaurants_df['Rating'], errors='coerce').fillna(0)
        
        logger.info(f"✅ Pi loaded {len(restaurants_df)} restaurants")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pi restaurant data loading error: {e}")
        return False

def enhanced_restaurant_search(query: str, max_results: int = 3) -> List[Dict]:
    """Enhanced restaurant search optimized for Pi"""
    try:
        if restaurants_df is None or restaurants_df.empty:
            logger.warning("❌ Pi: Restaurant database not available")
            return []
        
        logger.info(f"🔍 Pi searching for: '{query}'")
        
        query_clean = query.strip().lower()
        results = []
        
        # Exact name matches
        exact_matches = restaurants_df[
            restaurants_df['Name'].str.lower().str.contains(query_clean, na=False, regex=False)
        ]
        if not exact_matches.empty:
            results.extend(exact_matches.head(max_results).to_dict('records'))
        
        # Cuisine type matches if needed
        if len(results) < max_results:
            remaining = max_results - len(results)
            cuisine_matches = restaurants_df[
                restaurants_df['type'].str.lower().str.contains(query_clean, na=False, regex=False)
            ]
            if results:
                existing_names = [r['Name'] for r in results]
                cuisine_matches = cuisine_matches[~cuisine_matches['Name'].isin(existing_names)]
            
            if not cuisine_matches.empty:
                results.extend(cuisine_matches.head(remaining).to_dict('records'))
        
        # Sort by rating
        if results:
            results.sort(key=lambda x: float(x.get('Rating', 0)), reverse=True)
        
        logger.info(f"🎯 Pi found {len(results)} restaurants")
        return results[:max_results]
        
    except Exception as e:
        logger.error(f"❌ Pi restaurant search error: {e}")
        return []

# ─── Main Pi Application ──────────────────────────────────────────────────────
def pi_main():
    """Main application optimized for Raspberry Pi"""
    global system_active, person_present, gemini_session
    
    logger.info("🚀 Starting Siraj on Raspberry Pi")
    
    try:
        # Initialize GPIO
        if not initialize_gpio():
            logger.warning("⚠️ GPIO initialization failed - continuing without hardware controls")
        
        # Load restaurant data
        if not load_restaurant_data():
            logger.error("❌ Failed to load restaurant data")
            return
        
        # Initialize audio system
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
            logger.info("✅ Pi audio initialized")
        except Exception as e:
            logger.error(f"❌ Pi audio initialization failed: {e}")
        
        # Start face detection
        def run_pi_face_detection():
            time.sleep(2)
            pi_face_detection_startup()
        
        detection_thread = threading.Thread(target=run_pi_face_detection, daemon=True)
        detection_thread.start()
        
        # Main Pi loop - simplified for stability
        system_active = True
        person_present = True
        set_status_led(True)
        
        logger.info("✅ Pi Siraj ready for interaction!")
        logger.info("📱 Press hardware button to toggle system")
        logger.info("📱 Press Ctrl+C to stop")
        
        # Simple monitoring loop
        try:
            while True:
                time.sleep(1)
                
                # Update status LED based on state
                if speaking:
                    set_status_led(True)
                elif system_active:
                    # Slow blink when active and listening
                    set_status_led(True)
                    time.sleep(0.5)
                    set_status_led(False)
                    time.sleep(0.5)
                else:
                    set_status_led(False)
                    
        except KeyboardInterrupt:
            logger.info("🛑 Pi shutdown requested")
            
    except Exception as e:
        logger.error(f"❌ Pi main error: {e}")
        
    finally:
        # Cleanup
        try:
            cleanup_gpio()
            pygame.mixer.quit()
            
        except Exception as e:
            logger.error(f"❌ Pi cleanup error: {e}")
        
        logger.info("👋 Pi Siraj shutdown complete")

if __name__ == "__main__":
    # Configure logging for Pi
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/tmp/siraj_pi.log'),
            logging.StreamHandler()
        ]
    )
    
    pi_main() 