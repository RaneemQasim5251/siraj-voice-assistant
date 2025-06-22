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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/siraj_pi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# Strong greeting message for face detection
STRONG_GREETING = "السَّلامُ عَلَيْكُمْ، أَهْلاً وَسَهْلاً بِك، أَنَا سِراجْ، مُساعِدُكَ الذَّكِيُّ لِمِتْرُوالرِّيَاض!"

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

# ─── Optimized Audio System for Pi ──────────────────────────────────────────
class PiAudioManager:
    """Optimized audio manager for Raspberry Pi"""
    def __init__(self):
        self.pya = None
        self.input_stream = None
        self.output_stream = None
        self.audio_queue = deque()
        self.playback_task = None
        
        # Pi-optimized audio settings
        self.FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.CHANNELS = 1
        self.SEND_SAMPLE_RATE = 16000
        self.RECEIVE_SAMPLE_RATE = 24000
        self.CHUNK_SIZE = 1024  # Smaller chunks for Pi

    async def initialize(self):
        """Initialize audio for Pi with error handling"""
        try:
            if PYAUDIO_AVAILABLE:
                self.pya = pyaudio.PyAudio()
                
                # Initialize input stream
                self.input_stream = await asyncio.to_thread(
                    self.pya.open,
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.SEND_SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK_SIZE,
                )

                # Initialize output stream
                self.output_stream = await asyncio.to_thread(
                    self.pya.open,
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RECEIVE_SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=self.CHUNK_SIZE,
                )
                
                logger.info("✅ Pi audio streams initialized")
                
            elif SOUNDDEVICE_AVAILABLE:
                logger.info("✅ Using SoundDevice for Pi audio")
                
        except Exception as e:
            logger.error(f"❌ Pi audio initialization failed: {e}")
            raise

    def add_audio(self, audio_data):
        """Add audio to playback queue with Pi optimization"""
        self.audio_queue.append(audio_data)
        if self.playback_task is None or self.playback_task.done():
            self.playback_task = asyncio.create_task(self._play_audio())

    async def _play_audio(self):
        """Optimized audio playback for Pi"""
        global speaking
        
        with state_lock:
            speaking = True
        
        set_status_led(True)  # Visual feedback
        logger.info("🗣️ Pi playing Gemini audio...")
        
        while self.audio_queue:
            try:
                audio_data = self.audio_queue.popleft()
                
                if PYAUDIO_AVAILABLE and self.output_stream:
                    await asyncio.to_thread(self.output_stream.write, audio_data)
                else:
                    await self._play_with_pygame(audio_data)
                    
            except Exception as e:
                logger.error(f"❌ Pi audio playback error: {e}")
                break
        
        with state_lock:
            speaking = False
        
        set_status_led(False)
        logger.info("✅ Pi audio playback complete")

    async def _play_with_pygame(self, audio_data):
        """Pygame fallback optimized for Pi"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, np.frombuffer(audio_data, dtype=np.int16), 
                        self.RECEIVE_SAMPLE_RATE)
                
                if not pygame.mixer.get_init():
                    pygame.mixer.init(
                        frequency=self.RECEIVE_SAMPLE_RATE, 
                        size=-16, 
                        channels=1, 
                        buffer=2048  # Pi-optimized buffer
                    )
                
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.05)
                
                os.unlink(tmp_file.name)
                
        except Exception as e:
            logger.error(f"❌ Pi pygame audio error: {e}")

    async def read_audio_chunk(self):
        """Read audio optimized for Pi"""
        try:
            if PYAUDIO_AVAILABLE and self.input_stream:
                data = await asyncio.to_thread(
                    self.input_stream.read,
                    self.CHUNK_SIZE,
                    exception_on_overflow=False,
                )
                return data
            elif SOUNDDEVICE_AVAILABLE:
                duration = self.CHUNK_SIZE / self.SEND_SAMPLE_RATE
                audio = sd.rec(
                    int(duration * self.SEND_SAMPLE_RATE),
                    samplerate=self.SEND_SAMPLE_RATE,
                    channels=1,
                    dtype=np.int16
                )
                sd.wait()
                return audio.tobytes()
            else:
                logger.warning("⚠️ No audio input available")
                return None
                
        except Exception as e:
            logger.error(f"❌ Pi audio read error: {e}")
            return None

    def cleanup(self):
        """Cleanup Pi audio resources"""
        try:
            if PYAUDIO_AVAILABLE and self.pya:
                if self.input_stream:
                    self.input_stream.stop_stream()
                    self.input_stream.close()
                if self.output_stream:
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                self.pya.terminate()
            logger.info("🧹 Pi audio cleanup completed")
        except Exception as e:
            logger.error(f"❌ Pi audio cleanup error: {e}")

# ─── Pi-Optimized Gemini Live Session ────────────────────────────────────────
class PiGeminiLiveSession:
    """Raspberry Pi optimized Gemini Live session"""
    def __init__(self):
        self.session = None
        self.session_context = None
        self.audio_manager = None
        self.running = False
        self.greeting_sent = False
        self.greeting_requested = False

    async def start(self):
        """Start Pi-optimized Gemini Live session"""
        global gemini_session
        
        try:
            # Initialize Pi audio manager
            self.audio_manager = PiAudioManager()
            await self.audio_manager.initialize()
            
            # Gemini Live configuration
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
            
            logger.info("🚀 Starting Pi Gemini Live session...")
            
            self.session_context = client.aio.live.connect(
                model="gemini-2.5-flash-preview-native-audio-dialog", 
                config=config
            )
            self.session = await self.session_context.__aenter__()
            
            logger.info("✅ Pi Gemini Live connected!")
            
            self.running = True
            gemini_session = self
            
            # Start Pi-optimized tasks
            await asyncio.gather(
                self._pi_process_audio(),
                self._pi_receive_responses(),
                self._pi_check_greeting()
            )
                
        except Exception as e:
            logger.error(f"❌ Pi Gemini Live session error: {e}")
            await self.stop()

    async def _pi_process_audio(self):
        """Pi-optimized audio processing"""
        logger.info("🎤 Pi audio processing started...")
        
        while self.running:
            try:
                # Always listen for continuous interaction
                data = await self.audio_manager.read_audio_chunk()
                if data and self.session:
                    from google.genai.types import Blob
                    await self.session.send_realtime_input(
                        audio=Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
                    
            except Exception as e:
                logger.error(f"❌ Pi audio send error: {e}")
                await asyncio.sleep(0.1)

    async def _pi_receive_responses(self):
        """Pi-optimized response handling"""
        while self.running:
            try:
                async for response in self.session.receive():
                    server_content = response.server_content

                    if (hasattr(server_content, "interrupted") and server_content.interrupted):
                        logger.info("🤫 Pi: Interruption detected")

                    if server_content and server_content.model_turn:
                        for part in server_content.model_turn.parts:
                            # Handle audio immediately
                            if part.inline_data:
                                self.audio_manager.add_audio(part.inline_data.data)
                            
                            # Handle text for restaurant processing
                            if part.text:
                                text_content = part.text
                                logger.info(f"📝 Pi Gemini: {text_content}")
                                
                                # Process restaurant searches
                                self._pi_process_restaurant_query(text_content)

                    if server_content and server_content.turn_complete:
                        logger.info("✅ Pi Gemini turn complete")
                        
            except Exception as e:
                logger.error(f"❌ Pi response handling error: {e}")
                await asyncio.sleep(1)

    def _pi_process_restaurant_query(self, text: str):
        """Process restaurant queries on Pi"""
        try:
            # Extract restaurant name
            restaurant_name = self._extract_restaurant_from_text(text)
            if restaurant_name:
                results = enhanced_restaurant_search(restaurant_name, max_results=2)
                if results:
                    logger.info(f"🍽️ Pi found {len(results)} restaurants")
                    # Store for potential map display
                    global current_route_text
                    if results[0].get('Path'):
                        current_route_text = results[0]['Path']
                        
        except Exception as e:
            logger.error(f"❌ Pi restaurant processing error: {e}")

    def _extract_restaurant_from_text(self, text: str) -> str:
        """Extract restaurant name from text"""
        try:
            patterns = extract_restaurant_patterns()
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else (match[1] if len(match) > 1 else "")
                    
                    result = clean_extracted_text(match)
                    if len(result) > 2:
                        logger.info(f"✅ Pi extracted restaurant: '{result}'")
                        return result
            
            return ""
        except Exception as e:
            logger.error(f"❌ Pi restaurant extraction error: {e}")
            return ""

    async def _pi_check_greeting(self):
        """Check for greeting requests on Pi"""
        while self.running:
            try:
                if (hasattr(self, 'greeting_requested') and 
                    self.greeting_requested and not self.greeting_sent):
                    
                    logger.info("👋 Pi sending greeting...")
                    await self.send_greeting(STRONG_GREETING)
                    self.greeting_sent = True
                    self.greeting_requested = False
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Pi greeting check error: {e}")
                await asyncio.sleep(1)

    async def send_greeting(self, greeting_text: str):
        """Send greeting to Pi Gemini Live"""
        try:
            if self.session:
                system_message = f"قل للمستخدم: {greeting_text}"
                await self.session.send_realtime_input(text=system_message)
                logger.info("👋 Pi greeting sent to Gemini Live")
        except Exception as e:
            logger.error(f"❌ Pi greeting error: {e}")

    async def stop(self):
        """Stop Pi Gemini Live session"""
        self.running = False
        
        if hasattr(self, 'session_context') and self.session_context:
            try:
                await self.session_context.__aexit__(None, None, None)
            except:
                pass
                
        if self.audio_manager:
            self.audio_manager.cleanup()
            
        logger.info("🛑 Pi Gemini Live session stopped")

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
def main():
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
        
        # Initialize Pi Gemini Live session
        gemini_session = PiGeminiLiveSession()
        
        # Start Gemini Live in background thread
        def start_pi_gemini_live():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(gemini_session.start())
            loop.run_forever()
        
        gemini_thread = threading.Thread(target=start_pi_gemini_live, daemon=True)
        gemini_thread.start()
        logger.info("🎤 Pi Gemini Live started in background")
        
        # Wait for session to initialize
        time.sleep(3)
        
        # Start face detection
        def run_pi_face_detection():
            time.sleep(2)
            pi_face_detection_startup()
        
        detection_thread = threading.Thread(target=run_pi_face_detection, daemon=True)
        detection_thread.start()
        
        # Main Pi loop
        system_active = True
        person_present = True
        set_status_led(True)
        
        logger.info("✅ Pi Siraj ready for voice interaction!")
        logger.info("📱 Press hardware button to toggle system")
        logger.info("📱 Press Ctrl+C to stop")
        
        # Simple monitoring loop
        try:
            while True:
                time.sleep(1)
                
                # Update status LED based on speaking state
                if speaking:
                    set_status_led(True)
                elif listening:
                    # Blink LED when listening
                    set_status_led(True)
                    time.sleep(0.1)
                    set_status_led(False)
                    time.sleep(0.1)
                else:
                    set_status_led(system_active)
                    
        except KeyboardInterrupt:
            logger.info("🛑 Pi shutdown requested")
            
    except Exception as e:
        logger.error(f"❌ Pi main error: {e}")
        
    finally:
        # Cleanup
        try:
            if gemini_session:
                def stop_session_sync():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(gemini_session.stop())
                    loop.close()
                
                stop_thread = threading.Thread(target=stop_session_sync)
                stop_thread.start()
                stop_thread.join(timeout=5)
                
            cleanup_gpio()
            
        except Exception as e:
            logger.error(f"❌ Pi cleanup error: {e}")
        
        logger.info("👋 Pi Siraj shutdown complete")

if __name__ == "__main__":
    main() 