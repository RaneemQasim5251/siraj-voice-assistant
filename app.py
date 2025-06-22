#!/usr/bin/env python3
"""
Siraj Web Server
Flask-SocketIO web server for Siraj Metro Assistant
"""

import os
import base64
import cv2
import time
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np
from loguru import logger

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'siraj_metro_assistant_2024'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

# Global variables
video_frames = {
    'silent': [],
    'speaking': []
}
current_frame_index = 0
is_speaking = False
default_frame = None

def load_video_frames():
    """Load video frames for Siraj avatar"""
    global video_frames, default_frame
    
    logger.info("🎬 Loading video frames for Siraj avatar...")
    
    try:
        # Load silent video
        if os.path.exists("siraj_silent.mp4"):
            cap_silent = cv2.VideoCapture("siraj_silent.mp4")
            if cap_silent.isOpened():
                while True:
                    ret, frame = cap_silent.read()
                    if not ret:
                        break
                    # Resize frame to fit large web display (75vh)
                    frame_resized = cv2.resize(frame, (1280, 720))
                    video_frames['silent'].append(frame_resized)
                cap_silent.release()
                logger.info(f"✅ Loaded {len(video_frames['silent'])} silent frames")
        
        # Load speaking video
        if os.path.exists("siraj_speak2.mp4"):
            cap_speaking = cv2.VideoCapture("siraj_speak2.mp4")
            if cap_speaking.isOpened():
                while True:
                    ret, frame = cap_speaking.read()
                    if not ret:
                        break
                    # Resize frame to fit large web display (75vh)
                    frame_resized = cv2.resize(frame, (1280, 720))
                    video_frames['speaking'].append(frame_resized)
                cap_speaking.release()
                logger.info(f"✅ Loaded {len(video_frames['speaking'])} speaking frames")
        
        # Create default frame if videos not available
        if not video_frames['silent'] and not video_frames['speaking']:
            default_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(default_frame, "Siraj Metro Assistant", (350, 360), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
            logger.info("✅ Created default frame")
        
    except Exception as e:
        logger.error(f"❌ Error loading video frames: {e}")
        # Create default frame as fallback
        default_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(default_frame, "Siraj Metro Assistant", (350, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

def get_current_frame():
    """Get current video frame based on state"""
    global current_frame_index, is_speaking
    
    try:
        if is_speaking and video_frames['speaking']:
            frame = video_frames['speaking'][current_frame_index % len(video_frames['speaking'])]
        elif video_frames['silent']:
            frame = video_frames['silent'][current_frame_index % len(video_frames['silent'])]
        else:
            frame = default_frame
            
        current_frame_index += 1
        return frame
        
    except Exception as e:
        logger.error(f"❌ Error getting frame: {e}")
        return default_frame

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    try:
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64
    except Exception as e:
        logger.error(f"❌ Error encoding frame: {e}")
    return None

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/simple')
def simple():
    """Simple page using root index.html"""
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"🔌 Client connected: {request.sid}")
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"🔌 Client disconnected: {request.sid}")

@socketio.on('start_stream')
def handle_start_stream():
    """Start video stream"""
    try:
        frame = get_current_frame()
        if frame is not None:
            frame_b64 = frame_to_base64(frame)
            if frame_b64:
                emit('video_frame', {'frame': frame_b64})
    except Exception as e:
        logger.error(f"❌ Error in video stream: {e}")

@socketio.on('start_video')
def handle_start_video():
    """Legacy video start handler"""
    handle_start_stream()

@socketio.on('message')
def handle_message(data):
    """Handle text messages from client with AI response"""
    global is_speaking
    
    logger.info(f"💬 Received message: {data}")
    
    is_speaking = True
    
    # Get AI response
    ai_response = get_ai_response(data)
    
    # Send response back to client
    socketio.emit('server_response', {'data': ai_response})
    socketio.emit('voice_response', {'text': ai_response})
    
    # Reset speaking state after a delay
    def reset_speaking():
        global is_speaking
        time.sleep(3)
        is_speaking = False
    
    threading.Thread(target=reset_speaking, daemon=True).start()

def get_ai_response(user_message):
    """Get AI response using Gemini or fallback responses"""
    try:
        # Try to use Gemini AI
        import google.generativeai as genai
        
        # Load API key
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            
            # Create model with Arabic system prompt
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction="""أنت سِراج، المساعد الذكي لمترو الرياض. 
                أجب باللغة العربية بطريقة ودودة ومفيدة. 
                ساعد المستخدمين في أسئلة المترو والمطاعم والأماكن في الرياض.
                كن مختصراً ومفيداً في إجاباتك."""
            )
            
            # Generate response
            response = model.generate_content(user_message)
            return response.text.strip()
            
    except Exception as e:
        logger.warning(f"⚠️ Gemini AI error: {e}")
    
    # Fallback responses in Arabic
    fallback_responses = {
        "سلام": "وعليكم السلام ورحمة الله وبركاته! أهلاً وسهلاً بك، أنا سِراج مساعدك في مترو الرياض. كيف يمكنني مساعدتك اليوم؟",
        "مرحبا": "مرحباً بك! أنا سِراج، كيف يمكنني مساعدتك في رحلتك بمترو الرياض؟",
        "مساعدة": "بكل سرور! يمكنني مساعدتك في:\n• إيجاد أفضل المسارات في المترو\n• البحث عن المطاعم القريبة من المحطات\n• معلومات عن محطات المترو\n• أوقات التشغيل والأسعار",
        "شكرا": "العفو! سعيد لمساعدتك. إذا كان لديك أي سؤال آخر، لا تتردد في السؤال.",
        "مطعم": "يمكنني مساعدتك في العثور على أفضل المطاعم القريبة من محطات المترو. أخبرني عن نوع الطعام الذي تفضله أو المنطقة التي تريد الذهاب إليها.",
        "محطة": "أخبرني عن اسم المحطة التي تريد معلومات عنها، وسأقدم لك كل التفاصيل المتاحة."
    }
    
    # Check for keyword matches
    user_lower = user_message.lower().strip()
    for keyword, response in fallback_responses.items():
        if keyword in user_lower:
            return response
    
    # Default intelligent response
    if any(word in user_lower for word in ["مترو", "قطار", "محطة", "رحلة"]):
        return f"فهمت أنك تسأل عن المترو. يمكنني مساعدتك في التنقل وإيجاد أفضل المسارات. هل يمكنك توضيح سؤالك أكثر؟"
    elif any(word in user_lower for word in ["مطعم", "طعام", "أكل", "مأكولات"]):
        return f"أستطيع مساعدتك في العثور على أفضل المطاعم القريبة من محطات المترو. أي نوع طعام تفضل؟"
    elif any(word in user_lower for word in ["أين", "كيف", "متى", "ماذا"]):
        return f"أنا هنا لمساعدتك! يمكنني الإجابة على أسئلتك حول مترو الرياض والمطاعم والأماكن. اسأل ما تريد معرفته."
    else:
        return f"شكراً لك على رسالتك! أنا سِراج، مساعدك الذكي لمترو الرياض. كيف يمكنني مساعدتك اليوم؟"

@socketio.on('voice_input')
def handle_voice_input(data):
    """Handle voice input from client with AI response"""
    global is_speaking
    
    logger.info("🎤 Received voice input")
    
    is_speaking = True
    
    # For now, simulate voice transcription - in real implementation would convert audio to text
    # Then process with AI like text messages
    simulated_text = "مرحبا سِراج"  # This would be the transcribed text
    ai_response = get_ai_response(simulated_text)
    
    # Send response
    socketio.emit('voice_response', {'text': ai_response})
    socketio.emit('server_response', {'data': ai_response})
    
    # Reset speaking state
    def reset_speaking():
        global is_speaking
        time.sleep(3)
        is_speaking = False
    
    threading.Thread(target=reset_speaking, daemon=True).start()

def start_video_stream():
    """Background task to continuously send video frames"""
    while True:
        try:
            # Send frames to all connected clients at ~30 FPS
            frame = get_current_frame()
            if frame is not None:
                frame_b64 = frame_to_base64(frame)
                if frame_b64:
                    socketio.emit('video_frame', {'frame': frame_b64}, broadcast=True)
            
            time.sleep(1/30)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"❌ Video stream error: {e}")
            time.sleep(1)

if __name__ == '__main__':
    logger.info("🚀 Starting Siraj Web Server...")
    
    # Load video frames
    load_video_frames()
    
    # Start background video streaming
    video_thread = threading.Thread(target=start_video_stream, daemon=True)
    video_thread.start()
    
    logger.info("✅ Siraj Web Server ready!")
    logger.info("🌐 Access the application at: http://localhost:5000")
    logger.info("🌐 Simple interface at: http://localhost:5000/simple")
    
    # Start the Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False) 