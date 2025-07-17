# 🎙️ Siraj Voice Assistant - المساعد الصوتي سراج

![WhatsApp Image 2025-06-20 at 01 41 45_ab8de39c](https://github.com/user-attachments/assets/ab676baa-c2b0-4017-9c05-ac5ba1afc1a8)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Arabic](https://img.shields.io/badge/language-Arabic%20%7C%20English-green.svg)
![Raspberry Pi](https://img.shields.io/badge/platform-Raspberry%20Pi%20%7C%20Desktop-orange.svg)

**Siraj** (سراج) is an advanced Arabic voice assistant powered by Google Gemini API, designed specifically for Saudi Arabia's Riyadh Metro system and local services. It provides real-time voice interactions, navigation assistance, restaurant recommendations, and comprehensive local information.

## 🌟 Key Features

### 🗣️ Voice & AI
- **Real-time Voice Conversations** using Google Gemini Live API
- **Advanced Arabic Language Processing** with pronunciation optimization
- **Bidirectional Audio** for natural conversations
- **Face Detection** for automatic greeting and interaction
- **Video Avatar Display** with speaking/silent animations

### 🚇 Metro & Navigation
- **Riyadh Metro Navigation** with real-time route planning
- **Restaurant Search & Recommendations** near metro stations
- **Interactive Maps** with Folium integration
- **Path Optimization** using NetworkX algorithms
- **Station Information** and facilities lookup

### 🖥️ Multi-Platform Support
- **Desktop GUI** with PySimpleGUI
- **Web Interface** with Flask-SocketIO
- **Raspberry Pi** optimized deployment
- **Headless Mode** for server environments

### 🔧 Additional Features
- **RAG System** for comprehensive knowledge base
- **GitHub Profile Integration** for developers
- **Multi-language Support** (Arabic, English, Urdu, Chinese)
- **Database Integration** for restaurants and stations

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Raspberry Pi Setup](#-raspberry-pi-setup)
- [Web Interface](#-web-interface)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Audio input/output devices
- (Optional) Raspberry Pi for deployment

### Automatic Setup

```bash
# Clone the repository
git clone https://github.com/your-username/siraj-voice-assistant.git
cd siraj-voice-assistant

# Run the setup script
chmod +x setup.sh
./setup.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For Raspberry Pi
pip install -r requirements_raspberry_pi.txt

# For web interface only
pip install -r requirements_web.txt
```

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```env
# Required
GEMINI_API_KEY=your_google_gemini_api_key_here

# Optional
SIRAJ_HEADLESS=false  # Set to true for headless mode
SIRAJ_LANGUAGE=ar     # Default language (ar, en, ur, zh)
SIRAJ_VOICE_SPEED=1.0 # Voice playback speed
SIRAJ_DEBUG=false     # Enable debug logging
```

### 2. Audio Configuration

```bash
# Test audio setup
python3 -c "from full_inegration import test_audio; test_audio()"

# List audio devices
python3 -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"
```

### 3. Database Setup

The application will automatically create necessary databases on first run:
- `restaurants.db` - Restaurant and business information
- `stations.db` - Metro station data

## 🎯 Quick Start

### Desktop Application

```bash
# Standard GUI mode
python3 full_inegration.py

# Headless mode (audio only)
export SIRAJ_HEADLESS=true
python3 full_inegration.py
```

### Web Interface

```bash
# Start web server
python3 app.py

# Access at http://localhost:5000
```

### Test API Connection

```bash
# Test Gemini API
python3 check_api.py

# Test full integration
python3 test_api.py
```

## 📱 Usage

### Basic Voice Commands (Arabic)

```
🎤 "مرحبا سراج" - Greeting and activation
🎤 "كيف أصل إلى محطة الملك عبدالله؟" - Navigation
🎤 "أين أقرب مطعم؟" - Restaurant search
🎤 "ما هي ساعات عمل المترو؟" - Metro information
🎤 "وضع صامت" - Silent mode
```

### Basic Voice Commands (English)

```
🎤 "Hello Siraj" - Greeting and activation
🎤 "How do I get to King Abdullah Station?" - Navigation
🎤 "Where is the nearest restaurant?" - Restaurant search
🎤 "What are the metro operating hours?" - Metro information
🎤 "Silent mode" - Silent mode
```

### GUI Controls

- **🎤 Voice Button**: Click to start/stop voice recording
- **🗺️ Map Button**: Open interactive metro map
- **📋 Menu Button**: Access additional features
- **⚙️ Settings**: Configure language, voice, and preferences
- **📊 Status**: View connection and system status

## 🥧 Raspberry Pi Setup

### Quick Setup

```bash
# Download and run Pi setup script
wget https://raw.githubusercontent.com/your-username/siraj-voice-assistant/main/setup_pi.sh
chmod +x setup_pi.sh
./setup_pi.sh
```

### Manual Pi Setup

```bash
# SSH into your Raspberry Pi
ssh pi@your-pi-ip

# Clone and setup
git clone https://github.com/your-username/siraj-voice-assistant.git
cd siraj-voice-assistant
pip3 install -r requirements_raspberry_pi.txt

# Configure for headless mode
export SIRAJ_HEADLESS=true
python3 full_inegration.py
```

### Pi with Display

```bash
# For Pi with connected display
export DISPLAY=:0
python3 full_inegration.py

# Auto-start on boot
echo "@/home/pi/siraj-voice-assistant/run_siraj_gui.sh" >> ~/.config/lxsession/LXDE-pi/autostart
```

See [PI_DISPLAY_SETUP.md](PI_DISPLAY_SETUP.md) for detailed display configuration.

## 🌐 Web Interface

### Starting the Web Server

```bash
# Start Flask-SocketIO server
python3 app.py

# Server starts on http://localhost:5000
```

### Web Features

- **Real-time Voice Chat** with WebRTC
- **Interactive Metro Map** with route planning
- **Restaurant Search** with location-based results
- **Mobile-responsive Design** for all devices
- **WebSocket Communication** for real-time updates

### API Endpoints

```
GET  /                 - Main interface
POST /api/chat         - Text-based chat
POST /api/navigate     - Navigation requests
POST /api/restaurants  - Restaurant search
WebSocket /socket.io   - Real-time communication
```

## 📊 Project Structure

```
siraj-voice-assistant/
├── full_inegration.py      # Main desktop application
├── app.py                  # Web server (Flask-SocketIO)
├── today.py               # GitHub profile integration
├── requirements.txt       # Python dependencies
├── setup.sh              # Automated setup script
├── .env                   # Environment configuration
│
├── data/                  # Data files
│   ├── faq.json          # Frequently asked questions
│   ├── metro_data.json   # Metro system information
│   └── rules.json        # Conversation rules
│
├── templates/            # Web interface templates
│   └── index.html       # Main web interface
│
├── models/              # AI models and weights
│   └── yolov8n.pt      # YOLO face detection model
│
├── raspberry_pi/        # Raspberry Pi specific files
│   ├── pi_siraj_optimized.py
│   ├── setup_pi.sh
│   └── requirements_raspberry_pi.txt
│
└── docs/               # Documentation
    ├── CONTRIBUTING.md
    ├── PI_DISPLAY_SETUP.md
    └── RASPBERRY_PI_INSTRUCTIONS.md
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone with development dependencies
git clone https://github.com/your-username/siraj-voice-assistant.git
cd siraj-voice-assistant

# Install development dependencies
pip install -r requirements_complete.txt

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 . --max-line-length=100

# Type checking
mypy full_inegration.py
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_audio.py -v

# Run with coverage
pytest --cov=. tests/
```

## 🔧 Troubleshooting

### Common Issues

#### Audio Problems
```bash
# Check audio devices
python3 -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"

# Fix ALSA errors on Linux
export PULSE_RUNTIME_PATH=/run/user/$(id -u)/pulse
```

#### API Connection Issues
```bash
# Test API connection
python3 check_api.py

# Check environment variables
env | grep GEMINI
```

#### Raspberry Pi Display Issues
```bash
# Set display
export DISPLAY=:0

# Check X11 service
sudo systemctl status display-manager
```

### Performance Optimization

- **For Raspberry Pi**: Use headless mode (`SIRAJ_HEADLESS=true`)
- **For slower systems**: Reduce audio chunk size in configuration
- **For better accuracy**: Use higher quality audio input
- **For faster responses**: Use local caching for frequent queries

## 📖 API Reference

### Main Classes

#### `SirajVoiceAssistant`
```python
class SirajVoiceAssistant:
    def __init__(self, headless=False)
    def start_listening(self)
    def process_voice_input(self, audio_data)
    def generate_response(self, user_input)
    def speak_response(self, text)
```

#### `MetroNavigator`
```python
class MetroNavigator:
    def find_route(self, start_station, end_station)
    def get_station_info(self, station_name)
    def search_nearby_restaurants(self, station_name)
```

#### `CVGenerator`
```python
class CVGenerator:
    def generate_cv(self, template_type, user_data)
    def export_to_word(self, cv_data, filename)
    def export_to_pdf(self, cv_data, filename)
```

### Configuration Options

```python
# Default configuration
CONFIG = {
    'LANGUAGE': 'ar',
    'VOICE_SPEED': 1.0,
    'CHUNK_SIZE': 1024,
    'SAMPLE_RATE': 16000,
    'CHANNELS': 1,
    'HEADLESS': False,
    'DEBUG': False
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Contribution Guide

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Areas for Contribution

- 🌍 **Language Support**: Add support for more languages
- 🎨 **UI/UX**: Improve interface design and user experience
- 📊 **Data**: Add more restaurant and location data
- 🔧 **Performance**: Optimize for different hardware configurations
- 📚 **Documentation**: Improve guides and tutorials
- 🧪 **Testing**: Add more comprehensive tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

Special thanks to our amazing contributors who have helped build and improve Siraj:

<div align="center">

| Contributor | GitHub Profile |
|-------------|---------------|
| **Razan** | [@Razanfah](https://github.com/Razanfah) 
| **Ghadah** | [@GTawhari](https://github.com/GTawhari) 
| **Norah** | [@Norahmw21](https://github.com/Norahmw21) 
| **Raneem** | [@RaneemQasim5251](https://github.com/RaneemQasim5251)

</div>

## 🙏 Acknowledgments

- **Google Gemini API** for powerful AI capabilities
- **Riyadh Metro** for transportation data
- **Open Source Community** for various libraries and tools
- **Our Contributors** who make Siraj possible

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/your-username/siraj-voice-assistant/issues)
- **GitHub Discussions**: [Community discussions](https://github.com/your-username/siraj-voice-assistant/discussions)
- **Documentation**: [Wiki and guides](https://github.com/your-username/siraj-voice-assistant/wiki)

## 🔗 Links

- **Live Demo**: [Try Siraj Online](https://siraj-demo.herokuapp.com)
- **Documentation**: [Full Documentation](https://siraj-voice-assistant.readthedocs.io)
- **Video Tutorial**: [Getting Started with Siraj](https://youtube.com/watch?v=example)

---

<div align="center">

**🎙️ Made with ❤️ for the Arabic-speaking community**

[⭐ Star this repo](https://github.com/your-username/siraj-voice-assistant) | [🐛 Report Bug](https://github.com/your-username/siraj-voice-assistant/issues) | [✨ Request Feature](https://github.com/your-username/siraj-voice-assistant/issues)

</div>


