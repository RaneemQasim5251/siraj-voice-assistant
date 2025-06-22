# 🎙️ Siraj - Arabic Voice Assistant for Riyadh Metro

**Siraj** (Arabic: سراج, meaning "lamp/light") is an intelligent Arabic voice assistant specifically designed for Riyadh Metro navigation and restaurant discovery. Powered by Google Gemini Live API, it provides real-time voice interaction with advanced AI capabilities.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Raspberry%20Pi-lightgrey)](https://github.com/RaneemQasim5251/siraj-voice-assistant)

## ✨ Key Features

### 🎤 **Advanced Voice Interaction**
- **Gemini Live API Integration**: Real-time streaming voice conversations
- **Face Detection**: Automatic greeting when users are detected
- **Multi-language Support**: Arabic, English, Urdu, Chinese, and more
- **Natural Language Processing**: Understanding complex queries in Arabic

### 🗺️ **Riyadh Metro Navigation**
- **Complete Route Planning**: Navigate between any metro stations
- **Interactive Maps**: Visual route display using Folium
- **Station Information**: Comprehensive data for all metro lines
- **Real-time Directions**: Step-by-step navigation guidance

### 🍽️ **Restaurant Discovery**
- **Extensive Database**: 2,600+ restaurants in Riyadh
- **Smart Search**: Find restaurants by name, cuisine type, or location
- **Rating System**: View restaurant ratings and reviews
- **Metro Integration**: Get directions from metro stations to restaurants

### 📱 **Multi-Platform Support**
- **Desktop GUI**: Full-featured interface with video display
- **Web Interface**: Browser-based access via Flask
- **Raspberry Pi**: Optimized for embedded systems
- **Headless Mode**: Command-line operation without GUI

### 🎬 **Interactive Video Display**
- **Dynamic Video Switching**: Different videos for speaking/listening states
- **Face Recognition**: OpenCV-based user detection
- **Real-time Animation**: Smooth transitions between states
- **Touch Screen Support**: Optimized for interactive displays

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get yours here](https://aistudio.google.com/app/apikey))
- Microphone and speakers/headphones
- Webcam (optional, for face detection)

### 🔧 Installation

#### Option 1: Automatic Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/RaneemQasim5251/siraj-voice-assistant.git
cd siraj-voice-assistant

# Run setup script (detects your platform automatically)
chmod +x setup.sh
./setup.sh
```

#### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PySimpleGUI from private server
pip install --force-reinstall --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
```

### 🔑 Configuration
1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Gemini API key:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### ▶️ Running Siraj

#### Desktop Mode (with GUI)
```bash
python3 full_inegration.py
```

#### Headless Mode (audio only)
```bash
export SIRAJ_HEADLESS=true
python3 full_inegration.py
```

#### Web Interface
```bash
python3 app.py
# Open http://localhost:5000 in your browser
```

## 🍓 Raspberry Pi Setup

### System Requirements
- Raspberry Pi 3B+ or newer
- Raspbian OS (Bullseye or newer)
- USB microphone or compatible audio HAT
- HDMI display (optional, for GUI mode)

### Installation
```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-dev libportaudio2 libportaudio-dev python3-opencv alsa-utils

# Clone and setup
git clone https://github.com/RaneemQasim5251/siraj-voice-assistant.git
cd siraj-voice-assistant
pip install -r requirements_raspberry_pi.txt
```

### Running on Pi
```bash
# With display connected
export DISPLAY=:0
python3 full_inegration.py

# Headless mode (recommended for SSH)
export SIRAJ_HEADLESS=true
python3 full_inegration.py
```

## 📖 Usage Examples

### Metro Navigation
- **Arabic**: "كيف أروح من محطة الملز إلى المطار؟"
- **English**: "How do I get from Malaz Station to the Airport?"

### Restaurant Search
- **Arabic**: "أبغى مطعم هندي قريب من العليا"
- **English**: "I want an Indian restaurant near Olaya"

### General Queries
- **Arabic**: "وش أفضل المطاعم في منطقة الملك فهد؟"
- **English**: "What are the best restaurants in King Fahd area?"

## 🏗️ Architecture

### Core Components
```
siraj-voice-assistant/
├── full_inegration.py      # Main application
├── arabic_enhancement.py   # Arabic text processing
├── app.py                 # Web interface
├── data/                  # Metro and restaurant data
│   ├── metro_data.json
│   ├── faq.json
│   └── rules.json
├── models/                # AI models
│   └── yolov8n.pt
├── templates/             # Web templates
└── requirements*.txt      # Dependencies
```

### Technology Stack
- **AI**: Google Gemini Live API
- **Audio**: PyAudio, pygame, soundfile
- **Computer Vision**: OpenCV, YOLOv8
- **GUI**: PySimpleGUI
- **Web**: Flask
- **Data**: pandas, geopandas
- **Maps**: Folium, matplotlib

## 🛠️ Development

### Setting up Development Environment
```bash
# Clone with development tools
git clone https://github.com/RaneemQasim5251/siraj-voice-assistant.git
cd siraj-voice-assistant

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest pytest-cov

# Run code formatting
black .
flake8 . --max-line-length=100

# Run tests
pytest tests/ -v
```

### Adding New Features
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📊 Data Sources

### Metro Data
- **Stations**: All 83 stations across 6 metro lines
- **Routes**: Complete network topology with connections
- **GeoJSON**: Geographic data for mapping

### Restaurant Data
- **Coverage**: 2,600+ restaurants in Riyadh
- **Information**: Name, cuisine type, location, ratings
- **Integration**: Connected to metro station proximity

## 🔧 Configuration Options

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your_api_key_here

# Optional
SIRAJ_HEADLESS=false          # Enable headless mode
DEBUG=false                   # Enable debug logging
AUDIO_SAMPLE_RATE=16000      # Audio sample rate
AUDIO_CHUNK_SIZE=1024        # Audio buffer size
```

### Audio Settings
- **Sample Rate**: 16kHz (optimized for Gemini)
- **Channels**: Mono
- **Format**: 16-bit PCM
- **Buffer Size**: 1024 samples (Pi) / 2048 samples (Desktop)

## 🐛 Troubleshooting

### Common Issues

#### Audio Problems
```bash
# Check audio devices
python3 -c "import pyaudio; print([pyaudio.PyAudio().get_device_info_by_index(i)['name'] for i in range(pyaudio.PyAudio().get_device_count())])"

# Test microphone
python3 -c "from full_inegration import test_audio; test_audio()"
```

#### GUI Not Showing (Linux/Pi)
```bash
# Set display variable
export DISPLAY=:0

# Or use headless mode
export SIRAJ_HEADLESS=true
```

#### API Key Issues
- Verify your Gemini API key is valid
- Check quota limits at [Google AI Studio](https://aistudio.google.com)
- Ensure `.env` file is in the correct directory

### Getting Help
- 📝 [Open an issue](https://github.com/RaneemQasim5251/siraj-voice-assistant/issues)
- 💬 [Start a discussion](https://github.com/RaneemQasim5251/siraj-voice-assistant/discussions)
- 📧 Contact the maintainers

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Ways to Contribute
- 🐛 Report bugs
- ✨ Suggest new features
- 📚 Improve documentation
- 🌍 Add language support
- 🔧 Optimize performance
- 🧪 Write tests

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini Team** - For the powerful Live API
- **OpenCV Community** - Computer vision capabilities
- **Riyadh Metro Authority** - Public transportation data
- **Open Source Community** - Various libraries and tools

## 📈 Roadmap

### Current Version (v1.0)
- ✅ Basic voice interaction
- ✅ Metro navigation
- ✅ Restaurant search
- ✅ Multi-platform support

### Planned Features (v2.0)
- 🔄 Real-time traffic updates
- 📱 Mobile app companion
- 🤖 Advanced AI conversations
- 🌐 More language support
- 📍 GPS integration

## 📞 Support

### Getting Started
1. Check the [Quick Start](#-quick-start) guide
2. Review [common issues](#-troubleshooting)
3. Search existing [issues](https://github.com/RaneemQasim5251/siraj-voice-assistant/issues)

### Reporting Issues
When reporting issues, please include:
- Operating system and version
- Python version
- Error messages (full traceback)
- Steps to reproduce
- Expected vs actual behavior

---

<div align="center">

**Made with ❤️ for the people of Riyadh**

[🌟 Star this project](https://github.com/RaneemQasim5251/siraj-voice-assistant) • [🐛 Report Bug](https://github.com/RaneemQasim5251/siraj-voice-assistant/issues) • [💡 Request Feature](https://github.com/RaneemQasim5251/siraj-voice-assistant/issues)

</div> 