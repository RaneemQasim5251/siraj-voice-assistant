#!/bin/bash
# Siraj Setup Script for Raspberry Pi
# Usage: chmod +x setup_pi.sh && ./setup_pi.sh

echo "🍓 Setting up Siraj on Raspberry Pi..."
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi\|BCM" /proc/cpuinfo 2>/dev/null; then
    print_warning "This script is designed for Raspberry Pi"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
print_step "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_step "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libportaudio2 \
    libportaudio-dev \
    libasound2-dev \
    libsndfile1 \
    libsndfile1-dev \
    libopencv-dev \
    python3-opencv \
    pulseaudio \
    pulseaudio-utils

# Enable interfaces if needed
print_step "Enabling Raspberry Pi interfaces..."
sudo raspi-config nonint do_camera 0  # Enable camera
sudo raspi-config nonint do_i2c 0     # Enable I2C
sudo raspi-config nonint do_spi 0     # Enable SPI

# Create project directory
PROJECT_DIR="$HOME/siraj_gemini"
if [ ! -d "$PROJECT_DIR" ]; then
    print_step "Creating project directory..."
    mkdir -p "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# Create virtual environment
print_step "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip

# Install Python packages
print_step "Installing Python packages..."
if [ -f "requirements_raspberry_pi.txt" ]; then
    pip install -r requirements_raspberry_pi.txt
else
    print_warning "requirements_raspberry_pi.txt not found, installing core packages..."
    pip install google-genai python-dotenv pandas opencv-python
    pip install pyaudio pygame soundfile RPi.GPIO
    pip install fuzzywuzzy python-Levenshtein loguru
    pip install arabic-reshaper python-bidi
    pip install FreeSimpleGUI
fi

# Setup environment file
print_step "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Gemini API Key (replace with your actual key)
GEMINI_API_KEY=your_gemini_api_key_here

# Force headless mode for Raspberry Pi
SIRAJ_HEADLESS=true

# Audio settings
PI_AUDIO_DEVICE=default
PI_MIC_DEVICE=default

# GPIO Pin configurations
BUTTON_PIN=18
LED_PIN=24
BUZZER_PIN=23
EOF
    print_warning "Please edit .env file and add your GEMINI_API_KEY"
fi

# Create run script
print_step "Creating run script..."
cat > run_siraj.sh << 'EOF'
#!/bin/bash
cd ~/siraj_gemini
source venv/bin/activate
python3 full_inegration.py "$@"
EOF
chmod +x run_siraj.sh

# Create systemd service
print_step "Creating systemd service..."
cat > siraj.service << 'EOF'
[Unit]
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
EOF

print_warning "To install systemd service, run:"
echo "sudo cp siraj.service /etc/systemd/system/"
echo "sudo systemctl daemon-reload"
echo "sudo systemctl enable siraj.service"

# Audio configuration
print_step "Configuring audio..."
if [ ! -f "/etc/asound.conf" ]; then
    sudo tee /etc/asound.conf > /dev/null << 'EOF'
pcm.!default {
    type asym
    playback.pcm "plughw:1,0"
    capture.pcm "plughw:1,0"
}
EOF
fi

# Create audio test script
cat > test_audio.sh << 'EOF'
#!/bin/bash
echo "Testing audio devices..."
echo "Available playback devices:"
aplay -l
echo ""
echo "Available recording devices:"
arecord -l
echo ""
echo "Testing microphone (5 seconds)..."
arecord -D plughw:1,0 -f cd -t wav -d 5 test_mic.wav
echo "Playing back recording..."
aplay test_mic.wav
rm -f test_mic.wav
EOF
chmod +x test_audio.sh

# Create GPIO test script
cat > test_gpio.sh << 'EOF'
#!/bin/bash
echo "Testing GPIO..."
echo "Testing LED on GPIO 24..."
echo "24" > /sys/class/gpio/export 2>/dev/null || true
echo "out" > /sys/class/gpio/gpio24/direction 2>/dev/null || true
echo "1" > /sys/class/gpio/gpio24/value
sleep 1
echo "0" > /sys/class/gpio/gpio24/value
echo "LED test complete"
EOF
chmod +x test_gpio.sh

# Setup completion
print_status "Setup completed successfully! 🎉"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your GEMINI_API_KEY"
echo "2. Test audio: ./test_audio.sh"
echo "3. Test GPIO: ./test_gpio.sh"
echo "4. Run Siraj: ./run_siraj.sh"
echo ""
echo "🔧 Optional - Install as system service:"
echo "sudo cp siraj.service /etc/systemd/system/"
echo "sudo systemctl daemon-reload"
echo "sudo systemctl enable siraj.service"
echo "sudo systemctl start siraj.service"
echo ""
echo "📊 Monitor logs:"
echo "sudo journalctl -u siraj.service -f"
echo ""
print_status "Siraj is ready! 🚀" 