# Siraj Voice Assistant - Raspberry Pi 5 Complete Guide

## Table of Contents
- [Initial Setup](#initial-setup)
- [System Dependencies](#system-dependencies)
- [Project Setup](#project-setup)
- [Audio Configuration](#audio-configuration)
- [Camera Setup](#camera-setup)
- [Running the Application](#running-the-application)
- [Troubleshooting](#troubleshooting)

## Initial Setup

### Update System
```bash
# Update package list and upgrade system
sudo apt update
sudo apt upgrade -y
```

### Basic Tools
```bash
# Install git and other essential tools
sudo apt install -y git wget curl
```

## System Dependencies

### Core Dependencies
```bash
# Install system packages
sudo apt install -y \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    python3-opencv
```

### Libraries
```bash
# Install required libraries
sudo apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libatlas-base-dev \
    ffmpeg
```

### Audio Packages
```bash
# Install audio dependencies
sudo apt install -y \
    pulseaudio \
    alsa-utils \
    python3-pyaudio
```

## Project Setup

### Create Project Directory
```bash
# Create and enter project directory
mkdir ~/siraj_assistant
cd ~/siraj_assistant
```

### Python Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Install Python Packages
```bash
# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements_complete.txt

# Install Raspberry Pi specific packages
pip install "numpy<1.24"
pip install "opencv-python-headless"
```

### Environment Configuration
```bash
# Create and edit .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Audio Configuration

### Test Audio Setup
```bash
# List audio devices
arecord -l
aplay -l

# Test recording
arecord -d 5 test.wav  # Record 5 seconds
aplay test.wav         # Play recording
```

### Fix Audio Issues
```bash
# Start pulseaudio
pulseaudio --start
pulseaudio --check

# Add user to audio group
sudo usermod -a -G audio $USER
```

## Camera Setup

### USB Camera Setup
```bash
# Check camera devices
v4l2-ctl --list-devices

# Test camera
v4l2-ctl -d /dev/video0 --all
```

### Camera Permissions
```bash
# Add user to video group
sudo usermod -a -G video $USER
```

## Running the Application

### Start Application
```bash
# Activate virtual environment (if not already activated)
source ~/siraj_assistant/venv/bin/activate

# Run the application
python3 full_inegration.py
```

## Troubleshooting

### System Checks
```bash
# Check CPU temperature
vcgencmd measure_temp

# Check system resources
htop
```

### Audio Issues
```bash
# Restart audio
pulseaudio --kill
pulseaudio --start

# Check audio devices
pacmd list-sinks
pacmd list-sources
```

### Camera Issues
```bash
# Check camera modules
ls /dev/video*

# Test camera access
v4l2-ctl --all
```

### Permission Issues
```bash
# Fix common permission issues
sudo usermod -a -G audio,video,input $USER
```

## Performance Tips

### System Optimization
```bash
# Add to /boot/config.txt
sudo nano /boot/config.txt

# Add these lines:
gpu_mem=128
dtoverlay=vc4-fkms-v3d
```

### Monitor Resources
```bash
# Watch CPU temperature
watch -n 1 vcgencmd measure_temp

# Monitor system resources
htop
```

## Backup and Recovery

### Create Backup
```bash
# Backup entire project
cd ~/
tar -czf siraj_backup.tar.gz siraj_assistant/
```

### Restore from Backup
```bash
# Restore project
cd ~/
tar -xzf siraj_backup.tar.gz
```

## Additional Notes

- Keep the Raspberry Pi well-ventilated
- Monitor CPU temperature regularly
- Use a good quality power supply (5V 5A recommended)
- Regular backups are recommended
- Test microphone and camera before running the application

## Touch Screen Tips

### Copy Commands
- Long press on text to select
- Drag blue handles to adjust selection
- Tap selected text and choose "Copy"
- Long press in terminal and choose "Paste"

### Navigation
- Single tap: Place cursor
- Double tap: Select word
- Triple tap: Select line
- Two finger scroll: Scroll document
- Pinch: Zoom in/out

### Quick Access
```bash
# Create desktop shortcuts
cd ~/Desktop
ln -s ~/siraj_assistant/*.txt .
```

Remember to replace `your_api_key_here` with your actual Gemini API key in the `.env` file.

For any issues or questions, refer to the troubleshooting section or check the system logs:
```bash
# Check system logs
journalctl -xe
``` 