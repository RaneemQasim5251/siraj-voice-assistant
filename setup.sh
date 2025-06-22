#!/bin/bash

# Siraj Voice Assistant Setup Script
# Auto-detects platform and installs appropriate dependencies

set -e

echo "🎙️ Setting up Siraj Voice Assistant..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect platform
detect_platform() {
    if [[ -f /proc/cpuinfo ]] && grep -q "BCM" /proc/cpuinfo; then
        echo "raspberry_pi"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Install system dependencies for Raspberry Pi
install_pi_dependencies() {
    print_status "Installing Raspberry Pi system dependencies..."
    
    sudo apt update
    sudo apt install -y \
        python3 python3-dev python3-pip python3-venv \
        libportaudio2 libportaudio-dev libasound2-dev \
        libopencv-dev python3-opencv \
        alsa-utils pulseaudio \
        git curl wget \
        build-essential cmake

    print_success "Raspberry Pi dependencies installed"
}

# Install system dependencies for Linux
install_linux_dependencies() {
    print_status "Installing Linux system dependencies..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            python3 python3-dev python3-pip python3-venv \
            libportaudio2 libportaudio-dev \
            python3-opencv \
            alsa-utils pulseaudio \
            git curl wget
    elif command -v yum &> /dev/null; then
        sudo yum install -y \
            python3 python3-devel python3-pip \
            portaudio-devel \
            opencv-python \
            git curl wget
    else
        print_warning "Unknown package manager. Please install dependencies manually."
    fi
    
    print_success "Linux dependencies installed"
}

# Install system dependencies for macOS
install_macos_dependencies() {
    print_status "Installing macOS system dependencies..."
    
    if ! command -v brew &> /dev/null; then
        print_status "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    brew install portaudio opencv python3
    print_success "macOS dependencies installed"
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Install Python dependencies
install_python_deps() {
    local platform=$1
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    print_status "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    if [[ "$platform" == "raspberry_pi" ]]; then
        print_status "Installing Raspberry Pi Python dependencies..."
        pip install -r requirements_raspberry_pi.txt
        
        # Install PySimpleGUI from private server
        print_status "Installing PySimpleGUI from private server..."
        pip install --force-reinstall --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
    else
        print_status "Installing standard Python dependencies..."
        pip install -r requirements.txt
        
        # Install PySimpleGUI from private server
        print_status "Installing PySimpleGUI from private server..."
        pip install --force-reinstall --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
    fi
    
    print_success "Python dependencies installed"
}

# Setup environment file
setup_env() {
    if [[ ! -f ".env" ]]; then
        print_status "Setting up environment file..."
        cat > .env << EOF
# Siraj Voice Assistant Environment Variables
GEMINI_API_KEY=your_gemini_api_key_here
SIRAJ_HEADLESS=false
EOF
        print_warning "Please edit .env file and add your Gemini API key"
    else
        print_warning ".env file already exists"
    fi
}

# Download required data files
download_data() {
    print_status "Setting up data directory..."
    mkdir -p data models
    
    # Create sample data files if they don't exist
    if [[ ! -f "data/faq.json" ]]; then
        cat > data/faq.json << 'EOF'
{
    "faqs": [
        {
            "question": "ما هو سراج؟",
            "answer": "سراج هو مساعد صوتي ذكي لمترو الرياض"
        }
    ]
}
EOF
    fi
    
    print_success "Data directory setup complete"
}

# Main installation function
main() {
    print_status "Starting Siraj Voice Assistant setup..."
    
    # Detect platform
    PLATFORM=$(detect_platform)
    print_status "Detected platform: $PLATFORM"
    
    # Install system dependencies
    case $PLATFORM in
        "raspberry_pi")
            install_pi_dependencies
            ;;
        "linux")
            install_linux_dependencies
            ;;
        "macos")
            install_macos_dependencies
            ;;
        *)
            print_warning "Unknown platform. Skipping system dependencies."
            ;;
    esac
    
    # Create virtual environment
    create_venv
    
    # Install Python dependencies
    install_python_deps $PLATFORM
    
    # Setup environment
    setup_env
    
    # Download/setup data
    download_data
    
    print_success "🎉 Siraj Voice Assistant setup complete!"
    echo ""
    print_status "Next steps:"
    echo "1. Edit .env file and add your Gemini API key"
    echo "2. Add your data files to the data/ directory"
    echo "3. Run: source venv/bin/activate"
    echo "4. Run: python3 full_inegration.py"
    echo ""
    print_status "For Raspberry Pi with display:"
    echo "export DISPLAY=:0 && python3 full_inegration.py"
    echo ""
    print_status "For headless mode:"
    echo "export SIRAJ_HEADLESS=true && python3 full_inegration.py"
}

# Run main function
main "$@" 