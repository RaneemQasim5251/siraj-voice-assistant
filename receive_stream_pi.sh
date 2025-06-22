#!/bin/bash

echo "📺 Starting stream receiver on Raspberry Pi..."
echo "Receiving from Windows PC..."
echo

# Install required packages if not installed
if ! command -v vlc &> /dev/null; then
    echo "Installing VLC..."
    sudo apt update
    sudo apt install vlc -y
fi

if ! command -v cvlc &> /dev/null; then
    echo "Installing VLC command line..."
    sudo apt install vlc-bin -y
fi

# Set up for 7-inch display
export DISPLAY=:0

echo "🎬 Starting fullscreen playback for 7-inch display (1024x600)"
echo "Waiting for stream from Windows..."
echo

# Start RTSP server to receive stream
cvlc rtsp://0.0.0.0:8554/live \
     --fullscreen \
     --no-video-title-show \
     --no-osd \
     --intf dummy \
     --quiet

echo "Stream ended." 