@echo off
echo 🖥️ Starting screen streaming to Raspberry Pi...
echo.

REM Set Raspberry Pi IP
set PI_IP=192.168.100.17

echo 📡 Streaming Windows screen to Pi at %PI_IP%
echo Resolution: 1024x600 for 7-inch display
echo.

REM Using FFmpeg to stream (install first)
REM Download: https://ffmpeg.org/download.html

ffmpeg -f gdigrab -framerate 30 -video_size 1024x600 -i desktop ^
       -f dshow -i audio="Microphone" ^
       -c:v libx264 -preset ultrafast -tune zerolatency ^
       -c:a aac -ar 44100 -b:a 128k ^
       -f rtsp rtsp://%PI_IP%:8554/live

pause 