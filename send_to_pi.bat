@echo off
echo 🚀 Sending full_inegration.py to Raspberry Pi...
echo.
echo Target: siraj@192.168.100.17
echo.

REM Copy the file using SCP
scp full_inegration.py siraj@192.168.100.17:~/siraj_gemini/

if %ERRORLEVEL% EQU 0 (
    echo ✅ File transferred successfully!
    echo.
    echo 🔄 Now connecting via SSH to verify...
    ssh siraj@192.168.100.17 "ls -la ~/siraj_gemini/full_inegration.py"
) else (
    echo ❌ Transfer failed!
    echo 💡 Make sure SSH is enabled on Raspberry Pi
)

echo.
pause 