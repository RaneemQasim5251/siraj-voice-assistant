Write-Host "🚀 Sending full_inegration.py to Raspberry Pi..." -ForegroundColor Green
Write-Host ""
Write-Host "Target: siraj@192.168.100.17" -ForegroundColor Yellow
Write-Host ""

# Copy the file using SCP
Write-Host "📤 Copying file..." -ForegroundColor Cyan
scp full_inegration.py siraj@192.168.100.17:~/siraj_gemini/

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ File transferred successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🔄 Verifying transfer..."
    ssh siraj@192.168.100.17 "ls -la ~/siraj_gemini/full_inegration.py"
    Write-Host ""
    Write-Host "🎯 File size comparison:" -ForegroundColor Cyan
    $localSize = (Get-Item "full_inegration.py").Length
    Write-Host "Local file size: $localSize bytes" -ForegroundColor White
    ssh siraj@192.168.100.17 "stat -c%s ~/siraj_gemini/full_inegration.py" 2>$null | ForEach-Object { Write-Host "Remote file size: $_ bytes" -ForegroundColor White }
} else {
    Write-Host "❌ Transfer failed!" -ForegroundColor Red
    Write-Host "💡 Make sure SSH is enabled on Raspberry Pi" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to continue" 