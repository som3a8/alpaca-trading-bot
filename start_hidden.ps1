# start_hidden.ps1 — launches run_bot.bat and watchdog.ps1 with no visible
# window at all, so there's nothing on the taskbar to accidentally close.
$dir = "C:\Users\Latitude 7320 2-in-1\Desktop\programing\trading bot"
Start-Process -FilePath (Join-Path $dir "run_bot.bat") -WorkingDirectory $dir -WindowStyle Hidden
Start-Process -FilePath "powershell.exe" -ArgumentList "-WindowStyle Hidden -ExecutionPolicy Bypass -File `"$dir\watchdog.ps1`"" -WindowStyle Hidden
