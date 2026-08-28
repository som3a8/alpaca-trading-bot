# watchdog.ps1 — independent process watchdog for the trading bot
# =================================================================
# run_bot.bat already restarts main.py on a normal exit, but the whole
# process tree (cmd.exe + python.exe) was observed disappearing at once
# with no exit logged and no system sleep/crash/reboot event around it —
# something outside the script's own control killed it. This is a
# separate, independent process that just checks every few minutes
# whether python.exe is running at all, and relaunches run_bot.bat if
# not. Meant to survive even if the main supervisor chain gets killed
# again the same way.

$botDir = "C:\Users\Latitude 7320 2-in-1\Desktop\programing\trading bot"
$logPath = Join-Path $botDir "logs\watchdog.log"

while ($true) {
    $running = Get-Process python -ErrorAction SilentlyContinue
    if (-not $running) {
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path $logPath -Value "[$ts] python.exe not found — relaunching run_bot.bat"
        Start-Process -FilePath (Join-Path $botDir "run_bot.bat") -WorkingDirectory $botDir -WindowStyle Minimized
    }
    Start-Sleep -Seconds 180
}
