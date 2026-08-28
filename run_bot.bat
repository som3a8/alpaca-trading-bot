@echo off
REM run_bot.bat — outer supervisor for main.py
REM =============================================
REM main.py already catches and logs exceptions inside its own loop (see
REM CRASH_COOLDOWN_SECS in main.py) so a normal error never kills the
REM process. This wrapper is the second layer: it catches the cases the
REM in-script handler CAN'T — a hard interpreter crash, an OOM kill, a
REM forced process termination — by just restarting main.py whenever it
REM exits, for any reason, forever. Meant to run unattended for the whole
REM competition week.

setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

if not exist logs mkdir logs

:loop
echo [%date% %time%] Starting main.py >> logs\supervisor.log
python main.py
echo [%date% %time%] main.py exited with code %ERRORLEVEL% — restarting in 30s >> logs\supervisor.log
timeout /t 30 /nobreak >nul
goto loop
