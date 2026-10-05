@echo off
REM Opens the trading-bots dashboard in your browser. Close this window to stop it.
cd /d "%~dp0"
python -m streamlit run dashboard.py --server.port 8502
pause
