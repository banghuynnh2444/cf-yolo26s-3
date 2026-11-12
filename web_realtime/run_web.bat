@echo off
REM Chay web quet benh la ca phe. Them --host 0.0.0.0 --https de dien thoai trong mang LAN dung camera.
cd /d %~dp0
python app.py %*
pause
