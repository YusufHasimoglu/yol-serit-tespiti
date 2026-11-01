@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Python bulunamadi. Python 3.10 veya daha yeni surumunu kurun.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Kutuphane kurulumu basarisiz.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
