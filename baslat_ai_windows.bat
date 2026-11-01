@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Python 3.10+ bulunamadi. Python yukleyin.
    pause
    exit /b 1
  )
)
echo Gerekli paketler ve PyTorch yukleniyor (internet gerekli)...
".venv\Scripts\python.exe" -m pip install -r requirements-ai.txt
if errorlevel 1 (
  echo Paket kurulumu basarisiz. README dosyasini inceleyin.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause