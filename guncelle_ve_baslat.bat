@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo [SERIT GORU] GitHub guncellemesi kontrol ediliyor...
where git >nul 2>&1
if errorlevel 1 (
    echo Git bulunamadi. https://git-scm.com/download/win
    pause
    exit /b 1
)
git pull --ff-only
if errorlevel 1 (
    echo Guncelleme basarisiz. Yerel degisikliklerinizi kontrol edin; hicbir dosyaniz silinmedi.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 (
        python -m venv .venv
        if errorlevel 1 (echo Python kurulumu gerekli. & pause & exit /b 1)
    )
)
echo [SERIT GORU] Kutuphaneler kontrol ediliyor...
".venv\Scripts\python.exe" -m pip install -r requirements-ai.txt
if errorlevel 1 (echo Kutuphaneler kurulamadı. & pause & exit /b 1)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause