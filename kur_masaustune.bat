@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title Serit Analizi - Masaustu Kurulumu

echo.
echo ==============================================
echo    SERIT ANALIZI - MASAUSTU KURULUMU
echo ==============================================
echo.

where git >nul 2>nul
if errorlevel 1 (
  echo [HATA] Git bulunamadi.
  echo Git'i buradan kurun: https://git-scm.com/download/win
  echo Kurulumdan sonra bu dosyayi yeniden calistirin.
  pause
  exit /b 1
)

where py >nul 2>nul
if errorlevel 1 (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [HATA] Python bulunamadi.
    echo Python 3.11 veya 3.12 kurun: https://www.python.org/downloads/windows/
    echo Kurarken 'Add Python to PATH' secenegini isaretleyin.
    pause
    exit /b 1
  )
)

set "DESKTOP="
for /f "delims=" %%D in ('powershell.exe -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"') do set "DESKTOP=%%D"
if not defined DESKTOP (
  echo [HATA] Windows masaustu konumu okunamadi.
  pause
  exit /b 1
)

set "PROJECT=%DESKTOP%\Serit-Analizi"
set "OLDPROJECT=%DESKTOP%\Serit-Goru"

rem Previously installed projects are reused, not overwritten.
if not exist "%PROJECT%\.git" if exist "%OLDPROJECT%\.git" (
  git -C "%OLDPROJECT%" remote get-url origin 2>nul | findstr /I /C:"YusufHasimoglu/yol-serit-tespiti" >nul
  if not errorlevel 1 (
    set "PROJECT=%OLDPROJECT%"
    echo [BILGI] Onceki proje klasoru bulundu ve korunarak kullanilacak.
  )
)

if exist "%PROJECT%\.git" (
  git -C "%PROJECT%" remote get-url origin 2>nul | findstr /I /C:"YusufHasimoglu/yol-serit-tespiti" >nul
  if errorlevel 1 (
    echo [HATA] Klasor baska bir GitHub deposuna bagli. Dokunulmadi:
    echo "%PROJECT%"
    pause
    exit /b 1
  )
  echo [1/3] GitHub guncellemeleri aliniyor...
  git -C "%PROJECT%" pull --ff-only
  if errorlevel 1 (
    echo [HATA] Git guncellemesi tamamlanamadi. Yerel degisiklikler silinmedi.
    echo Proje klasoru: "%PROJECT%"
    pause
    exit /b 1
  )
) else (
  if exist "%PROJECT%" (
    echo [HATA] Bu isimde bir klasor var ancak Git projesi degil:
    echo "%PROJECT%"
    echo Dosyalara dokunulmaz. Klasor adini degistirip yeniden deneyin.
    pause
    exit /b 1
  )
  echo [1/3] GitHub'dan proje masaustune indiriliyor...
  git clone --branch main https://github.com/YusufHasimoglu/yol-serit-tespiti.git "%PROJECT%"
  if errorlevel 1 (
    echo [HATA] Proje indirilemedi. Internet ve Git ayarlarinizi kontrol edin.
    pause
    exit /b 1
  )
)

if not exist "%PROJECT%\guncelle_ve_baslat.bat" (
  echo [HATA] Proje baslatma dosyasi bulunamadi.
  pause
  exit /b 1
)

set "SA_PROJECT=%PROJECT%"
set "SA_DESKTOP=%DESKTOP%"
echo [2/3] Masaustu kisayolu olusturuluyor...
powershell.exe -NoProfile -Command "$lnk=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $env:SA_DESKTOP (([char]0x015E)+'erit Analizi.lnk'))); $lnk.TargetPath=(Join-Path $env:SA_PROJECT 'guncelle_ve_baslat.bat'); $lnk.WorkingDirectory=$env:SA_PROJECT; $lnk.Description='Serit Analizi - GitHub guncelle ve baslat'; $lnk.Save()"
if errorlevel 1 (
  echo [UYARI] Kisayol olusturulamadi. Projeyi klasorden baslatabilirsiniz.
) else (
  echo [OK] Masaustune "Serit Analizi" kisayolu eklendi.
)

echo [3/3] Ilk baslatma ve Python kutuphaneleri kurulumu...
echo [BILGI] Ilk acilis internet hizina gore zaman alabilir.
call "%PROJECT%\guncelle_ve_baslat.bat"
exit /b %errorlevel%
