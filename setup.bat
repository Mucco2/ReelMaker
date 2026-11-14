@echo off
setlocal
cd /d "%~dp0"
title ReelMaker setup
echo.
echo  ===== ReelMaker setup =====
echo  Installerer alt en gang. Det kan tage 5-10 minutter.
echo  (Installs everything once. Can take 5-10 minutes.)
echo.

where winget >nul 2>nul
if errorlevel 1 (
  echo [!] winget mangler. Installer "App Installer" fra Microsoft Store og koer igen.
  pause & exit /b 1
)

echo [1/7] FFmpeg...
where ffmpeg >nul 2>nul || winget install --id Gyan.FFmpeg -e --silent --accept-source-agreements --accept-package-agreements

echo [2/7] yt-dlp...
where yt-dlp >nul 2>nul || winget install --id yt-dlp.yt-dlp -e --silent --accept-source-agreements --accept-package-agreements

echo [3/7] Node.js...
where node >nul 2>nul || winget install --id OpenJS.NodeJS.LTS -e --silent --accept-source-agreements --accept-package-agreements

echo [4/7] Python...
call :findpython
if not defined PY (
  winget install --id Python.Python.3.13 -e --silent --accept-source-agreements --accept-package-agreements
  call :refreshpath
  call :findpython
)
if not defined PY (
  echo [!] Python blev ikke fundet. Luk vinduet, aabn setup.bat igen.
  pause & exit /b 1
)
call :refreshpath

echo [5/7] Python-pakker (Whisper, OpenCV, yt-dlp, Claude)...
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q
if errorlevel 1 ( echo [!] pip install fejlede & pause & exit /b 1 )

echo [6/7] HyperFrames (animationer)...
pushd engine
call npm install --no-fund --no-audit
if errorlevel 1 ( popd & echo [!] npm install fejlede - luk vinduet og koer setup.bat igen. & pause & exit /b 1 )
call node_modules\.bin\hyperframes.cmd browser ensure
popd

echo [7/7] Ansigtsgenkendelse...
if not exist "assets\models" mkdir "assets\models"
if not exist "assets\models\face_detection_yunet_2023mar.onnx" (
  curl -sSL -o "assets\models\face_detection_yunet_2023mar.onnx" "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

echo.
".venv\Scripts\python.exe" -m reelmaker doctor
echo.
echo  Faerdig! Dobbeltklik paa ReelMaker.bat for at starte.
echo  (Done! Double-click ReelMaker.bat to start.)
pause
exit /b 0

:refreshpath
rem Pick up PATH changes from installers without opening a new window.
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')"`) do set "PATH=%%P"
exit /b 0

:findpython
rem The Microsoft Store "python" alias exists even when Python is not installed, so test that it really runs.
set "PY="
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python" && exit /b 0
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=py -3" && exit /b 0
exit /b 0
