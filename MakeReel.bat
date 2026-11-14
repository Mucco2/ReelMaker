@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title ReelMaker
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
  echo ReelMaker is not installed yet - running setup first...
  call setup.bat
)

echo.
echo  ===== ReelMaker =====
echo  Turns any video into a TikTok / Reels / Shorts ready vertical video.
echo.

set "VIDEO=%~1"
if not defined VIDEO set /p "VIDEO=Paste a video file path or a link and press Enter: "
if not defined VIDEO ( echo No video given. & pause & exit /b 1 )
set "VIDEO=%VIDEO:"=%"

set "HOOK="
set /p "HOOK=Hook text shown at the start (optional, Enter to skip): "
if defined HOOK set "HOOK=%HOOK:"=%"

set "REF="
set /p "REF=Link to a reel whose style you like (optional, Enter to skip): "
if defined REF set "REF=%REF:"=%"

set "LEN="
set /p "LEN=Max length in seconds (Enter = 60): "

set "ARGS="
if defined HOOK set ARGS=%ARGS% --hook "%HOOK%"
if defined REF set ARGS=%ARGS% --reference "%REF%"
if defined LEN set ARGS=%ARGS% --max-duration %LEN%

echo.
".venv\Scripts\python.exe" -m reelmaker "%VIDEO%" %ARGS%
if errorlevel 1 (
  echo.
  echo Something went wrong - see the message above.
  pause
  exit /b 1
)

echo.
echo Opening the output folder...
for /f "delims=" %%D in ('dir /b /ad /o-d "output" 2^>nul') do ( start "" "output\%%D" & goto :done )
:done
pause
