@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo ReelMaker er ikke installeret endnu - koerer setup foerst...
  call setup.bat
)
rem Modern window (opens as its own app window). The classic window is: pythonw -m reelmaker.gui
start "" ".venv\Scripts\pythonw.exe" -m reelmaker.webui "%~1"
exit /b 0
