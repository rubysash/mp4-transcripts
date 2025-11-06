@echo off
set HUGGINGFACE_HUB_CACHE=C:\models
C:
cd \v\whisper_transcribe\
call Scripts\activate.bat
python main.py
pause