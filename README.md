


# Whisper GPU Transcription Setup (Windows Native)
A complete working reference for setting up **faster-whisper (large-v3)** on **Windows with CUDA GPU acceleration**, verified on RTX 3080 with CUDA 12.9 and cuDNN 9.14.

## TLDR

- you need ffmpeg, downloaded somewhere
- you need a powerful gpu to get the transcripts at 6x speed
- these instructions suck


## Prerequisites

### Install Microsoft Visual C++ Redistributables
Whisper and CUDA shaders rely on MSVC runtime.

Install:
Microsoft Visual C++ 2015-2022 Redistributable (x64)

Download:
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

Without this, some DLL loads fail silently.

### Must have Audio!

Your mp4 must have audio streams!

`for %i in (*.mp4) do ffprobe "%i" | findstr /r /c:"Audio:"`

### Assumes ffmpeg in path

Whisper requires proper audio stream decoding. FFmpeg handles audio demux, resample, silence, VFR, and codecs.

Required executables:
- C:\ffmpeg\bin\ffmpeg.exe
- C:\ffmpeg\bin\ffprobe.exe
- C:\ffmpeg\bin\ffplay.exe (optional)


### FFmpeg Windows Download

Download static build:
https://www.gyan.dev/ffmpeg/builds/

Use:
ffmpeg-release-full.7z or ffmpeg-release-essentials.7z

Extract to:
C:\ffmpeg\

Ensure:
C:\ffmpeg\bin in PATH

### Verify FFmpeg Works

Open terminal:
`ffmpeg -version`
`ffprobe -version`

If version prints successfully, PATH resolution is correct.

--------------------------------------------------------------------

## Folder Structure Overview (Updated With FFmpeg)
```
C:\
├── ffmpeg\
│   └── bin\
│       ├── ffmpeg.exe
│       ├── ffplay.exe
│       └── ffprobe.exe
│
├── models\
│   ├── models--Systran--faster-whisper-large-v3\
│   └── .locks\
│
├── v\
│   └── whisper_transcribe\
│       ├── Lib\
│       ├── Scripts\
│       ├── Include\
│       ├── main.py
│       ├── start.bat
│       ├── models\
│       ├── videos_in\
│       ├── transcripts_out\
│       └── pyvenv.cfg
│
└── Program Files\
    └── NVIDIA GPU Computing Toolkit\
        └── CUDA\
            └── v12.9\
                ├── bin\
                │   ├── cudnn_ops64_9.dll
                │   ├── cudnn_cnn64_9.dll
                │   ├── cudnn_adv64_9.dll
                │   └── others...
                ├── lib\
                └── include\
```

----------------------------------------------------------------------
## 1. GPU Driver & Toolkit

### **Verify NVIDIA GPU**
Confirm GPU is visible, driver around 581.xx or newer.

`nvidia-smi`

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.29                 Driver Version: 581.29         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080      WDDM  |   00000000:01:00.0  On |                  N/A |
| 46%   58C    P2             96W /  320W |    4060MiB /  10240MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### Baseline GPU Temp, driver, model and usage if desired:

```
PS C:\Users\user> nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv -l 2
utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], temperature.gpu
3 %, 0 %, 686 MiB, 10240 MiB, 38
3 %, 0 %, 686 MiB, 10240 MiB, 39
1 %, 0 %, 686 MiB, 10240 MiB, 39
1 %, 0 %, 686 MiB, 10240 MiB, 39
2 %, 0 %, 686 MiB, 10240 MiB, 39
3 %, 0 %, 686 MiB, 10240 MiB, 40
1 %, 0 %, 686 MiB, 10240 MiB, 40
```

```
PS C:\Users\user> Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM

Name                            DriverVersion   AdapterRAM
----                            -------------   ----------
NVIDIA GeForce RTX 3080         32.0.15.8129    4293918720
Microsoft Basic Display Adapter 10.0.19041.3636          0
```

### Enforce Max Power
NVIDIA Control Panel → Manage 3D Settings → Power Management Mode → Prefer Maximum Performance


### **Install CUDA Toolkit 12.9.x**
Download from:
https://developer.nvidia.com/cuda-downloads

version 13 did not work!

Path after install:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\

----------------------------------------------------------------------

## 2. Virtual Environment Setup

**Create Folder & venv**
```
C:\> mkdir v
C:\v> python -m venv whisper_transcribe
C:\v\whisper_transcribe\Scripts\activate
```
**Install Required Packages**
`pip install faster-whisper colorama numpy torch nvidia-cudnn-cu12 nvidia-cublas-cu12`

## 3. cuDNN Installation (Hackery Method)

**Install cuDNN via pip**
`pip install nvidia-cudnn-cu12`

This downloads cuDNN 9.x for CUDA 12 and places it in:
C:\v\whisper_transcribe\Lib\site-packages\nvidia\cudnn\

however, it's not in path and won't work here (blame nvidia)

**Manual DLL Integration**
Copy:
bin\*.dll  →  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\
lib\*.lib  →  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\
include\*.h  →  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include\

Add to PATH (if missing):
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin

Reboot after copying, suggested, but it worked without reboot

----------------------------------------------------------------------
## 4. Model Caching

Optional but recommended to cache the model manually so you know the location of the large models.

Set environment variable or folder in the start.bat file:

C:\models
Inside:
models--Systran--faster-whisper-large-v3\
(Downloaded automatically on first run)

----------------------------------------------------------------------
## 5. Batch Script for Launch

```
**start.bat**
@echo off
set HUGGINGFACE_HUB_CACHE=C:\models
C:
cd \v\whisper_transcribe\
call Scripts\activate.bat
python main.py
pause
```
----------------------------------------------------------------------
## 6. Main Transcription Script (main.py)

See `main.py` from this build:
- Supports progress bar
- ETA and elapsed time
- Predictive benchmark (3.00 throughput default)
- Final throughput reporting

Core Parameters:
device="cuda"
compute_type="float16"
model="large-v3"

Benchmark constant is defined at the top:
BENCHMARK_THROUGHPUT = 3.00

----------------------------------------------------------------------
## 7. Benchmarking and Predictive Runtime

On first successful run:
```
Video Duration:      hh:mm:ss
Processing Time:     hh:mm:ss
Throughput:          X.XXx (video sec per wall sec)
Benchmark Suggest:   Use BENCHMARK_THROUGHPUT=X.XX
```
Update this constant in main.py for accurate ETA on future videos.

Example:
If Throughput = 3.00
1-hour video takes ~20 minutes.

----------------------------------------------------------------------
## 8. Using the Tool

**Input Folder**
Place .mp4 files with audio in:
C:\v\whisper_transcribe\videos_in

**Output Folder**
Text transcripts appear in:
C:\v\whisper_transcribe\transcripts_out

**Run**
Double-click start.bat

----------------------------------------------------------------------
## 9. Verification Tests

Check GPU availability in Python:
python -c "import ctranslate2 as ct; print(ct.get_supported_compute_types('cuda'))"
Expected output includes "float16"

Check cuDNN linkage:
where cudnn_ops64_9.dll
Expected path: ...CUDA\v12.9\bin\cudnn_ops64_9.dll

----------------------------------------------------------------------
## 10. Known Working Versions

- Windows 10/11 x64
- NVIDIA Driver 581.29
- CUDA Toolkit 12.9.1 / 12.9.3
- cuDNN 9.14.0.64
- faster-whisper 1.0+
- Python 3.12.6+
- RTX 3080 (Ampere)
- Model: large-v3

----------------------------------------------------------------------
## 11. Maintenance Tips

- If errors mention cudnn_ops64_9.dll missing, re-copy DLLs.
- If no progress bar appears, ensure faster-whisper version supports streaming.
- Reboot after any CUDA/cuDNN path changes.
- Store your benchmark constant for consistency.

----------------------------------------------------------------------
## 12. Example Session Output

Processing: training_day.mp4  
Predicted Runtime: 0:47:02  
Video Duration:    2:21:06  
[######################------------------------]  52%  ETA: 0:23:14  Elapsed: 0:23:48  
-----------------------------------------  
Video Duration:      2:21:06  
Processing Time:     0:47:00  
Throughput:          3.00x (video sec per wall sec)  
Benchmark Suggest:   Use BENCHMARK_THROUGHPUT=3.00  
-----------------------------------------  

----------------------------------------------------------------------
## Summary

This setup yields a **stable, native GPU-accelerated transcription system** for large-v3 whisper on Windows.  
Predictive benchmarking keeps future job times accurate without manual tuning.  
Once installed, you can process any number of videos just by dropping them into `videos_in`.

End of Setup Guide.
```
