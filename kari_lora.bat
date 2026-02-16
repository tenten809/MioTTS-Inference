@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
set "DATA_ROOT=%CD%\data"
set "CODEC_MODEL_ID=Aratako/MioCodec-25Hz-44.1kHz-v2"
set "DEVICE=cuda"
set "MAX_AUDIO_SEC=30"
set "PRINT_EVERY=20"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  exit /b 1
)

if not exist "%DATA_ROOT%" (
  echo [ERROR] Data root not found: %DATA_ROOT%
  exit /b 1
)

set /a RUN_COUNT=0
set /a SKIP_COUNT=0

echo [INFO] Data root: %DATA_ROOT%
echo.

for /d %%D in ("%DATA_ROOT%\*") do (
  set "TARGET_DIR=%%~fD"
  set "NAME=%%~nD"
  set "ESD_LIST=!TARGET_DIR!\esd.list"
  set "AUDIO_ROOT=!TARGET_DIR!\raw"
  set "OUT_JSONL=!TARGET_DIR!\train_lora.jsonl"

  if not exist "!ESD_LIST!" (
    echo [SKIP] !NAME! ^(esd.list not found^)
    set /a SKIP_COUNT+=1
  ) else if not exist "!AUDIO_ROOT!" (
    echo [SKIP] !NAME! ^(raw folder not found^)
    set /a SKIP_COUNT+=1
  ) else (
    echo [RUN ] !NAME!
    "%PYTHON_EXE%" ".\scripts\prepare_lora_dataset.py" ^
      --esd-list "!ESD_LIST!" ^
      --audio-root "!AUDIO_ROOT!" ^
      --output-jsonl "!OUT_JSONL!" ^
      --codec-model-id "%CODEC_MODEL_ID%" ^
      --device "%DEVICE%" ^
      --max-audio-sec %MAX_AUDIO_SEC% ^
      --print-every %PRINT_EVERY%

    if errorlevel 1 (
      echo [FAIL] !NAME!
    ) else (
      echo [ OK ] !NAME! output !OUT_JSONL!
    )
    set /a RUN_COUNT+=1
    echo.
  )
)

echo [DONE] run=%RUN_COUNT% skip=%SKIP_COUNT%
exit /b 0
