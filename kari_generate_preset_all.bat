@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
set "DATA_ROOT=%CD%\data"
set "PRESET_ROOT=%CD%\presets"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  exit /b 1
)

if not exist "%DATA_ROOT%" (
  echo [ERROR] Data root not found: %DATA_ROOT%
  exit /b 1
)

if not exist "%PRESET_ROOT%" (
  mkdir "%PRESET_ROOT%"
)

set /a RUN_COUNT=0
set /a SKIP_COUNT=0
set /a FAIL_COUNT=0

echo [INFO] Data root   : %DATA_ROOT%
echo [INFO] Preset root : %PRESET_ROOT%
echo.

for /d %%D in ("%DATA_ROOT%\*") do (
  set "NAME=%%~nD"
  set "AUDIO_DIR=%%~fD\raw"
  set "OUT_DIR=%PRESET_ROOT%\!NAME!"

  if not exist "!AUDIO_DIR!" (
    echo [SKIP] !NAME! ^(raw folder not found^)
    set /a SKIP_COUNT+=1
  ) else (
    if not exist "!OUT_DIR!" (
      mkdir "!OUT_DIR!"
    )

    echo [RUN ] !NAME!
    "%PYTHON_EXE%" ".\scripts\generate_preset_multi.py" ^
      --audio-dir "!AUDIO_DIR!" ^
      --audio-glob "**/*" ^
      --extensions ".wav,.flac,.ogg,.mp3,.m4a" ^
      --preset-id "!NAME!" ^
      --output-dir "!OUT_DIR!" ^
      --device "cuda" ^
      --segment-seconds 15 ^
      --segment-hop-seconds 15 ^
      --min-segment-seconds 3 ^
      --keep-ratio 0.8 ^
      --max-clusters 4 ^
      --cluster-min-size 3 ^
      --max-files 0 ^
      --max-segments 0 ^
      --save-meta

    if errorlevel 1 (
      echo [FAIL] !NAME!
      set /a FAIL_COUNT+=1
    ) else (
      echo [ OK ] !NAME! output !OUT_DIR!
    )
    set /a RUN_COUNT+=1
    echo.
  )
)

echo [DONE] run=%RUN_COUNT% skip=%SKIP_COUNT% fail=%FAIL_COUNT%
exit /b 0

