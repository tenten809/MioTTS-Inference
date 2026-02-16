@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

if not defined PYTHON_EXE set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined CONVERTER_PY set "CONVERTER_PY=.\third_party\llama.cpp\convert_lora_to_gguf.py"
if not defined LORA_ROOT set "LORA_ROOT=%CD%\loras"
if not defined BASE_MODEL set "BASE_MODEL=%CD%\models\MioTTS-2.6B"
if not defined OUTTYPE set "OUTTYPE=bf16"
if not defined OUTFILE_NAME set "OUTFILE_NAME=adapter-lora-bf16.gguf"
if not defined OVERWRITE set "OVERWRITE=0"
if not defined DRY_RUN set "DRY_RUN=0"
set "OVERWRITE_FLAG=%OVERWRITE: =%"
set "DRY_RUN_FLAG=%DRY_RUN: =%"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  exit /b 1
)

if not exist "%CONVERTER_PY%" (
  echo [ERROR] Converter script not found: %CONVERTER_PY%
  exit /b 1
)

if not exist "%LORA_ROOT%" (
  echo [ERROR] LoRA root not found: %LORA_ROOT%
  exit /b 1
)

if not exist "%BASE_MODEL%" (
  echo [ERROR] Base model not found: %BASE_MODEL%
  exit /b 1
)

set /a RUN_COUNT=0
set /a OK_COUNT=0
set /a SKIP_COUNT=0
set /a FAIL_COUNT=0

echo [INFO] LoRA root  : %LORA_ROOT%
echo [INFO] Base model : %BASE_MODEL%
echo [INFO] Outtype    : %OUTTYPE%
echo [INFO] Outfile    : %OUTFILE_NAME%
echo [INFO] Overwrite  : %OVERWRITE_FLAG%
echo [INFO] Dry run    : %DRY_RUN_FLAG%
echo.

for /f "delims=" %%D in ('dir /b /ad "%LORA_ROOT%" 2^>nul') do (
  set "IN_DIR=%LORA_ROOT%\%%D"
  set "NAME=%%D"
  set "OUT_FILE=!IN_DIR!\%OUTFILE_NAME%"
  set "HAS_LORA=0"

  if exist "!IN_DIR!\adapter_config.json" (
    if exist "!IN_DIR!\adapter_model.safetensors" set "HAS_LORA=1"
    if exist "!IN_DIR!\adapter_model.bin" set "HAS_LORA=1"
  )

  if "!HAS_LORA!"=="0" (
    echo [SKIP] !NAME! ^(adapter files not found^)
    set /a SKIP_COUNT+=1
    echo.
  ) else (
    if exist "!OUT_FILE!" (
      if "%OVERWRITE_FLAG%"=="1" (
        echo [RUN ] !NAME!
        if "%DRY_RUN_FLAG%"=="1" (
          echo        "%PYTHON_EXE%" "%CONVERTER_PY%" "!IN_DIR!" --base "%BASE_MODEL%" --outtype %OUTTYPE% --outfile "!OUT_FILE!"
          set /a OK_COUNT+=1
        ) else (
          "%PYTHON_EXE%" "%CONVERTER_PY%" ^
            "!IN_DIR!" ^
            --base "%BASE_MODEL%" ^
            --outtype %OUTTYPE% ^
            --outfile "!OUT_FILE!"
          if errorlevel 1 (
            echo [FAIL] !NAME!
            set /a FAIL_COUNT+=1
          ) else (
            echo [ OK ] !NAME! : !OUT_FILE!
            set /a OK_COUNT+=1
          )
        )
        set /a RUN_COUNT+=1
        echo.
      ) else (
        echo [SKIP] !NAME! ^(!OUT_FILE! already exists, set OVERWRITE=1 to regenerate^)
        set /a SKIP_COUNT+=1
        echo.
      )
    ) else (
      echo [RUN ] !NAME!
      if "%DRY_RUN_FLAG%"=="1" (
        echo        "%PYTHON_EXE%" "%CONVERTER_PY%" "!IN_DIR!" --base "%BASE_MODEL%" --outtype %OUTTYPE% --outfile "!OUT_FILE!"
        set /a OK_COUNT+=1
      ) else (
        "%PYTHON_EXE%" "%CONVERTER_PY%" ^
          "!IN_DIR!" ^
          --base "%BASE_MODEL%" ^
          --outtype %OUTTYPE% ^
          --outfile "!OUT_FILE!"
        if errorlevel 1 (
          echo [FAIL] !NAME!
          set /a FAIL_COUNT+=1
        ) else (
          echo [ OK ] !NAME! : !OUT_FILE!
          set /a OK_COUNT+=1
        )
      )
      set /a RUN_COUNT+=1
      echo.
    )
  )
)

echo [DONE] run=%RUN_COUNT% ok=%OK_COUNT% skip=%SKIP_COUNT% fail=%FAIL_COUNT%
if "%FAIL_COUNT%"=="0" (
  exit /b 0
) else (
  exit /b 1
)
