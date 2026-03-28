@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem MioTTS stack launcher (Windows)
rem Starts:
rem   1) llama-server (LLM backend)
rem   2) MioTTS API server (run_server.py)
rem   3) Gradio UI script (configurable via GRADIO_SCRIPT)
rem ============================================================

cd /d "%~dp0"

rem ----------------------------
rem User settings
rem ----------------------------
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PAUSE_ON_ERROR set "PAUSE_ON_ERROR=1"
if not defined PAUSE_ON_SUCCESS set "PAUSE_ON_SUCCESS=1"

rem Base GGUF model for llama-server (relative to repo root)
set "BASE_GGUF=models\MioTTS-2.6B-BF16.gguf"

rem Runtime LoRA GGUF adapter (relative path recommended; blank to disable)
set "LORA_ADAPTER=\loras\60s_documentary_v2\adapter-lora-bf16.gguf"
set "LORA_SCALE=0.80"
set "LORAS_DIR=loras"
set "LOAD_ALL_LORAS=1"

rem Ports
set "LLM_PORT=8000"
set "MIOTTS_PORT=8001"
set "GRADIO_PORT=7860"
set "GRADIO_SCRIPT=run_gradio_line_batch.py"

rem llama-server runtime settings
set "LLM_CTX=8192"
set "LLM_BATCH=8"

rem MioTTS settings
set "ENABLE_BEST_OF_N=true"
set "CODEC_MODEL=Aratako/MioCodec-25Hz-44.1kHz-v2"
rem Optional: codec adapter safetensors path (relative to repo). Leave blank to disable.
set "CODEC_ADAPTER="
rem OCR resize cap (4K equivalent = 3840x2160)
set "MIOTTS_OCR_MAX_PIXELS=8294400"

rem Optional: set explicitly. Leave blank to auto-detect.
set "LLAMA_SERVER_EXE="

rem ----------------------------
rem Validation
rem ----------------------------
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  echo         Please create venv and install dependencies first.
  goto :error_exit
)

if not exist "%BASE_GGUF%" (
  echo [ERROR] Base GGUF not found: %BASE_GGUF%
  goto :error_exit
)

if not exist "%GRADIO_SCRIPT%" (
  echo [ERROR] Gradio script not found: %GRADIO_SCRIPT%
  goto :error_exit
)

if defined LLAMA_SERVER_EXE (
  if not exist "%LLAMA_SERVER_EXE%" (
    for /f "delims=" %%I in ('where "%LLAMA_SERVER_EXE%" 2^>nul') do (
      set "LLAMA_SERVER_EXE=%%~fI"
      goto :llama_found
    )
    echo [WARN] Configured LLAMA_SERVER_EXE was not found: %LLAMA_SERVER_EXE%
    set "LLAMA_SERVER_EXE="
  )
)

if not defined LLAMA_SERVER_EXE (
  set "WINGET_LLAMA=%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
  if exist "!WINGET_LLAMA!" set "LLAMA_SERVER_EXE=!WINGET_LLAMA!"
)

if not defined LLAMA_SERVER_EXE (
  for /f "delims=" %%I in ('dir /b /s "%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp*\\llama-server.exe" 2^>nul') do (
    set "LLAMA_SERVER_EXE=%%~fI"
    goto :llama_found
  )
)

if not defined LLAMA_SERVER_EXE (
  for /f "delims=" %%I in ('where llama-server 2^>nul') do (
    set "LLAMA_SERVER_EXE=%%~fI"
    goto :llama_found
  )
)

:llama_found
if not defined LLAMA_SERVER_EXE (
  echo [ERROR] llama-server not found.
  echo         Set LLAMA_SERVER_EXE in this bat file.
  goto :error_exit
)

if defined LORA_ADAPTER (
  if not exist "%LORA_ADAPTER%" (
    echo [WARN] LoRA adapter not found: %LORA_ADAPTER%
    echo [WARN] Continue without single LoRA adapter.
    set "LORA_ADAPTER="
  )
)

if defined LORA_ADAPTER (
  rem if absolute path starts with current repo path, convert to relative for --lora-scaled
  set "LORA_ADAPTER=!LORA_ADAPTER:%CD%\=!"
)

set "ALL_LORAS="
set "ALL_LORAS_COUNT=0"
if "%LOAD_ALL_LORAS%"=="1" (
  if exist "%LORAS_DIR%" (
    for /r "%LORAS_DIR%" %%F in (*.gguf) do (
      set "ONE_LORA=%%~fF"
      set "ONE_LORA=!ONE_LORA:%CD%\=!"
      if defined ALL_LORAS (
        set "ALL_LORAS=!ALL_LORAS!,!ONE_LORA!"
      ) else (
        set "ALL_LORAS=!ONE_LORA!"
      )
      set /a ALL_LORAS_COUNT+=1
    )
  )
)

if "%LOAD_ALL_LORAS%"=="1" if not defined ALL_LORAS if not defined LORA_ADAPTER (
  echo [WARN] No LoRA files found under %LORAS_DIR%. Continue without LoRA.
)

rem ----------------------------
rem Shared env for child processes
rem ----------------------------
set "MIOTTS_BEST_OF_N_ENABLED=%ENABLE_BEST_OF_N%"
set "MIOTTS_CODEC_MODEL=%CODEC_MODEL%"
set "MIOTTS_LORAS_DIR=%LORAS_DIR%"
if defined CODEC_ADAPTER (
  if exist "%CODEC_ADAPTER%" (
    set "MIOTTS_CODEC_ADAPTER=%CODEC_ADAPTER%"
  ) else (
    echo [WARN] CODEC_ADAPTER not found: %CODEC_ADAPTER%
  )
)
set "MIOTTS_API_BASE=http://localhost:%MIOTTS_PORT%"
set "MIOTTS_LLM_BASE=http://localhost:%LLM_PORT%"
set "GRADIO_SERVER_PORT=%GRADIO_PORT%"

echo.
echo [INFO] Repo           : %CD%
echo [INFO] llama-server   : %LLAMA_SERVER_EXE%
echo [INFO] Base model     : %BASE_GGUF%
if defined ALL_LORAS (
  echo [INFO] LoRA mode      : all from %LORAS_DIR% count=%ALL_LORAS_COUNT%
) else if defined LORA_ADAPTER (
  echo [INFO] LoRA adapter   : %LORA_ADAPTER% scale=%LORA_SCALE%
) else (
  echo [INFO] LoRA adapter   : disabled
)
echo [INFO] LLM URL        : http://localhost:%LLM_PORT%
echo [INFO] MioTTS API URL : http://localhost:%MIOTTS_PORT%
echo [INFO] Gradio URL     : http://localhost:%GRADIO_PORT%
echo.

rem ----------------------------
rem Start llama-server
rem ----------------------------
call :check_port %LLM_PORT%
if defined PORT_PID (
  echo [INFO] llama-server port %LLM_PORT% already in use pid=%PORT_PID%. Skip start.
) else (
  if defined ALL_LORAS (
    start "llama-server" /d "%CD%" "%LLAMA_SERVER_EXE%" -m "%BASE_GGUF%" --lora "%ALL_LORAS%" --lora-init-without-apply --special --port %LLM_PORT% -c %LLM_CTX% --cont-batching --batch-size %LLM_BATCH%
  ) else if defined LORA_ADAPTER (
    set "USE_LORA_SCALED=1"
    if not "!LORA_ADAPTER::=!"=="!LORA_ADAPTER!" (
      echo [WARN] Absolute LoRA path may break --lora-scaled on Windows: !LORA_ADAPTER!
      echo [WARN] Falling back to --lora, no explicit scale.
      set "USE_LORA_SCALED="
    )
    if defined USE_LORA_SCALED (
      start "llama-server" /d "%CD%" "%LLAMA_SERVER_EXE%" -m "%BASE_GGUF%" --lora-scaled "%LORA_ADAPTER%:%LORA_SCALE%" --special --port %LLM_PORT% -c %LLM_CTX% --cont-batching --batch-size %LLM_BATCH%
    ) else (
      start "llama-server" /d "%CD%" "%LLAMA_SERVER_EXE%" -m "%BASE_GGUF%" --lora "%LORA_ADAPTER%" --special --port %LLM_PORT% -c %LLM_CTX% --cont-batching --batch-size %LLM_BATCH%
    )
  ) else (
    start "llama-server" /d "%CD%" "%LLAMA_SERVER_EXE%" -m "%BASE_GGUF%" --special --port %LLM_PORT% -c %LLM_CTX% --cont-batching --batch-size %LLM_BATCH%
  )
)

rem give LLM a little time before API starts
timeout /t 3 /nobreak >nul

rem ----------------------------
rem Start MioTTS API
rem ----------------------------
call :check_port %MIOTTS_PORT%
if defined PORT_PID (
  echo [INFO] MioTTS API port %MIOTTS_PORT% already in use pid=%PORT_PID%. Skip start.
) else (
  start "MioTTS API" /d "%CD%" "%PYTHON_EXE%" run_server.py --llm-base-url http://localhost:%LLM_PORT%/v1 --port %MIOTTS_PORT%
)

timeout /t 2 /nobreak >nul

rem ----------------------------
rem Start Gradio UI
rem ----------------------------
call :check_port %GRADIO_PORT%
if defined PORT_PID (
  echo [INFO] Gradio port %GRADIO_PORT% already in use pid=%PORT_PID%. Skip start.
) else (
  start "MioTTS Gradio" /d "%CD%" "%PYTHON_EXE%" "%GRADIO_SCRIPT%"
)

echo [INFO] All processes started.
echo [INFO] Close each opened window to stop each server.
if "%PAUSE_ON_SUCCESS%"=="1" (
  echo.
  pause
)
exit /b 0

:error_exit
if "%PAUSE_ON_ERROR%"=="1" (
  echo.
  pause
)
exit /b 1

:check_port
set "PORT_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%~1 .*LISTENING"') do (
  set "PORT_PID=%%P"
  goto :eof
)
goto :eof
