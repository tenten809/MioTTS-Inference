@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
set "DATA_ROOT=%CD%\data"
set "BASE_MODEL=%CD%\models\MioTTS-2.6B"
set "LORA_ROOT=%CD%\loras"

set "TARGET_MODULES=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj"
set "LORA_R=32"
set "LORA_ALPHA=64"
set "LORA_DROPOUT=0.1"
set "MAX_LENGTH=2048"
set "EPOCHS=10"
set "LEARNING_RATE=1e-4"
set "WEIGHT_DECAY=0.01"
set "TRAIN_BATCH_SIZE=2"
set "GRAD_ACC=16"
set "DTYPE=bf16"
set "ATTN_IMPL=sdpa"
set "SAVE_STEPS=50"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  exit /b 1
)

if not exist "%DATA_ROOT%" (
  echo [ERROR] Data root not found: %DATA_ROOT%
  exit /b 1
)

if not exist "%BASE_MODEL%" (
  echo [ERROR] Base model not found: %BASE_MODEL%
  exit /b 1
)

if not exist "%LORA_ROOT%" (
  mkdir "%LORA_ROOT%"
)

set /a RUN_COUNT=0
set /a SKIP_COUNT=0
set /a FAIL_COUNT=0

echo [INFO] Data root  : %DATA_ROOT%
echo [INFO] Base model : %BASE_MODEL%
echo [INFO] LoRA root  : %LORA_ROOT%
echo.

for /d %%D in ("%DATA_ROOT%\*") do (
  set "TARGET_DIR=%%~fD"
  set "NAME=%%~nD"
  set "TRAIN_JSONL=!TARGET_DIR!\train_lora.jsonl"
  set "OUT_DIR=%LORA_ROOT%\!NAME!"

  if not exist "!TRAIN_JSONL!" (
    echo [SKIP] !NAME! ^(train_lora.jsonl not found^)
    set /a SKIP_COUNT+=1
  ) else (
    echo [RUN ] !NAME!
    "%PYTHON_EXE%" ".\scripts\train_lora.py" ^
      --base-model "%BASE_MODEL%" ^
      --train-jsonl "!TRAIN_JSONL!" ^
      --output-dir "!OUT_DIR!" ^
      --target-modules "%TARGET_MODULES%" ^
      --lora-r %LORA_R% ^
      --lora-alpha %LORA_ALPHA% ^
      --lora-dropout %LORA_DROPOUT% ^
      --max-length %MAX_LENGTH% ^
      --epochs %EPOCHS% ^
      --learning-rate %LEARNING_RATE% ^
      --weight-decay %WEIGHT_DECAY% ^
      --train-batch-size %TRAIN_BATCH_SIZE% ^
      --gradient-accumulation-steps %GRAD_ACC% ^
      --dtype %DTYPE% ^
      --attn-implementation %ATTN_IMPL% ^
      --save-steps %SAVE_STEPS%

    if errorlevel 1 (
      echo [FAIL] !NAME!
      set /a FAIL_COUNT+=1
    ) else (
      echo [ OK ] !NAME! -> !OUT_DIR!
    )
    set /a RUN_COUNT+=1
    echo.
  )
)

echo [DONE] run=%RUN_COUNT% skip=%SKIP_COUNT% fail=%FAIL_COUNT%
exit /b 0

