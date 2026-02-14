# MioTTS LoRA 学習フロー（`data/test1`）

対象:
- 音声: `E:\Python\MioTTS-Inference\data\test1\raw\*.wav`
- 文字起こし: `E:\Python\MioTTS-Inference\data\test1\esd.list`
- ベースモデル: `E:\Python\MioTTS-Inference\models\MioTTS-2.6B`

## 0) 依存を入れる

```powershell
cd E:\Python\MioTTS-Inference
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install "peft>=0.17.0"
```

## 1) 学習JSONLを作る（音声 -> speech token）

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_lora_dataset.py `
  --esd-list "E:\Python\MioTTS-Inference\data\test1\esd.list" `
  --audio-root "E:\Python\MioTTS-Inference\data\test1\raw" `
  --output-jsonl "E:\Python\MioTTS-Inference\data\test1\train_lora.jsonl" `
  --codec-model-id "Aratako/MioCodec-25Hz-24kHz" `
  --device "cuda" `
  --max-audio-sec 30 `
  --print-every 20
```

## 2) LoRA学習

最初は attention のみを学習対象にする設定を推奨。

```powershell
.\.venv\Scripts\python.exe .\scripts\train_lora.py `
  --base-model "E:\Python\MioTTS-Inference\models\MioTTS-2.6B" `
  --train-jsonl "E:\Python\MioTTS-Inference\data\test1\train_lora.jsonl" `
  --output-dir "E:\Python\MioTTS-Inference\outputs\lora_test1_attn" `
  --target-modules "q_proj,k_proj,v_proj,out_proj" `
  --lora-r 16 `
  --lora-alpha 32 `
  --lora-dropout 0.05 `
  --max-length 2048 `
  --epochs 2 `
  --learning-rate 2e-4 `
  --train-batch-size 1 `
  --gradient-accumulation-steps 16 `
  --dtype bf16 `
  --attn-implementation sdpa `
  --save-steps 200
```

## 3) adapterをベースにマージ（推論用）

```powershell
.\.venv\Scripts\python.exe .\scripts\merge_lora_adapter.py `
  --base-model "E:\Python\MioTTS-Inference\models\MioTTS-2.6B" `
  --adapter-dir "E:\Python\MioTTS-Inference\outputs\lora_test1_attn" `
  --output-dir "E:\Python\MioTTS-Inference\models\MioTTS-2.6B-lora-test1" `
  --dtype bf16
```

## 4) 推論サーバで使う

WSL/Linux の vLLM 例:

```bash
vllm serve /mnt/e/Python/MioTTS-Inference/models/MioTTS-2.6B-lora-test1 --max-model-len 4096
```

その後、MioTTS API:

```powershell
.\.venv\Scripts\python.exe .\run_server.py --llm-base-url http://localhost:8000/v1
```

## 5) うまくいかない場合の調整

- まず増やす: `--epochs`（2 -> 3）, `--save-steps` を小さくして中間確認
- 過学習気味: `--learning-rate` を下げる（`2e-4 -> 1e-4`）
- 効きが弱い: `--target-modules` に `w1,w2,w3` を追加して再学習
- VRAM不足: `--max-length` を `1024` へ下げる / 勾配蓄積を増やす

