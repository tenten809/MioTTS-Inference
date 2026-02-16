# MioCodec 追加学習フロー（声質寄せ）

目的:
- `MioCodec` のうち content token 系は固定し、global/decoder 側のみ追加学習して声質を寄せる
- 生成後は `MIOTTS_CODEC_ADAPTER` で推論時に適用（ベースモデルはそのまま）

## 1) 学習実行

```powershell
cd E:\Python\MioTTS-Inference

.\.venv\Scripts\python.exe .\scripts\train_codec_adapter.py `
  --esd-list "E:\Python\MioTTS-Inference\data\test1\esd.list" `
  --audio-root "E:\Python\MioTTS-Inference\data\test1\raw" `
  --codec-model-id "Aratako/MioCodec-25Hz-44.1kHz-v2" `
  --output-adapter "E:\Python\MioTTS-Inference\outputs\codec_adapter_test1.safetensors" `
  --device cuda `
  --epochs 8 `
  --lr 1e-5 `
  --eval-ratio 0.1 `
  --min-audio-sec 1 `
  --max-audio-sec 20 `
  --train-global-encoder `
  --train-decoder `
  --stft-loss-weight 0.5 `
  --print-every 20
```

出力:
- `outputs\codec_adapter_test1.safetensors`
- `outputs\codec_adapter_test1.json`

## 2) 推論時にadapter適用

PowerShell で環境変数を設定して API サーバを起動:

```powershell
$env:MIOTTS_CODEC_MODEL = "Aratako/MioCodec-25Hz-44.1kHz-v2"
$env:MIOTTS_CODEC_ADAPTER = "E:\Python\MioTTS-Inference\outputs\codec_adapter_test1.safetensors"

.\.venv\Scripts\python.exe .\run_server.py --llm-base-url http://localhost:8000/v1 --best-of-n-enabled
```

## 3) 注意点

- この学習は `content token` 意味空間を壊さない前提（content側は凍結）
- 学習を強くしすぎると破綻しやすいので、まずは `lr=1e-5` 付近から開始
- かすれが増える場合は `--train-decoder` を外し、globalのみ学習して比較
