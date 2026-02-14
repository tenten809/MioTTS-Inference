# generate_preset_multi 実行メモ

対象スクリプト: `scripts/generate_preset_multi.py`  
目的: 多数の参照音声から1つの `preset` (`.pt`) を作る。

## 1) 前提

- 実行場所: リポジトリルート (`E:\Python\MioTTS-Inference`)
- Python: `.venv` を使う
- GPU利用時: `--device cuda`（CPUなら `--device cpu`）

## 2) 最短コマンド

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_preset_multi.py `
  --audio-dir "E:\dataset\my_voice_wavs" `
  --preset-id "my_voice_agg" `
  --output-dir ".\presets" `
  --device "cuda" `
  --save-meta
```

出力:
- `presets\my_voice_agg.pt`
- `presets\my_voice_agg.json`（`--save-meta` を付けた場合）

## 3) 30-60分データ向け推奨パラメータ

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_preset_multi.py `
  --audio-dir "E:\dataset\my_voice_wavs" `
  --audio-glob "**/*" `
  --extensions ".wav,.flac,.ogg,.mp3,.m4a" `
  --preset-id "my_voice_agg" `
  --output-dir ".\presets" `
  --device "cuda" `
  --segment-seconds 12 `
  --segment-hop-seconds 12 `
  --min-segment-seconds 3 `
  --keep-ratio 0.8 `
  --max-files 0 `
  --max-segments 0 `
  --save-meta
```

補足:
- `--keep-ratio 0.8` は外れ値除去用（上位80%を採用）
- 外れ値除去を無効化するなら `--disable-outlier-removal`

## 4) 複数ファイルを個別指定する例

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_preset_multi.py `
  --audio "E:\a.wav" `
  --audio "E:\b.wav" `
  --audio "E:\c.wav" `
  --preset-id "my_voice_manual" `
  --output-dir ".\presets" `
  --device "cuda"
```

## 5) 使い方（生成後）

- Gradioの `Reference Mode` を `preset` にする
- `Preset ID` に `my_voice_agg` を選ぶ（`my_voice_agg.pt` に対応）

## 6) よくある失敗

- `No audio files found.`
  - `--audio-dir` のパス/権限/`--extensions` を確認
- `No usable segments found.`
  - 無音が多い、または `--min-segment-seconds` が長すぎる
- CUDAメモリエラー
  - `--max-segments` で上限をつける、または `--device cpu` にする

