# 説明
次の音声認識モデルについて一括で文字起こしを実行し、結果を出力するためのDockerイメージ&スクリプト。

- OpenAI Whisper: large-v2, large-v3, large-v3-turbo
- Faster Whisper: large-v2, large-v3, large-v3-turbo, kotoba-tech/kotoba-whisper-v2.0-faster
- FunASR: unAudioLLM/Fun-ASR-Nano-2512
- ReazonSpeech: nemo, k2, espnet

音声認識結果は以下のようなCSVフォーマットで複数モデルによる認識結果がまとめて出力される。

```csv
AUDIO_FILE,OSS,MODEL,PROCESS_TIME,TRANSCRIPTION
inputs/input.mp4,OpenAI Whisper,large-v2,23.91,おはようございます。
inputs/input.mp4,OpenAI Whisper,large-v3-turbo,4.86,おはようございます。
inputs/input.mp4,Faster Whisper,large-v2,13.14,おはようございます。
inputs/input.mp4,Faster Whisper,large-v3,8.37,おはようございます。
inputs/input.mp4,Faster Whisper,large-v3-turbo,2.36,おはようございます。
inputs/input.mp4,Faster Whisper,kotoba-tech/kotoba-whisper-v2.0-faster,1.54,おはようございます。
inputs/input.mp4,ReazonSpeech,k2,9.73,おはようございます。
inputs/input.mp4,FunASR,FunAudioLLM/Fun-ASR-Nano-2512,7.78,おはようございます。

```

# 使用方法
## 1. Dockerイメージビルド
docker build -t transcribers-app .

## 2. コンテナ起動

```bash
docker run --rm -it \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)/inputs:/home/ubuntu/app/inputs" \
    -v "$(pwd)/outputs:/home/ubuntu/app/outputs" \
    -v "$HOME/.cache/huggingface:/home/ubuntu/.cache/huggingface" \
    -v "$HOME/.cache/modelscope:/home/ubuntu/.cache/modelscope" \
    -v "$HOME/.cache/whisper:/home/ubuntu/.cache/whisper" \
    --gpus all \
    transcribers-app bash
```

inputs: 入力とする音声ファイル置き場
outputs: 音声認識結果の出力先ディレクトリ


## 3. 文字起こしを実行する

```bash
./run.sh
```

run.shは以下のようにデフォルトの音声入力元としてinputs/input.mp4, 認識結果のデフォルトの出力先としてoutputs/result.csvを指定している。必要に応じて変更して使用すること。

```sh
python ./src/transcriber_whipser.py  --audio-path inputs/input.mp4 --result-file outputs/result.csv
```

## 4. Qwen0.6Bのモデルを手動で配置する
FunARSの実行時にモデルの取得が行われるが、Qwen0.6Bのモデルが[Fun](FunAudioLLM/Fun-ASR-Nano-2512)にアップロードされておらず、Qwen0.6Bのモデルがないことで初回はrun.shはFunARSは必ず失敗する。このため、初回実行後に以下を実行し、手動でQwen0.6Bのモデルを配置する。

```bash
curl -L https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors \
     -o ~/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512/Qwen3-0.6B/model.safetensors
```