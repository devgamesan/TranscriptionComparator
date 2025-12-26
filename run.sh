#!/bin/bash

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# デフォルト値の設定
AUDIO_PATH="${SCRIPT_DIR}/inputs/input.mp4"
RESULT_FILE="${SCRIPT_DIR}/outputs/result.csv"

# 引数の解析
while [[ $# -gt 0 ]]; do
  case $1 in
    --audio-path)
      AUDIO_PATH="$2"
      shift 2
      ;;
    --result-file)
      RESULT_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--audio-path AUDIO_PATH] [--result-file RESULT_FILE]"
      exit 1
      ;;
  esac
done

# 引数の確認
echo "Audio path: ${AUDIO_PATH}"
echo "Result file: ${RESULT_FILE}"

# 実行するスクリプト
echo "Starting transcription process..."

# OpenAI Whisper
source "${SCRIPT_DIR}/.venv_whisper/bin/activate"
python "${SCRIPT_DIR}/src/transcriber_whisper.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "base"
python "${SCRIPT_DIR}/src/transcriber_whisper.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "large-v2"
python "${SCRIPT_DIR}/src/transcriber_whisper.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "large-v3"
python "${SCRIPT_DIR}/src/transcriber_whisper.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "large-v3-turbo"

# Faster Whisper
source "${SCRIPT_DIR}/.venv_fasterwhisper/bin/activate"
python "${SCRIPT_DIR}/src/transcriber_fasterwhisper.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "large-v2"
python "${SCRIPT_DIR}/src/transcriber_fasterwhisper.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "large-v3"
python "${SCRIPT_DIR}/src/transcriber_fasterwhisper.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "large-v3-turbo"
python "${SCRIPT_DIR}/src/transcriber_fasterwhisper.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "kotoba-tech/kotoba-whisper-v2.0-faster"

# ReazonSpeech
source "${SCRIPT_DIR}/.venv_reazonspeech/bin/activate"
python "${SCRIPT_DIR}/src/transcriber_reazonspeech.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "nemo"
python "${SCRIPT_DIR}/src/transcriber_reazonspeech.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "k2"
python "${SCRIPT_DIR}/src/transcriber_reazonspeech.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "espnet"

# FunASR
source "${SCRIPT_DIR}/.venv_funasr/bin/activate"
python "${SCRIPT_DIR}/src/transcriber_funasr.py" --audio-path "${AUDIO_PATH}" --result-file "${RESULT_FILE}" --model-name "FunAudioLLM/Fun-ASR-Nano-2512"