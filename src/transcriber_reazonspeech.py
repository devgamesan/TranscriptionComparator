import os
import time
from typing import Dict, Any
import importlib 

from base import BaseTranscriber, append_result, decide_device, get_args

class ReazonSpeechTranscriber(BaseTranscriber):
    """ReazonSpeech (nemo/k2/espnet) 用の汎用トランスクリプタ"""
    
    def __init__(self, backend: str = "nemo", device: str = "cuda"):
        # バックエンドの検証
        if backend not in ("nemo", "k2", "espnet"):
            raise ValueError("backend must be one of 'nemo', 'k2', or 'espnet'")
            
        self.backend = backend
        self.device = device
        
        print(f"Loading ReazonSpeech ({backend}) on {device}...")
        start_time = time.time()
        
        # 動的にモジュールをインポート（バックエンドごとに異なる実装）
        self.asr_module = importlib.import_module(f"reazonspeech.{backend}.asr")
        self.model = self.asr_module.load_model(device=device)
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        # ファイル存在チェック
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print(f"Transcribing with ReazonSpeech ({self.backend}): {audio_path}")
        
        start_time = time.time()
        # 音声ファイルの読み込みと文字起こし
        audio = self.asr_module.audio_from_path(audio_path)
        result_obj = self.asr_module.transcribe(self.model, audio)
        result = result_obj.text
        
        transcribe_time = time.time() - start_time
        print(f"Transcription completed in {transcribe_time:.2f} seconds")
        
        return result, transcribe_time

def transcribe(
    backend: str,
    device: str,
    audio_path: str,
):
    # トランスクリプタのインスタンス化と文字起こし実行
    transcriber = ReazonSpeechTranscriber(backend=backend, device=device)
    result, transcribe_time = transcriber.transcribe(audio_path=audio_path)
    transcriber.unload_model()
    return result, transcribe_time

def main():
    # コマンドライン引数の取得
    args = get_args()
    models = ["nemo", "k2", "espnet"]
    
    # 各バックエンドモデルで順次文字起こしを実行
    for model in models:
        result, time = transcribe(model, device=decide_device(), audio_path=args.audio_path)
        append_result(args.result_file, args.audio_path, "ReazonSpeech", model, time, result)

if __name__ == "__main__":
    main()