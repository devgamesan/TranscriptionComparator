import whisper
import os
import time
from typing import Optional, Dict, Any
import torch
from base import BaseTranscriber, append_result, decide_device, get_args

class WhisperTranscriber(BaseTranscriber):
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        """Whisperモデルの初期化"""
        self.model_name = model_name
        self.device = device

        print(f"Loading {self.model_name} model on {self.device}...")
        start_time = time.time()
        self.model = whisper.load_model(self.model_name, device=self.device)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        temperature: float = 0.0,
        best_of: int = 5,
        beam_size: int = 5,
        fp16: bool = True
    ) -> Dict[str, Any]:
        """音声ファイルの文字起こしを実行"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing: {audio_path}")
        print(f"Language: {language if language else 'auto-detect'}")

        start_time = time.time()

        # Whisperモデルで文字起こし実行
        transcribe_result = self.model.transcribe(
            audio_path,
            language=language,
            temperature=temperature,
            best_of=best_of,
            beam_size=beam_size,
            fp16=fp16,
        )

        transcribe_time = time.time() - start_time
        print(f"Transcription completed in {transcribe_time:.2f} seconds")

        # セグメントからテキストを結合
        segment_texts = [seg["text"].strip() for seg in transcribe_result["segments"]]
        result = " ".join(segment_texts)

        return result, transcribe_time


def transcribe(model_name: str, device: str, audio_path: str, language: str = "ja"):
    """指定されたモデルで文字起こしを実行するラッパー関数"""
    transcriber = WhisperTranscriber(model_name=model_name, device=device)
    result, transcribe_time = transcriber.transcribe(
        audio_path=audio_path,
        language=language,
    )
    print(f"モデル: {model_name}, 所要時間秒:{transcribe_time:.1f}, 文字起こし結果:{result}")
    transcriber.unload_model()
    return result, transcribe_time


def main():
    """メイン関数: 単一のWhisperモデルで文字起こしを実行"""
    args = get_args()

    # モデル名が指定されていない場合、デフォルトのモデルを使用
    model_name = args.model_name if hasattr(args, 'model_name') and args.model_name else "large-v3"

    result, time = transcribe(model_name, device=decide_device(), audio_path=args.audio_path)
    append_result(args.result_file, args.audio_path, "OpenAI Whisper", model_name, time, result)


if __name__ == "__main__":
    main()