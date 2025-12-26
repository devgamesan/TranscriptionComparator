import os
import time
from typing import Optional, Dict, Any
import torch
from base import BaseTranscriber, append_result, decide_device, get_args
from faster_whisper import WhisperModel


class FasterWhisperTranscriber(BaseTranscriber):
    def __init__(self, model_name: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        """Faster Whisperモデルの初期化"""
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

        print(f"Loading faster-whisper {self.model_name} on {self.device} with compute_type={self.compute_type}...")
        start_time = time.time()
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type
        )
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        temperature: float = 0.0,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> Dict[str, Any]:
        """音声ファイルを文字起こしするメイン関数"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing with faster-whisper: {audio_path}")
        print(f"Language: {language if language else 'auto-detect'}")
        print(f"VAD filter: {'enabled' if vad_filter else 'disabled'}")

        start_time = time.time()

        # 実際の文字起こし処理
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            vad_filter=vad_filter,
        )

        # セグメントを結合して結果を作成
        result = " ".join(segment.text for segment in segments)
        transcribe_time = time.time() - start_time
        print(f"Transcription completed in {transcribe_time:.2f} seconds")

        return result, transcribe_time


def transcribe(model_name: str, device: str, audio_path: str, language: str = "ja"):
    """単一モデルでの文字起こしを実行する関数"""
    compute_type = "float16" if device == "cuda" else "int8"
    transcriber = FasterWhisperTranscriber(
        model_name=model_name,
        device=device,
        compute_type=compute_type
    )
    result, transcribe_time = transcriber.transcribe(
        audio_path=audio_path,
        language=language,
    )
    print(f"モデル: {model_name}, 所要時間:{transcribe_time:.1f}秒, 文字起こし結果:{result}")
    transcriber.unload_model()
    return result, transcribe_time


def main():
    """メイン関数: 単一モデルで文字起こしを実行し結果を保存"""
    args = get_args()

    # モデル名が指定されていない場合、デフォルトのモデルを使用
    model_name = args.model_name if hasattr(args, 'model_name') and args.model_name else "large-v3"

    result, time = transcribe(model_name, device=decide_device(), audio_path=args.audio_path)
    append_result(args.result_file, args.audio_path, "Faster Whisper", model_name, time, result)


if __name__ == "__main__":
    main()