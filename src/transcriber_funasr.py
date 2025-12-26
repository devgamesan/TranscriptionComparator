from funasr import AutoModel
import os
import time
from typing import Dict, Any
from base import BaseTranscriber, append_result, decide_device, get_args

class FunASRTranscriber(BaseTranscriber):
    """FunASRを使用した音声認識トランスクリバークラス"""
    def __init__(
        self,
        model_dir: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        vad_model: str = "fsmn-vad",
        max_single_segment_time: int = 30000,
        device: str = "cuda"
    ):
        """FunASRトランスクリバーの初期化"""
        self.model_dir = model_dir
        self.vad_model = vad_model
        self.max_single_segment_time = max_single_segment_time
        self.device = device

        print(f"Loading FunASR model from {self.model_dir} on {self.device}...")
        start_time = time.time()
        # FunASRモデルのロード
        self.model = AutoModel(
            model=self.model_dir,
            vad_model=self.vad_model,
            vad_kwargs={"max_single_segment_time": self.max_single_segment_time},
            device=self.device,
        )
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """音声ファイルをテキストに変換"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing with FunASR: {audio_path}")

        start_time = time.time()
        # 音声認識の実行
        res = self.model.generate(input=[audio_path], cache={}, batch_size_s=0)
        text = res[0]["text"]
        transcribe_time = time.time() - start_time

        result = {"text": text}
        print(f"Transcription completed in {transcribe_time:.2f} seconds")
        return result, transcribe_time

def transcribe(
    model_dir: str,
    device: str,
    audio_path: str,
    vad_model: str = "fsmn-vad",
    max_single_segment_time: int = 30000
):
    """FunASRを使用して音声をテキストに変換する関数"""
    transcriber = FunASRTranscriber(
        model_dir=model_dir,
        vad_model=vad_model,
        max_single_segment_time=max_single_segment_time,
        device=device
    )
    result, transcribe_time = transcriber.transcribe(audio_path=audio_path)
    transcriber.unload_model()
    return result["text"], transcribe_time


def main():
    """メイン関数：コマンドライン引数から音声認識を実行"""
    args = get_args()

    # モデル名が指定されていない場合、デフォルトのモデルを使用
    model_dir = args.model_name if hasattr(args, 'model_name') and args.model_name else "FunAudioLLM/Fun-ASR-Nano-2512"

    result, time = transcribe(model_dir, device=decide_device(), audio_path=args.audio_path)
    append_result(args.result_file, args.audio_path, "FunASR", model_dir, time, result)

if __name__ == "__main__":
    main()