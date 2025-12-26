import gc
import torch
import csv
import os
import argparse

class BaseTranscriber:
    """音声文字起こしモデルの基底クラス"""

    def unload_model(self):
        """モデルをアンロードしてメモリを解放"""
        if hasattr(self, 'model') and self.model is not None:
            print(f"Unloading model...")

            # GPUメモリからモデルを削除
            if hasattr(self.model, 'to'):
                self.model.to('cpu')

            del self.model
            self.model = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared")

            print("Model unloaded successfully")
        else:
            print("No model loaded to unload")

def append_result(csv_path, audio_path, oss_name, model_name, time, transcription):
    """CSVファイルに文字起こし結果を追記"""
    # ファイルが存在するかどうかを先にチェック
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # ファイルが存在しない場合のみヘッダーを書き込む
        if not file_exists:
            writer.writerow(["AUDIO_FILE", "OSS", "MODEL", "PROCESS_TIME", "TRANSCRIPTION"])

        # データを書き込む
        writer.writerow([audio_path, oss_name, model_name, f"{time:.2f}", transcription])

def decide_device():
    """GPUが利用可能かどうかに基づいてデバイスを決定"""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return  "cuda"
    else:
        print("No GPU available. Using CPU.")
        return "cpu"

def decide_torch_dtype():
    """GPUが利用可能かどうかに基づいてデータ型を決定"""
    return torch.float16 if torch.cuda.is_available() else torch.float32

def get_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio-path", required=True,
                       help="Path to the audio file to transcribe")
    parser.add_argument("-r", "--result-file", default="result.csv",
                       help="Path to the result CSV file (default: result.csv)")
    parser.add_argument("-m", "--model-name",
                       help="Name of the model to use (e.g., large-v3, large-v2, etc.)")
    return parser.parse_args()