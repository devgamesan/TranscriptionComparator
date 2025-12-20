# openai whisper
source .venv_whisper/bin/activate
python ./src/transcriber_whipser.py  --audio-path inputs/input.mp4 --result-file outputs/result.csv

# faster whisper
source .venv_fasterwhisper/bin/activate
python ./src/transcriber_fasterwhisper.py --audio-path inputs/input.mp4 --result-file outputs/result.csv

# ReazonSpeech
source .venv_reazonspeech/bin/activate
python ./src/transcriber_reazonspeech.py --audio-path inputs/input.mp4 --result-file outputs/result.csv

# funasr
source .venv_funasr/bin/activate
python ./src/transcriber_funasr.py --audio-path inputs/input.mp4 --result-file outputs/result.csv

