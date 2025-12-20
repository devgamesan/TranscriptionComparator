FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Install system dependencies
RUN apt update && \
    apt install -y git ffmpeg curl python3-dev build-essential cmake pkg-config

# Switch to non-root user
USER ubuntu
WORKDIR /home/ubuntu/app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# --- Copy only what's needed for dependency installation ---
COPY --chown=ubuntu:ubuntu requirements/ ./requirements/

# Clone ReazonSpeech at specific commit (no local source needed yet)
RUN git clone https://github.com/reazon-research/ReazonSpeech && \
    cd ReazonSpeech && \
    git checkout 9d80a30af1b5f456817db901a28ae462731b8157

# Create venvs and install dependencies
RUN uv venv .venv_whisper && \
    /bin/bash -c "source .venv_whisper/bin/activate && \
    UV_HTTP_TIMEOUT=300 uv pip install -r requirements/requirements_openaiwhipser.txt"

RUN uv venv .venv_fasterwhisper && \
    /bin/bash -c "source .venv_fasterwhisper/bin/activate && \
    UV_HTTP_TIMEOUT=300 uv pip install -r requirements/requirements_fasterwhisper.txt"

RUN uv venv .venv_reazonspeech && \
    /bin/bash -c "source .venv_reazonspeech/bin/activate && \
    UV_HTTP_TIMEOUT=300 uv pip install ReazonSpeech/pkg/nemo-asr && \
    UV_HTTP_TIMEOUT=300 uv pip install ReazonSpeech/pkg/k2-asr && \
    UV_HTTP_TIMEOUT=300 uv pip install ReazonSpeech/pkg/espnet-asr && \
    UV_HTTP_TIMEOUT=300 uv pip install -r requirements/requirements_reazonspeech.txt"

RUN uv venv .venv_funasr && \
    /bin/bash -c "source .venv_funasr/bin/activate && \
    UV_HTTP_TIMEOUT=300 uv pip install -r requirements/requirements_funasr.txt"

# Apply FunASR patch
# https://github.com/modelscope/FunASR/issues/2741
RUN /bin/bash -c "\
    sed -i '12s/.*//' .venv_funasr/lib/python3.12/site-packages/funasr/models/fun_asr_nano/model.py && \
    sed -i '44i\\            from funasr import AutoModel' .venv_funasr/lib/python3.12/site-packages/funasr/models/fun_asr_nano/model.py"

# --- Finally, copy the rest of the application code ---
COPY --chown=ubuntu:ubuntu . .