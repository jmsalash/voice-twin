# Voice assistant server: a fine-tuned Mini-Omni2 over a WebSocket (see assistant/README.md).
# Build from the repository root:  docker build -f assistant/Dockerfile -t mini-omni2-assistant .
# The fine-tuned model is NOT in the image: mount its folder at /models/assistant.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf
WORKDIR /app
COPY assistant/requirements.txt /app/assistant/requirements.txt
RUN pip install -r /app/assistant/requirements.txt

# Public weights the server needs, baked in so the container starts offline:
# Whisper small (Mini-Omni2's audio encoder) and the SNAC 24 kHz audio decoder.
RUN python -c "import whisper; whisper._download(whisper._MODELS['small'], '/opt/whisper', False)" \
 && python -c "from snac import SNAC; SNAC.from_pretrained('hubertsiuzdak/snac_24khz')"

COPY model/omni_lm.py /app/model/omni_lm.py
COPY assistant/engine.py assistant/batch.py assistant/server.py assistant/rag.py assistant/retrieval.py /app/assistant/

ENV MODEL_DIR=/models/assistant WHISPER_PATH=/opt/whisper/small.pt HF_HUB_OFFLINE=1 HOST=0.0.0.0 PORT=8765
EXPOSE 8765
# /health only answers once the model is loaded and warmed up
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=4)"
CMD ["python", "/app/assistant/server.py"]
