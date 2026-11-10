# Voice twin

A small voice assistant that answers spoken questions about one person's career, for their website. It is a
fine-tuned [Mini-Omni2](https://github.com/gpt-omni/mini-omni2) (a half-billion-parameter speech-to-speech model)
plus a retrieval step, served from your own computer as an API. It can speak about you in the third person, or as
your "digital twin" in the first person, in a cloned voice.

The point it makes: for a focused job like "answer questions about my CV", a small specialised model running on a
home PC is enough. No large language model, no paid API.

* `finetune/`: turns a Q&A file about a person into training data (spoken questions, spoken answers, retrieved
  context) and fine-tunes Mini-Omni2 on it, locally or on Colab.
* `assistant/`: the server. A WebSocket API (Docker image included) that listens, answers and speaks, with a
  push-to-talk test page.
* `model/omni_lm.py`: Mini-Omni2's language model in plain PyTorch, used by both.
* `docs/HOW_IT_WORKS.md`: how retrieval and the answer checks work, with a worked example.
* `docs/WEBSITE_INTEGRATION.md`: the API protocol and a hands-free (always listening) web client.

To run Mini-Omni2 (original or fine-tuned) **in the browser** instead, see
[mini-omni2-web](https://github.com/jmsalash/mini-omni2-web), the sister project this one came from.

---

## Contents

* [How it works](#how-it-works)
* [Clone only your own voice](#clone-only-your-own-voice)
* [1. Write the Q&A](#1-write-the-qa)
* [2. Build the training data](#2-build-the-training-data)
* [3. Train](#3-train)
* [4. Package and serve](#4-package-and-serve)
* [Testing](#testing)
* [Limitations](#limitations)
* [Credits and licences](#credits-and-licences)

---

## How it works

```
question (speech) ─▶ Whisper transcript ─▶ retriever: 2 closest answers ─┐
       │                                                                  ▼
       └──────────────▶ Mini-Omni2 (fine-tuned) reads the 2 answers + the spoken question
                                  │ its first words say which answer it gives (or "my responses are limited")
                                  ▼
                checks ─▶ that answer's recording, in the assistant's voice ─▶ visitor
```

* **Fine-tuning puts the person into the weights.** Mini-Omni2 has no system prompt: given a CV as text, it picks
  up names and invents the rest. So every answer is trained in, from many spoken phrasings of each question.
* **Retrieval keeps it on track.** A small embedding model (multilingual-e5-base, fine-tuned on the question wordings) finds the
  two answers closest to the question, and they go in front of it, as in training. The model then mostly has to
  choose, which a 0.5B model does far more reliably than recalling.
* **Recorded answers.** Every answer is recorded once (with a Whisper check) during data preparation. The model's
  first words identify the answer it chose, and the server plays that recording: the voice never garbles or drifts
  into the other retrieved answer. First audio arrives in about 0.8 s on a GTX 1070.
* **Safety rules** (`assistant/rag.py`): an answer must be one of the retrieved ones or a fallback line; questions
  that match nothing well only get a fallback; a clear question the model refuses gets the retrieved answer's
  recording. The details and a worked example are in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).
* **Chatty visitors.** Before the search, leading acknowledgements and summaries ("That's really impressive. So...")
  are dropped and a multi-part question is searched by its main part; a bare acknowledgement ("Of course.", "Got
  it.") gets no reply.

Measured on the author's assistant (86 topics, 111 held-out spoken questions, GTX 1070): with the default
transcript, 77 of 86 career questions got the right answer (first audio 0.8 s median); with `ASR_MODEL` set to
large-v3-turbo, 80 of 86 (1.15 s). The rest got a fallback line or a neighbouring true answer, 22 of 25 off-topic
questions got the fallback, and no answer invented a fact.

---

## Clone only your own voice

The first-person mode speaks in a cloned voice. **Clone only your own voice, or one whose owner has explicitly
agreed.** A first-person assistant must also say it is an AI: the server plays a recorded disclaimer when a
conversation starts, the Q&A should have a "Am I talking to the real person?" topic, and the page must show a
visible "AI voice" label ([docs/WEBSITE_INTEGRATION.md](docs/WEBSITE_INTEGRATION.md#5-what-the-page-must-show)).
The assistant must never claim to be the person.

---

## 1. Write the Q&A

Copy `finetune/data/cv_qa.example.json` to `finetune/data/cv_qa.json` (`finetune/data/` is git-ignored, so personal
data stays out of the repository) and fill it in: topics, each with a few phrasings of the question and **one
spoken answer**.

* **Answers are spoken**: 15-60 words, plain words, no URLs or symbols. Write names as they should sound, or use
  respellings at recording time (`--pronounce Siobhan=Shivawn`).
* **Cover an interview**, not just the CV: per company and role, "tell me about yourself", strengths and weaknesses,
  why this career, places lived, what you are looking for, small talk (hello, thanks, goodbye), the assistant itself,
  and "I don't know" topics for what the CV doesn't say (age, salary, family).
* **Whatever is trained in can be said to anyone.** Leave out phone numbers, email addresses and anything private.
* **Wordings.** Every topic needs 15-20 ways of asking. `finetune/paraphrase_questions.py` writes them with a local
  LLM (Ollama) and has the prompt in it, so any LLM can be used; a strong model writes noticeably better wordings.

---

## 2. Build the training data

Requirements: a CUDA GPU (8 GB is enough for data preparation; a GTX 1070 works), Python 3.11, and:

```bash
pip install -r finetune/requirements.txt      # after CUDA PyTorch from pytorch.org
hf download gpt-omni/mini-omni2 lit_model.pth small.pt model_config.yaml tokenizer.json tokenizer_config.json \
    --local-dir models/mini-omni2
# Kokoro voices for the questions
mkdir -p models/kokoro && (cd models/kokoro && \
  wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
       https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin)
export MINI_OMNI2_DIR=$PWD/models/mini-omni2 KOKORO_DIR=$PWD/models/kokoro FT_WORK=$PWD/finetune/work
```

Then, from `finetune/`:

| step | what | time (GTX 1070) |
|---|---|---|
| `01_cv_audio.py [--extra-questions wordings.json]` | every wording spoken by several Kokoro voices | ~20 min on GPU |
| `02_general.py --n 1500 --offtopic` | off-topic questions from Mini-Omni2's own data, answered with a fallback line | ~15 min |
| `omnivoice_data.py --ref ref.wav --speed 0.9 --pronounce Siobhan=Shivawn` | answers in the assistant's voice (cloned from `ref.wav` + `ref.txt`), more question voices, each clip checked by Whisper | ~1-2 h |
| `03_encode.py --manifests work/cv/manifest.ov.jsonl work/general/manifest.ov.jsonl` | Whisper features, SNAC codes, token sequences | ~15 min |
| `train_retriever.py --manifest work/cv/manifest.ov.jsonl --out work/retriever` | fine-tune the retriever on the wordings | ~1 min |
| `rag_context.py --manifests ... --retriever work/retriever --patch-encoded work/encoded` | put the 2 retrieved answers in front of every training question | ~1 min |

OmniVoice runs in its own environment (`pip install omnivoice`). Its weights are CC-BY-NC: **a voice made with it,
and a model trained on that voice, are non-commercial** ([NOTICE.md](NOTICE.md)). Kokoro voices have no such limit.
The off-topic answers: the fallback lines, e.g. "I'm sorry, my responses are limited. You must ask the right
questions. For example, ask me what I did at my last company." Several lines, so it doesn't repeat itself.

---

## 3. Train

Training needs a GPU with 16 GB or more. On Colab: `finetune/make_colab_bundle.sh` packs the code and
`work/encoded` into one zip (~1.2 GB); upload it to a Google Drive folder `omni-cv` and run
[`finetune/colab_train.ipynb`](finetune/colab_train.ipynb). The notebook picks precision and batch from the GPU
(A100, H100, G4, L4 or T4), checks the zip, and resumes after a browser disconnect.

```bash
python 04_train.py --amp bf16 --batch 8 --lr 1e-5 --epochs 6 --cv-weight 1 --typed \
    --val-frac 0.05 --val-every 200 --patience 8 --save-last --out work/ckpt
```

* Full fine-tune of the language model; the Whisper encoder and adapter stay frozen.
* **Early stopping** on 5% held-out question wordings: the best checkpoint is kept, and with `--save-last` the
  final one too (`work/ckpt/last`).
* Batch 8 at lr 1e-5 worked; batch 32 undertrained the voice.
* `--typed` also trains typed questions, so the model can answer text input.

---

## 4. Package and serve

```bash
python finetune/package_assistant.py --ckpt work/ckpt --out /path/to/my-assistant \
    --manifest work/cv/manifest.ov.jsonl --general work/general/manifest.ov.jsonl \
    --retriever work/retriever --clips work/cv/answers_ov --fixed-lines work/voice
```

Optional `finetune/data/timeline.json` lists topics in time order (`{"sequences": [[first_job, second_job, ...],
[first_degree, ...]]}`) so that "and before that?" works. `work/voice/` holds the fixed lines in the assistant's voice: `disclaimer.wav/.txt` (first person) and
`rephrase.wav/.txt`, `rephrase_1.wav/.txt`, ... (said in turn when the assistant isn't sure).

Run the server with Docker (`assistant/docker-compose.yml`, settings in `assistant/.env`, see `.env.example`) or
directly:

```bash
MODEL_DIR=/path/to/my-assistant WHISPER_PATH=models/mini-omni2/small.pt python assistant/server.py
```

Then open `http://localhost:8765/` for the test page: it listens all the time (a simple loudness detector finds where
each question ends) and has a Mute button. Settings that matter:

| setting | |
|---|---|
| `ALLOWED_ORIGINS` | your website's origin; empty = any (testing only) |
| `MAX_SESSIONS`, `SLOTS` | open conversations / answers generated together on the GPU |
| `ASR_MODEL` | optional faster-whisper folder (e.g. large-v3-turbo, 8-bit) for the transcript: hears short questions better, about +0.35 s |
| `RECORDED_ANSWERS` | 1 (default): play the recorded answer the model chose; 0: stream the model's own generated voice |
| `RAG_MIN_SCORE`, `RAG_RESCUE_SCORE` | retrieval trust thresholds (both 0.85) |
| `LOG_QUESTIONS` | 1 logs what was heard and answered, and why (testing; visitors' questions, so off by default) |
| `SSL_CERTFILE`, `SSL_KEYFILE` | HTTPS directly, for testing on a LAN (the microphone needs HTTPS) |

For the website side (protocol, hands-free client with voice activity detection, what the page must show), see
[docs/WEBSITE_INTEGRATION.md](docs/WEBSITE_INTEGRATION.md).

---

## Testing

* `assistant/test_questions.py questions.json out.jsonl` asks the server a list of recorded questions and saves
  what was heard and answered. Use the held-out wordings (`split: eval` in the manifests).
* With `LOG_QUESTIONS=1`, the server log shows for each question what was heard, the retrieved topics with their
  scores, the answer and, for fallbacks, why. Real conversations are the best test data: misheard or unmatched
  questions can be added as wordings in `finetune/data/retrieval_extra.json` (no retraining), and the rest go into
  the next training round.

---

## Limitations

* **English only.**
* **Simple follow-ups only.** The server remembers the previous question of each conversation: "and before
  that?" / "what came next?" step through `timeline.json`, and "what did you do there?", "why?" or "tell me more"
  are searched together with the previous question. But each topic has one answer, so "tell me more" repeats it:
  deeper follow-ups need second-level topics in the Q&A.
* **Fixed answers**: it says what was written and recorded, nothing else. That is the point (no invented facts),
  but it means the Q&A has to cover what visitors ask.
* **Short questions** are the hardest to hear (e.g. "Why AI?"); `ASR_MODEL` helps.

---

## Credits and licences

The model is **Mini-Omni2** by Zhifei Xie and Changqiao Wu (gpt-omni); this project fine-tunes and serves it. The
training pipeline, retrieval, server and data tools are this project's own.

* Z. Xie, C. Wu. *Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities.*
  [arXiv:2410.11190](https://arxiv.org/abs/2410.11190), 2024.
* Code and weights: [github.com/gpt-omni/mini-omni2](https://github.com/gpt-omni/mini-omni2) (MIT).

| component | licence |
|---|---|
| [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) (Mini-Omni2's language model) | Apache-2.0 |
| [Whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT |
| [SNAC](https://github.com/hubertsiuzdak/snac) (audio codec) | MIT |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) (retriever) | MIT |
| [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) (off-topic questions) | Apache-2.0 |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) / [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (question voices) | Apache-2.0 / MIT |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (answer voice, optional) | weights CC-BY-NC 4.0: non-commercial |

This project's own code is released under the [MIT licence](LICENSE), the same as Mini-Omni2. **If you clone the
voice with OmniVoice, the result is non-commercial**: OmniVoice's weights are CC-BY-NC, so treat the generated audio
and any model trained on it as non-commercial. See [NOTICE.md](NOTICE.md).
