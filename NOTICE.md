# Licence notice

**The code** in this repository is released under the [MIT licence](LICENSE), the same licence as
[Mini-Omni2](https://github.com/gpt-omni/mini-omni2) (MIT, copyright gpt-omni), which this project fine-tunes.

**What you build with it** also depends on the models and data you use. In particular:

* **Voices cloned or designed with OmniVoice: non-commercial only.** OmniVoice's code is Apache-2.0, but its model
  weights are licensed **CC-BY-NC 4.0** ("due to constraints from its training data"). If you generate the
  assistant's voice with OmniVoice (`finetune/omnivoice_data.py`), treat the generated audio, and any model
  fine-tuned on it, as **non-commercial**: personal, research or demonstration use. The assistant this project was
  built for uses an OmniVoice voice and is non-commercial.
* **Without OmniVoice** (for example, every voice from Kokoro, Apache-2.0), nothing above restricts commercial use;
  the remaining components are MIT or Apache-2.0:

| component | licence |
|---|---|
| Mini-Omni2 code and weights | MIT |
| Qwen2-0.5B (Mini-Omni2's language model) | Apache-2.0 |
| Whisper, faster-whisper, SNAC, multilingual-e5-base | MIT |
| VoiceAssistant-400K (off-topic questions) | Apache-2.0 |
| Kokoro-82M / kokoro-onnx | Apache-2.0 / MIT |
| OmniVoice weights | **CC-BY-NC 4.0** |

**Voice cloning needs consent.** Clone only your own voice, or the voice of someone who has explicitly agreed. A
first-person assistant must tell visitors it is an AI (see the README).

This notice is a summary, not legal advice: check each component's licence for your use.
