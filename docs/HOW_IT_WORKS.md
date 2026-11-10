# How the voice assistant works

This explains, in plain language, how a visitor's spoken question becomes a spoken answer, and why each piece is
there. The examples use a fictional person, Ada, a data engineer who has worked in Lisbon and Berlin.

## The problem

We want a small model (Mini-Omni2, about half a billion parameters) to answer spoken questions about one person,
in a natural voice, without inventing anything.

Mini-Omni2 is a *speech-to-speech* model: it hears audio and answers with audio directly. It is small enough to run
on a home GPU, but out of the box it knows nothing about Ada, and if you give it Ada's CV as text it picks up names
and then makes the rest up. So two things have to happen:

1. **Teach it** (fine-tuning): the answers about Ada, the voice, and when to politely decline.
2. **Help it at question time** (retrieval, the "R" in RAG): look up the right answers and put them in front of the
   question, so the model reads them instead of having to remember everything.

## The pieces

```
 visitor speaks
      |
      v
 [1] Whisper transcribes      "What did you do in Berlin?"
      |
      v
 [2] Retriever looks it up    -> topics: berlin_job (0.91), lisbon_job (0.62)
      |
      v
 [3] The two answers are put in front of the question
      |
      v
 [4] Mini-Omni2 answers       hears the question, reads the two answers, speaks the right one
      |
      v
 [5] Safety checks            does it agree with itself? is it one of the retrieved answers?
      |
      v
 visitor hears the answer (or "Sorry, could you ask it another way?")
```

### 1. Whisper: what did they say?

Whisper (the same speech encoder Mini-Omni2 uses to hear) also writes the question down as text. That text is only
used for the lookup in step 2; the model itself still hears the audio. A short list of names ("Berlin, Lisbon,
DataWorks GmbH...") is given to Whisper as a hint, otherwise it writes unusual names wrongly ("Data Works Gum").

### 2. The retriever: which answers are relevant?

The Q&A file has, for every topic, one curated answer and many ways a visitor might ask for it ("wordings"):

| topic | answer | some wordings |
|---|---|---|
| `berlin_job` | "From 2019 to 2023 I was a data engineer at DataWorks in Berlin, building..." | "What did you do in Berlin?", "Tell me about DataWorks.", "So, your job in Germany?" |
| `lisbon_job` | "From 2015 to 2019 I worked at..." | "What did you do in Lisbon?", ... |

A small *embedding model* turns every sentence into a list of numbers such that sentences that mean the same thing
get similar numbers. The retriever compares the visitor's question with all the wordings and returns the topics of
the closest ones, with a score. Two lessons from building this:

* **Match against example questions, not against the answers.** A question looks much more like another question
  than like a paragraph of CV. On new questions, matching against the answers put the right one first 48% of the time;
  matching against example wordings, 78%.
* **Fine-tune the embedding model on your wordings.** A few minutes of training (wordings of the same topic pulled
  together, different topics pushed apart) raised the right answer being among the top two from 91% to 97%.

### 3. The context

The two best answers are written in front of the question, as text. To the model, the input looks like:

```
[text]   From 2019 to 2023 I was a data engineer at DataWorks in Berlin...
         From 2015 to 2019 I worked at...
[audio]  (the visitor's question, as sound)
```

### 4. Mini-Omni2 answers

The fine-tuned model reads the two answers, hears the question and speaks the answer that fits, in the trained
voice. It was trained on thousands of examples of exactly this situation, including the hard ones:

| in training | context | the model is taught to say |
|---|---|---|
| most questions | the right answer is first or second | the right answer |
| a few questions | retrieval got it wrong: neither answer fits | still the right answer (don't blindly trust the context) |
| off-topic questions ("how do I bake bread?") | two unrelated answers | the polite redirect |

That last row matters: without it, the model learns to read out whatever is in front of it.

Why train at all, if the answer is in the context? Measured on held-out questions (64 questions never used in
training), the share of wrong answers:

| | wrong |
|---|---|
| original Mini-Omni2, no help | 44% |
| original Mini-Omni2 + retrieved answers | 59% (it mixes and invents) |
| fine-tuned, no retrieval | 16% |
| fine-tuned, retrieval added afterwards | 70% (the unfamiliar context confuses it) |
| **fine-tuned with retrieval in training** | **8%** |
| ... + the fine-tuned retriever + safety checks (through the server) | **0%** (and 9% "please rephrase") |

Retrieval only helps a model that was trained to use it.

### 5. Safety checks

Mini-Omni2 answers with two copies of itself running side by side: one writes the text, the other speaks. Before
any sound reaches the visitor:

* **The two copies must agree** on the first few words. When they disagree, the answer is usually wrong (and the
  voice garbles).
* **The answer must be one of the retrieved answers**, or the polite redirect.
* **The retriever must be confident** (its best topic clearly ahead of the second). If it isn't, only the redirect
  is allowed: off-topic questions match nothing well, so they still get "I can only talk about my work"; ambiguous
  questions get "Could you ask it another way?".
* If the voice stops early while the text goes on, the rest is played from a recording of that answer.

Asking the visitor to rephrase is better than a confident wrong answer.

## What the training data is made of

| part | made with |
|---|---|
| answers | written by the person (and checked against the CV) |
| wordings | written by an LLM (Claude, or any LLM with the prompt in `finetune/`), checked for overlaps between topics |
| the questions as audio | Kokoro (26 voices, varied speeds) and OmniVoice (voices of random gender, age and accent), plus light noise |
| off-topic questions | real recordings from Mini-Omni2's own training set |
| the answers as audio | OmniVoice, in one voice: designed from a description, or cloned from the person's recording |

Spoken *and* typed versions of every question are trained, so the model learns what the questions mean, not only
how they sound.

## Two modes

* **Assistant** (third person): "She worked at DataWorks in Berlin...", in any voice.
* **Digital twin** (first person): "I worked at DataWorks...", in the person's own cloned voice. It must say
  clearly that it is an AI: a spoken disclaimer when the conversation starts, a trained answer to "am I talking to
  the real Ada?", and a visible label on the page. Only clone your own voice, or one whose owner has agreed.

## Limits

* English only.
* It only knows its topics. Anything else gets the redirect or "please rephrase", by design.
* It answers one question at a time: follow-ups like "and what year was that?" need the previous question to be
  understood (work in progress).
* It speaks fixed, checked answers rather than composing new ones. That is what keeps it from inventing things.
