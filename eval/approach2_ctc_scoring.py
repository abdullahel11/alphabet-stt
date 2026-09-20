"""
Approach 2: force the model to choose one of the 28 letters.

Why this is different from the Whisper baseline
-----------------------------------------------
Whisper is an encoder-DECODER model: it generates text token by token, guided
by a language model. wav2vec2-CTC has no language model and no generation
step. It emits a probability distribution over characters for each ~20ms
frame of audio.

That means I can do something Whisper cannot: take the 28 letter names, ask
"how likely is THIS audio if the speaker said THIS name?", and rank all 28.
The model is never free to invent anything. It must commit.

The score is the CTC loss of each candidate name given the audio, which is
exactly "negative log-likelihood of this label sequence". Lower loss = better
match. I divide by the length of the name so that longer names are not
penalised simply for being longer.

Run it with:   modal run eval/approach2_ctc_scoring.py
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import modal

app = modal.App("alphabet-stt-ctc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("transformers==4.44.2", "torch", "numpy")
)

# Arabic wav2vec2, character-level CTC. Chosen because it is a CTC model
# (scorable) and it is fine-tuned on Arabic, so the emphatic consonants
# ص ض ط ظ are in its training distribution.
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"


def _download_model():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)


image = image.run_function(_download_model)


# All 28 letters and how their NAMES are spelled in Arabic. I decided to
# accept the letter name, so these spellings are what I score against.
# The candidate set is always all 28, even though I currently only have
# 5 recordings -- picking 1 of 5 would be a much easier problem than the
# real one and would flatter the numbers.
LETTERS = [
    ("ا", "ألف", "alif"),
    ("ب", "باء", "baa"),
    ("ت", "تاء", "taa"),
    ("ث", "ثاء", "thaa"),
    ("ج", "جيم", "jeem"),
    ("ح", "حاء", "Haa"),
    ("خ", "خاء", "khaa"),
    ("د", "دال", "daal"),
    ("ذ", "ذال", "dhaal"),
    ("ر", "راء", "raa"),
    ("ز", "زاي", "zay"),
    ("س", "سين", "seen"),
    ("ش", "شين", "sheen"),
    ("ص", "صاد", "saad"),
    ("ض", "ضاد", "Daad"),
    ("ط", "طاء", "Taa"),
    ("ظ", "ظاء", "DHaa"),
    ("ع", "عين", "ayn"),
    ("غ", "غين", "ghayn"),
    ("ف", "فاء", "faa"),
    ("ق", "قاف", "qaaf"),
    ("ك", "كاف", "kaaf"),
    ("ل", "لام", "laam"),
    ("م", "ميم", "meem"),
    ("ن", "نون", "noon"),
    ("ه", "هاء", "haa"),
    ("و", "واو", "waw"),
    ("ي", "ياء", "yaa"),
]

EXPECTED = {
    "baa.m4a": "baa",
    "meem.m4a": "meem",
    "seen.m4a": "seen",
    "saad.m4a": "saad",
    "ayn.m4a": "ayn",
}


def decode_audio(audio_bytes: bytes):
    """Turn recorded bytes into mono 16kHz float samples.

    NOTE: an earlier version of this piped the bytes into ffmpeg on stdin.
    That silently returned little or no audio, because .m4a is an MP4
    container and ffmpeg has to seek to an index stored at the END of the
    file, which it cannot do on a pipe. Whisper pads everything to 30
    seconds, so it never complained -- it just hallucinated on silence and
    I nearly recorded that as a real result. Writing to a real file first
    fixes it, and the length check below makes the failure loud instead
    of silent.
    """
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-i", tmp_path,
             "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            capture_output=True,
        )
    finally:
        os.unlink(tmp_path)

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[-300:]}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size < 1600:  # under 0.1 seconds
        raise RuntimeError(f"only {audio.size} samples decoded - the audio did not decode")
    return audio


@app.cls(image=image, cpu=2)  # same CPU count as the Whisper baseline, so the
class Scorer:                 # latency comparison between the two is fair
    @modal.enter()
    def load(self):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        started = time.time()
        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        self.model.eval()

        # Pre-tokenise the 28 candidate names once, not per request.
        self.candidates = []
        for letter, name_ar, name_en in LETTERS:
            ids = self.processor.tokenizer(name_ar).input_ids
            self.candidates.append((letter, name_ar, name_en, ids))

        self.blank_id = self.processor.tokenizer.pad_token_id
        print(f"model loaded in {time.time() - started:.1f}s")

    @modal.method()
    def score(self, audio_bytes: bytes) -> dict:
        torch = self.torch
        F = torch.nn.functional

        started = time.time()
        audio = decode_audio(audio_bytes)

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits  # (1, frames, vocab)

        # What the model hears with no constraints at all. A diagnostic only --
        # it is NOT how the verdict is decided.
        greedy = self.processor.batch_decode(torch.argmax(logits, dim=-1))[0]

        # (frames, batch, vocab) is the layout ctc_loss expects.
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        input_lengths = torch.tensor([log_probs.shape[0]])

        scored = []
        for letter, name_ar, name_en, ids in self.candidates:
            loss = F.ctc_loss(
                log_probs,
                torch.tensor([ids], dtype=torch.long),
                input_lengths,
                torch.tensor([len(ids)]),
                blank=self.blank_id,
                reduction="sum",
                zero_infinity=True,
            )
            # Negative loss = log-likelihood. Divided by name length so a
            # 3-character name is not automatically preferred over a 4.
            scored.append((letter, name_en, -loss.item() / len(ids)))

        scored.sort(key=lambda row: row[2], reverse=True)

        # Turn the 28 scores into something that behaves like a probability.
        # A first pass only; calibrating this properly is a Day 4 job, once
        # I know what the score gap looks like on real errors.
        raw = torch.tensor([row[2] for row in scored])
        probs = torch.softmax(raw, dim=0).tolist()

        return {
            "greedy": greedy.strip(),
            "top": [
                {"letter": letter, "name": name, "score": round(score, 3), "conf": round(p, 3)}
                for (letter, name, score), p in zip(scored[:3], probs[:3])
            ],
            "latency_ms": round((time.time() - started) * 1000),
            "duration_s": round(len(audio) / 16000, 2),
        }


@app.local_entrypoint()
def main():
    audio_dir = Path(__file__).parent / "audio"
    files = sorted(p for p in audio_dir.glob("*.m4a"))
    if not files:
        print(f"No .m4a files found in {audio_dir}")
        return

    scorer = Scorer()
    results = [(p.name, scorer.score.remote(p.read_bytes())) for p in files]

    correct = 0
    print()
    print(f"{'file':<11} {'secs':>5} {'expected':<9} {'top-1':<9} {'conf':>6} {'':<3} "
          f"{'runners-up':<26} {'ms':>6}")
    print("-" * 84)
    for name, out in results:
        expected = EXPECTED.get(name, "?")
        top = out["top"][0]
        hit = top["name"] == expected
        correct += hit
        runners = ", ".join(f"{c['name']} {c['conf']:.2f}" for c in out["top"][1:])
        mark = "OK" if hit else "X"
        print(
            f"{name:<11} {out['duration_s']:>5} {expected:<9} {top['name']:<9} "
            f"{top['conf']:>6.2f} {mark:<3} {runners:<26} {out['latency_ms']:>6}"
        )

    print()
    print(f"Accuracy: {correct}/{len(results)}  (chosen from all 28 letters)")
    print()
    print("Check the secs column first. My clips should be roughly 1-2 seconds.")
    print("Anything near zero means the audio did not decode and the run is void.")
    print()
    print("Unconstrained greedy CTC output, for comparison with Whisper:")
    for name, out in results:
        print(f"  {name:<11} {out['greedy'] or '(nothing)'}")
