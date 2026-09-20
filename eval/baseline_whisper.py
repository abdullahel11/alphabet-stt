"""
Day 2 baseline: run Whisper over my recorded letters and see what it hears.

This is deliberately a BASELINE, not the final service. The brief predicts
Whisper will struggle on single isolated letters because it is built for
words and sentences. The point of this script is to find out whether that
is true for my recordings, with numbers rather than an assumption.

Run it with:   modal run eval/baseline_whisper.py

"modal run" is different from "modal deploy":
  - deploy = put a permanent web endpoint on the internet
  - run    = run this code once in the cloud, print the result, shut down
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import modal

app = modal.App("alphabet-stt-baseline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("transformers==4.44.2", "torch", "numpy")
)

MODEL_NAME = "openai/whisper-small"


def _download_model():
    """Runs once, at image build time, so the ~1GB model is baked into the
    image instead of being downloaded on every request. This is the first
    half of my answer to the cold-start problem in the brief."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    WhisperProcessor.from_pretrained(MODEL_NAME)
    WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)


image = image.run_function(_download_model)


EXPECTED = {
    "baa.m4a": ("ب", "baa"),
    "meem.m4a": ("م", "meem"),
    "seen.m4a": ("س", "seen"),
    "saad.m4a": ("ص", "saad"),
    "ayn.m4a": ("ع", "ayn"),
}


def decode_audio(audio_bytes: bytes):
    """Turn recorded bytes into mono 16kHz float samples.

    The first version of this piped bytes into ffmpeg on stdin, which
    silently returned little or no audio: .m4a is an MP4 container and
    ffmpeg must seek to an index at the END of the file, which a pipe
    does not allow. Whisper pads everything to 30 seconds, so it never
    errored -- it just hallucinated on silence, and I nearly wrote that
    up as a real result about Whisper. Writing to a real file fixes the
    decode; the length check makes any future failure loud.
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


@app.cls(image=image, cpu=2)
class Baseline:
    """A class, not a plain function, so the model is loaded ONCE per
    container and reused for every clip instead of being reloaded each time."""

    @modal.enter()
    def load(self):
        from transformers import pipeline

        started = time.time()
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            device=-1,  # -1 means CPU. Checking whether CPU is enough before
                        # paying for a GPU is one of the brief's questions.
        )
        print(f"model loaded in {time.time() - started:.1f}s")

    @modal.method()
    def transcribe(self, audio_bytes: bytes) -> dict:
        audio = decode_audio(audio_bytes)

        started = time.time()
        result = self.pipe(
            {"raw": audio, "sampling_rate": 16000},
            generate_kwargs={
                "language": "arabic",
                "task": "transcribe",
                # Without these, Whisper loops on very short clips and spends
                # 40+ seconds emitting the same token over and over.
                "max_new_tokens": 12,
                "no_repeat_ngram_size": 3,
            },
        )
        return {
            "text": result["text"].strip(),
            "latency_ms": round((time.time() - started) * 1000),
            "duration_s": round(len(audio) / 16000, 2),
        }


@app.local_entrypoint()
def main():
    """Runs on MY laptop. It reads the audio files locally and hands the raw
    bytes to the cloud function above."""
    audio_dir = Path(__file__).parent / "audio"
    files = sorted(p for p in audio_dir.glob("*.m4a"))

    if not files:
        print(f"No .m4a files found in {audio_dir}")
        return

    baseline = Baseline()
    rows = []
    for path in files:
        expected_letter, expected_name = EXPECTED.get(path.name, ("?", "?"))
        out = baseline.transcribe.remote(path.read_bytes())
        rows.append((path.name, expected_letter, expected_name, out))

    print()
    print(f"{'file':<12} {'secs':>5} {'expected':<10} {'whisper heard':<30} {'ms':>6}")
    print("-" * 68)
    for name, letter, expected_name, out in rows:
        heard = out["text"] if out["text"] else "(nothing)"
        print(f"{name:<12} {out['duration_s']:>5} {expected_name:<10} {heard:<30} "
              f"{out['latency_ms']:>6}")

    print()
    print("Check the secs column first: my clips should be roughly 1-2 seconds.")
    print("Then: does Whisper return the letter, something else, or nothing?")
