"""
The evaluation. Runs every clip in the test set through BOTH approaches,
scores them the same way, and reports where each one fails.

What it produces
----------------
1. Overall accuracy for each approach, on the same clips.
2. A confusion list: every letter that was got wrong, and what it was
   mistaken for.
3. The seven confusable pairs from the brief, checked explicitly. An
   overall accuracy number hides exactly this, which is the whole point.
4. Wrong-pronunciation cases: a clip paired with the WRONG expected letter,
   to check the service says "incorrect" rather than "correct".
5. The unclear cases (silence, a cough, a full sentence) and how confident
   each model was, which is the evidence for where to set the threshold.
6. Latency: cold (first call, includes container start and model load) and
   warm (everything after), with median and worst case.

It writes eval/results.json and eval/RESULTS.md so the numbers are
committed rather than living only in a terminal I later close.

Run it with:   modal run eval/evaluate.py
"""

import csv
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import modal

app = modal.App("alphabet-stt-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("transformers==4.44.2", "torch", "numpy")
)

CTC_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
WHISPER_MODEL = "openai/whisper-small"


def _download_models():
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    Wav2Vec2Processor.from_pretrained(CTC_MODEL)
    Wav2Vec2ForCTC.from_pretrained(CTC_MODEL)
    WhisperProcessor.from_pretrained(WHISPER_MODEL)
    WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)


image = image.run_function(_download_models)


# The 28 candidates. This is the decision space: every verdict must be one
# of these, whatever the model would have said if left free.
LETTERS = [
    ("ا", "ألف", "alif"), ("ب", "باء", "baa"), ("ت", "تاء", "taa"),
    ("ث", "ثاء", "thaa"), ("ج", "جيم", "jeem"), ("ح", "حاء", "Haa"),
    ("خ", "خاء", "khaa"), ("د", "دال", "daal"), ("ذ", "ذال", "dhaal"),
    ("ر", "راء", "raa"), ("ز", "زاي", "zay"), ("س", "سين", "seen"),
    ("ش", "شين", "sheen"), ("ص", "صاد", "saad"), ("ض", "ضاد", "Daad"),
    ("ط", "طاء", "Taa"), ("ظ", "ظاء", "DHaa"), ("ع", "عين", "ayn"),
    ("غ", "غين", "ghayn"), ("ف", "فاء", "faa"), ("ق", "قاف", "qaaf"),
    ("ك", "كاف", "kaaf"), ("ل", "لام", "laam"), ("م", "ميم", "meem"),
    ("ن", "نون", "noon"), ("ه", "هاء", "haa"), ("و", "واو", "waw"),
    ("ي", "ياء", "yaa"),
]

# The pairs the brief calls out. These are measured separately because an
# overall accuracy figure hides precisely the errors that matter to a learner.
CONFUSABLE_PAIRS = [
    ("seen", "saad"), ("taa", "Taa"), ("daal", "Daad"),
    ("dhaal", "DHaa"), ("haa", "Haa"), ("kaaf", "qaaf"), ("alif", "ayn"),
]

# Clips deliberately paired with the WRONG expected letter. A wrong answer
# is defined by this pairing, not by a special recording, so no extra audio
# was needed. The right verdict for every row here is "incorrect".
WRONG_PAIRINGS = [
    ("seen.m4a", "saad"), ("saad.m4a", "seen"),
    ("daal.m4a", "Daad"), ("taa.m4a", "Taa"),
    ("haa.m4a", "Haa"), ("kaef.m4a", "qaaf"),
]


def decode_audio(audio_bytes: bytes):
    """Bytes -> mono 16kHz float samples.

    ffmpeg reads from a real file, not a pipe: .m4a is an MP4 container and
    ffmpeg must seek to an index at the end of the file. Piping silently
    returned near-empty audio and invalidated my first day of results.
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
    if audio.size < 1600:
        raise RuntimeError(f"only {audio.size} samples decoded")
    return audio


def normalise_arabic(text: str) -> str:
    """Strip diacritics and collapse letter variants, so that comparing
    Whisper's free text to a candidate name is not defeated by an accent
    mark or a different alif."""
    out = []
    for ch in text:
        code = ord(ch)
        if 0x064B <= code <= 0x0652 or code == 0x0640:  # harakat, tatweel
            continue
        if ch in "أإآٱ":
            ch = "ا"
        elif ch == "ى":
            ch = "ي"
        elif ch == "ة":
            ch = "ه"
        out.append(ch)
    return "".join(out).strip()


def edit_distance(a: str, b: str) -> int:
    """Plain Levenshtein. Used to map Whisper's free text onto the nearest
    of the 28 names, because Whisper cannot be asked the 28-way question
    directly -- which is itself part of the argument against it."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@app.cls(image=image, cpu=2)
class CTCScorer:
    """Approach 2: score the audio against all 28 candidate names."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        started = time.time()
        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(CTC_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(CTC_MODEL)
        self.model.eval()
        self.candidates = [
            (letter, name_en, self.processor.tokenizer(name_ar).input_ids)
            for letter, name_ar, name_en in LETTERS
        ]
        self.blank_id = self.processor.tokenizer.pad_token_id
        self.load_seconds = round(time.time() - started, 2)
        print(f"CTC model loaded in {self.load_seconds}s")

    @modal.method()
    def run(self, audio_bytes: bytes) -> dict:
        torch = self.torch
        F = torch.nn.functional

        started = time.time()
        audio = decode_audio(audio_bytes)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        greedy = self.processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()

        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        input_lengths = torch.tensor([log_probs.shape[0]])

        scored = []
        for letter, name_en, ids in self.candidates:
            loss = F.ctc_loss(
                log_probs,
                torch.tensor([ids], dtype=torch.long),
                input_lengths,
                torch.tensor([len(ids)]),
                blank=self.blank_id,
                reduction="sum",
                zero_infinity=True,
            )
            scored.append((letter, name_en, -loss.item() / len(ids)))

        scored.sort(key=lambda r: r[2], reverse=True)
        probs = torch.softmax(torch.tensor([r[2] for r in scored]), dim=0).tolist()

        return {
            "greedy": greedy,
            "ranking": [
                {"letter": l, "name": n, "score": round(s, 4), "conf": round(p, 4)}
                for (l, n, s), p in zip(scored, probs)
            ][:5],
            "margin": round(scored[0][2] - scored[1][2], 4),
            "latency_ms": round((time.time() - started) * 1000),
            "duration_s": round(len(audio) / 16000, 2),
            "load_seconds": self.load_seconds,
        }


@app.cls(image=image, cpu=2)
class WhisperBaseline:
    """The baseline. Whisper returns free text, so it has to be mapped onto
    the 28 names afterwards -- a step the CTC approach does not need."""

    @modal.enter()
    def load(self):
        from transformers import pipeline

        started = time.time()
        self.pipe = pipeline(
            "automatic-speech-recognition", model=WHISPER_MODEL, device=-1
        )
        self.load_seconds = round(time.time() - started, 2)
        print(f"Whisper model loaded in {self.load_seconds}s")

    @modal.method()
    def run(self, audio_bytes: bytes) -> dict:
        audio = decode_audio(audio_bytes)
        started = time.time()
        result = self.pipe(
            {"raw": audio, "sampling_rate": 16000},
            generate_kwargs={
                "language": "arabic",
                "task": "transcribe",
                "max_new_tokens": 12,
                "no_repeat_ngram_size": 3,
            },
        )
        text = result["text"].strip()
        return {
            "text": text,
            "latency_ms": round((time.time() - started) * 1000),
            "duration_s": round(len(audio) / 16000, 2),
            "load_seconds": self.load_seconds,
        }


def match_whisper_to_letter(text: str):
    """Map Whisper's free text to the nearest of the 28 names."""
    cleaned = normalise_arabic(text)
    if not cleaned:
        return None, 99
    best, best_d = None, 99
    for letter, name_ar, name_en in LETTERS:
        d = edit_distance(cleaned, normalise_arabic(name_ar))
        if d < best_d:
            best, best_d = name_en, d
    return best, best_d


@app.local_entrypoint()
def main():
    eval_dir = Path(__file__).parent
    audio_dir = eval_dir / "audio"

    with open(eval_dir / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    letters = [r for r in rows if r["kind"] == "letter"]
    unclear = [r for r in rows if r["kind"] == "unclear"]
    missing = [r["filename"] for r in rows if not (audio_dir / r["filename"]).exists()]
    if missing:
        print("MISSING FILES, aborting:", missing)
        return

    ctc, whisper = CTCScorer(), WhisperBaseline()
    results = {"ctc": {}, "whisper": {}}

    print(f"Running {len(rows)} clips through both models...\n")
    for r in rows:
        data = (audio_dir / r["filename"]).read_bytes()
        results["ctc"][r["filename"]] = ctc.run.remote(data)
        results["whisper"][r["filename"]] = whisper.run.remote(data)

    # ---------- accuracy ----------
    ctc_hits, wh_hits, confusions, per_clip = 0, 0, [], []
    for r in letters:
        fn, want = r["filename"], r["name_en"]
        c = results["ctc"][fn]
        w = results["whisper"][fn]
        c_got = c["ranking"][0]["name"]
        w_got, w_dist = match_whisper_to_letter(w["text"])

        ctc_hits += c_got == want
        wh_hits += w_got == want
        if c_got != want:
            confusions.append((want, c_got, c["ranking"][0]["conf"], r["mic_processing"]))
        per_clip.append({
            "file": fn, "letter": r["letter"], "expected": want,
            "ctc_heard": c_got, "ctc_conf": c["ranking"][0]["conf"],
            "ctc_margin": c["margin"], "ctc_greedy": c["greedy"],
            "whisper_text": w["text"], "whisper_matched": w_got,
            "whisper_edit_distance": w_dist,
            "mic": r["mic_processing"], "duration_s": c["duration_s"],
        })

    n = len(letters)
    print("=" * 72)
    print(f"ACCURACY over {n} letters, each chosen from all 28 candidates")
    print("=" * 72)
    print(f"  CTC candidate scoring : {ctc_hits}/{n}  ({100*ctc_hits/n:.1f}%)")
    print(f"  Whisper + text match  : {wh_hits}/{n}  ({100*wh_hits/n:.1f}%)")

    # ---------- mic-setting check ----------
    print("\nMIC SETTING CHECK (is the recording condition a confound?)")
    for setting in ("on", "off"):
        grp = [p for p in per_clip if p["mic"] == setting]
        if grp:
            hits = sum(p["ctc_heard"] == p["expected"] for p in grp)
            print(f"  processing {setting:<3}: CTC {hits}/{len(grp)}")

    # ---------- confusions ----------
    print("\nCTC ERRORS (what it heard instead)")
    if not confusions:
        print("  none")
    for want, got, conf, mic in confusions:
        print(f"  {want:<8} -> heard {got:<8} conf {conf:.2f}  (mic {mic})")

    # ---------- the pairs that matter ----------
    print("\nCONFUSABLE PAIRS FROM THE BRIEF")
    by_name = {p["expected"]: p for p in per_clip}
    pair_rows = []
    for a, b in CONFUSABLE_PAIRS:
        pa, pb = by_name.get(a), by_name.get(b)
        if not pa or not pb:
            continue
        a_ok = pa["ctc_heard"] == a
        b_ok = pb["ctc_heard"] == b
        collapsed = pa["ctc_heard"] == b or pb["ctc_heard"] == a
        verdict = "BOTH OK" if (a_ok and b_ok) else ("COLLAPSED" if collapsed else "MIXED")
        pair_rows.append((a, b, pa["ctc_heard"], pb["ctc_heard"], verdict))
        print(f"  {a:<6} / {b:<6}  heard {pa['ctc_heard']:<7} / {pb['ctc_heard']:<7}  {verdict}")

    # ---------- wrong pairings ----------
    print("\nWRONG-PRONUNCIATION CASES (clip paired with the wrong expected letter)")
    wrong_ok = 0
    for fn, pretend_expected in WRONG_PAIRINGS:
        heard = results["ctc"][fn]["ranking"][0]["name"]
        conf = results["ctc"][fn]["ranking"][0]["conf"]
        caught = heard != pretend_expected
        wrong_ok += caught
        print(f"  {fn:<12} asked for {pretend_expected:<7} heard {heard:<7} "
              f"conf {conf:.2f}  {'CAUGHT' if caught else 'MISSED'}")
    print(f"  -> {wrong_ok}/{len(WRONG_PAIRINGS)} wrong answers correctly rejected")

    # ---------- unclear ----------
    print("\nUNCLEAR CASES (these should NOT get a confident letter)")
    for r in unclear:
        c = results["ctc"][r["filename"]]
        top = c["ranking"][0]
        print(f"  {r['filename']:<14} top {top['name']:<7} conf {top['conf']:.2f} "
              f"margin {c['margin']:.3f}  greedy: {c['greedy'][:28] or '(nothing)'}")

    # ---------- confidence separation ----------
    right = [p["ctc_conf"] for p in per_clip if p["ctc_heard"] == p["expected"]]
    wrong = [p["ctc_conf"] for p in per_clip if p["ctc_heard"] != p["expected"]]
    unclear_confs = [results["ctc"][r["filename"]]["ranking"][0]["conf"] for r in unclear]
    print("\nCONFIDENCE SEPARATION (can confidence tell these apart?)")
    if right:
        print(f"  correct  : min {min(right):.2f}  median {statistics.median(right):.2f}")
    if wrong:
        print(f"  incorrect: max {max(wrong):.2f}  median {statistics.median(wrong):.2f}")
    if unclear_confs:
        print(f"  unclear  : max {max(unclear_confs):.2f}")
    if right and wrong and min(right) > max(wrong):
        print(f"  -> a threshold near {(min(right)+max(wrong))/2:.2f} would separate them")
    else:
        print("  -> confidence alone does NOT separate correct from incorrect")

    # ---------- latency ----------
    print("\nLATENCY (CPU, 2 cores)")
    for label, key in (("CTC", "ctc"), ("Whisper", "whisper")):
        lat = [results[key][r["filename"]]["latency_ms"] for r in rows]
        load = results[key][rows[0]["filename"]]["load_seconds"]
        print(f"  {label:<8} median {statistics.median(lat):>6.0f} ms   "
              f"max {max(lat):>6.0f} ms   model load {load:.1f}s (cold start)")

    # ---------- save ----------
    out = {
        "test_set": {"letters": n, "unclear": len(unclear)},
        "accuracy": {"ctc": ctc_hits, "whisper": wh_hits, "total": n},
        "per_clip": per_clip,
        "confusable_pairs": pair_rows,
        "raw": results,
    }
    (eval_dir / "results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Evaluation results", "",
        f"Test set: {n} letters + {len(unclear)} unclear clips, one speaker, "
        "letter names, single laptop microphone.", "",
        "| Approach | Correct | Accuracy |", "| --- | --- | --- |",
        f"| CTC candidate scoring | {ctc_hits}/{n} | {100*ctc_hits/n:.1f}% |",
        f"| Whisper + text match | {wh_hits}/{n} | {100*wh_hits/n:.1f}% |", "",
        "## Per-clip", "",
        "| File | Expected | CTC heard | Conf | Whisper text | Whisper matched |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for p in per_clip:
        lines.append(
            f"| {p['file']} | {p['expected']} | {p['ctc_heard']} | "
            f"{p['ctc_conf']:.2f} | {p['whisper_text']} | {p['whisper_matched']} |"
        )
    (eval_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nWrote eval/results.json and eval/RESULTS.md")
