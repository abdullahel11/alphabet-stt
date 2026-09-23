"""
Level 0 Alphabet STT - the deployed service.

One Modal app, one URL:
    GET  /        the demo page (pick a letter, record, see the verdict)
    POST /check   the API: audio + expected letter -> verdict

How a verdict is reached
------------------------
1. Decode the audio to 16 kHz mono with ffmpeg.
2. Run an Arabic wav2vec2 CTC model over it once.
3. Score the audio against all 28 letter NAMES (I accept the name, e.g.
   "seen", not the bare sound). The model is never free to output
   anything that is not one of the 28 - it has to rank them.
4. If no letter fits the audio well, or the top two are nearly tied,
   the answer is "unclear". Otherwise compare the winner to the letter
   the learner was asked for: "correct" or "incorrect".

Why these two unclear signals and not confidence: on my test set,
confidence overlapped completely between right and wrong answers (worst
wrong 0.98, weakest right 0.40). The raw score and the top-two margin
did separate - see eval/RESULTS.md and the write-up.

Deploy:   modal deploy modal_app.py
"""

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path

import modal

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

# ---- thresholds, set on evidence from eval/evaluate.py ----
# Worst genuine letter on my test set scored -18.8; a spoken sentence -38.6.
SCORE_FLOOR = -25.0
# Silence produced a top-two gap of 0.23; the weakest correct letter 0.65.
# 1.1 also catches the cough (1.03) at the cost of that one weak letter.
MARGIN_FLOOR = 1.1
# A single letter name is ~1-3 s. Anything much longer is not one letter.
MAX_SECONDS = 6.0
MIN_SECONDS = 0.1

# All 28 letters: the letter, how its NAME is spelled, and a Latin label.
# Latin labels are case-sensitive on purpose: "haa" is ه and "Haa" is ح.
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
BY_LETTER = {letter: (letter, ar, en) for letter, ar, en in LETTERS}
BY_NAME = {en: (letter, ar, en) for letter, ar, en in LETTERS}


class AudioError(Exception):
    """The upload could not be turned into usable audio."""


def decode_audio(audio_bytes: bytes):
    """Any common audio format -> mono 16 kHz float32 samples.

    ffmpeg reads from a real temp file, never a pipe: .m4a/.mp4 keep their
    index at the end of the file, and reading them from a pipe silently
    returned near-empty audio in my first day of testing.
    """
    import numpy as np

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-i", path,
             "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            capture_output=True,
        )
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        raise AudioError("could not decode the audio")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def resolve_letter(value: str):
    """Accept the Arabic letter ("س") or its Latin label ("seen")."""
    value = (value or "").strip()
    return BY_LETTER.get(value) or BY_NAME.get(value)


def check_duration(duration_s: float):
    if duration_s < MIN_SECONDS:
        raise AudioError("no audio")
    if duration_s > MAX_SECONDS:
        raise AudioError("too long for a single letter")


def decide(ranking, expected):
    """Pure verdict logic, separated so it can be tested without a model.

    ranking:  list of (letter, name_en, score, prob), best first.
    expected: (letter, name_ar, name_en) the learner was asked for.
    """
    top = ranking[0]
    margin = top[2] - ranking[1][2]

    reason = None
    if top[2] < SCORE_FLOOR:
        reason = "no letter matched the audio well"
    elif margin < MARGIN_FLOOR:
        reason = "could not tell between the closest letters"

    if reason:
        verdict, heard = "unclear", None
    else:
        verdict = "correct" if top[0] == expected[0] else "incorrect"
        heard = top[0]

    return {
        "verdict": verdict,
        "heard": heard,
        "heard_name": top[1] if heard else None,
        "expected": expected[0],
        "confidence": round(top[3], 3),
        "reason": reason,
        "top3": [{"letter": l, "name": n, "score": round(s, 2)} for l, n, s, _ in ranking[:3]],
        "margin": round(margin, 2),
    }


def build_api(score_fn, demo_html: str):
    """The web layer. score_fn(bytes) -> (ranking, duration_s).

    Kept separate from the Modal class so the request handling can be
    tested locally with a stub scorer, without downloading a model.
    """
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from starlette.concurrency import run_in_threadpool

    api = FastAPI(title="Level 0 Alphabet STT")
    api.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @api.get("/", response_class=HTMLResponse)
    def demo():
        return demo_html

    @api.get("/letters")
    def letters():
        return [{"letter": l, "name": en, "name_ar": ar} for l, ar, en in LETTERS]

    @api.post("/check")
    async def check(request: Request):
        started = time.time()

        # Two ways in: JSON with base64 audio (the contract in the brief),
        # or a multipart file upload (what the demo page and curl -F send).
        if request.headers.get("content-type", "").startswith("multipart/"):
            form = await request.form()
            upload = form.get("audio")
            audio_bytes = await upload.read() if hasattr(upload, "read") else b""
            expected_raw = str(form.get("expected_letter", ""))
        else:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "send JSON or a multipart upload"}, 400)
            b64 = body.get("audio", "")
            if "," in b64 and b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]  # accept data: URLs too
            try:
                audio_bytes = base64.b64decode(b64, validate=True)
            except Exception:
                return JSONResponse({"error": "audio is not valid base64"}, 400)
            expected_raw = body.get("expected_letter", "")

        expected = resolve_letter(expected_raw)
        if expected is None:
            return JSONResponse(
                {"error": f"unknown expected_letter {expected_raw!r}; "
                          "use the Arabic letter or its name, see GET /letters"}, 400)
        if not audio_bytes:
            return JSONResponse({"error": "no audio received"}, 400)

        try:
            ranking, duration_s = await run_in_threadpool(score_fn, audio_bytes)
        except AudioError as exc:
            result = {"verdict": "unclear", "heard": None, "heard_name": None,
                      "expected": expected[0], "confidence": 0.0,
                      "reason": str(exc), "top3": [], "margin": None,
                      "duration_s": None}
        else:
            result = decide(ranking, expected)
            # Reported on every response: a near-zero duration was how a
            # silent decode failure hid in my first day of results.
            result["duration_s"] = round(duration_s, 2)

        result["latency_ms"] = round((time.time() - started) * 1000)
        return result

    return api


# ---------------------------------------------------------------- Modal ----

app = modal.App("alphabet-stt")


def _download_model():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("transformers==4.44.2", "torch", "numpy", "fastapi[standard]")
    .run_function(_download_model)  # model baked into the image, not fetched per request
    .add_local_file(Path(__file__).parent / "demo" / "index.html", "/root/demo/index.html")
)


# CPU, not GPU: on my eval a clip takes ~1.4 s on 2 CPU cores, and total
# spend for the whole project so far is a few pence.
# scaledown_window: after a request, keep the container alive 5 minutes.
# A learner works through letters in a burst, so the first request pays the
# cold start and the rest of the session is warm. Idle time is billed, so
# this is the trade-off: 5 min of CPU after each burst vs a cold start on
# every letter.
@app.cls(image=image, cpu=2, memory=4096, scaledown_window=300)
class Checker:
    @modal.enter()
    def load(self):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        started = time.time()
        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).eval()
        self.candidates = [
            (letter, en, self.processor.tokenizer(ar).input_ids)
            for letter, ar, en in LETTERS
        ]
        self.blank_id = self.processor.tokenizer.pad_token_id
        self.demo_html = Path("/root/demo/index.html").read_text(encoding="utf-8")
        print(f"model loaded in {time.time() - started:.1f}s")

    def score(self, audio_bytes: bytes):
        """Rank all 28 letter names against the audio."""
        torch = self.torch
        F = torch.nn.functional

        audio = decode_audio(audio_bytes)
        duration_s = len(audio) / 16000
        check_duration(duration_s)

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        frames = torch.tensor([log_probs.shape[0]])

        scored = []
        for letter, en, ids in self.candidates:
            loss = F.ctc_loss(
                log_probs, torch.tensor([ids]), frames, torch.tensor([len(ids)]),
                blank=self.blank_id, reduction="sum", zero_infinity=True,
            )
            # Log-likelihood per character, so short names are not favoured.
            scored.append((letter, en, -loss.item() / len(ids)))

        scored.sort(key=lambda r: r[2], reverse=True)
        probs = torch.softmax(torch.tensor([s for _, _, s in scored]), dim=0).tolist()
        ranking = [(l, n, s, p) for (l, n, s), p in zip(scored, probs)]
        return ranking, duration_s

    @modal.asgi_app()
    def web(self):
        return build_api(self.score, self.demo_html)
