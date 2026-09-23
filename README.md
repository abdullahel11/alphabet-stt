# Level 0 Alphabet STT

A pronunciation checker for the 28 Arabic letters. Send it a short clip of
someone saying a letter, plus the letter they were asked to say, and it
returns `correct`, `incorrect` or `unclear`.

- **Demo:** `https://abdullahel11--alphabet-stt-checker-web.modal.run` (pick a letter, record, see the verdict)
- **API:** `POST https://abdullahel11--alphabet-stt-checker-web.modal.run/check`

It accepts the letter **name** ("seen"), not the bare sound. That choice,
and everything else, is argued in [WRITEUP.md](WRITEUP.md).

---

## Try it in one minute

Open `https://abdullahel11--alphabet-stt-checker-web.modal.run` in a browser and record a letter. The first attempt after
a quiet spell takes longer while a container starts; after that it is a
second or two.

Or from a terminal, using a clip from the committed test set:

```bash
curl -F "audio=@eval/audio/seen.m4a" -F "expected_letter=seen" https://abdullahel11--alphabet-stt-checker-web.modal.run/check
```

```json
{"verdict":"correct","heard":"س","heard_name":"seen","expected":"س",
 "confidence":0.988,"reason":null,"margin":4.78,"duration_s":2.71,
 "latency_ms":1412,
 "top3":[{"letter":"س","name":"seen","score":-5.08},
         {"letter":"ش","name":"sheen","score":-9.87},
         {"letter":"ع","name":"ayn","score":-11.05}]}
```

Ask for a different letter and it should reject it:

```bash
curl -F "audio=@eval/audio/seen.m4a" -F "expected_letter=saad" https://abdullahel11--alphabet-stt-checker-web.modal.run/check
```

On Windows PowerShell, use `curl.exe`, not `curl`.

---

## API

### `POST /check`

Two ways to send audio. Either a JSON body:

```json
{ "audio": "<base64 of a wav/m4a/webm clip>", "expected_letter": "س" }
```

or a multipart upload with fields `audio` (the file) and `expected_letter`.
`expected_letter` accepts the Arabic letter (`س`) or its Latin name
(`seen`). **Names are case-sensitive**, because `haa` is ه and `Haa` is ح.

Response:

| Field | Meaning |
| --- | --- |
| `verdict` | `correct`, `incorrect` or `unclear` |
| `heard` | the letter it heard, or `null` when unclear |
| `heard_name` | that letter's name |
| `expected` | the letter that was asked for |
| `confidence` | 0 to 1. **Not calibrated**, see the write-up |
| `reason` | why it said unclear, otherwise `null` |
| `top3` | the three best-scoring letters, with raw scores |
| `margin` | gap between the best and second-best score |
| `duration_s` | length of the decoded audio |
| `latency_ms` | server-side time |

`GET /letters` returns all 28 letters with their names.

Errors return HTTP 400 with `{"error": "..."}` for an unknown
`expected_letter` or missing audio.

---

## How it decides

1. ffmpeg decodes the clip to 16 kHz mono.
2. An Arabic wav2vec2 CTC model (`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`)
   produces character probabilities per 20 ms frame.
3. Each of the **28 letter names** is scored against that audio with CTC
   loss, normalised by name length. The model can only ever return one of
   the 28, and is never free to emit anything else.
4. It answers `unclear` if the best letter does not fit the audio well
   (score < −25), if the top two are nearly tied (margin < 1.1), or if the
   clip is under 0.1 s or over 6 s. Otherwise it compares the winner to
   the letter that was asked for.

Those two thresholds come from measurement, not intuition. Confidence turned
out to be useless for this, and the write-up explains why.

---

## Results

Measured through the deployed endpoint, which is the thing you would actually
call. One speaker, each clip chosen from all 28 candidates.

| Case | Result |
| --- | --- |
| Clean letters | 23/28 correct (82.1%), 3 wrong, 2 abstained |
| Mispronunciations (right name, wrong consonant) | 6/9 caught |
| Wrong letter entirely | 5/6 caught |
| Unclear: silence, cough, sentence | 3/3 caught |
| Latency, warm | median ~1.4 s, p90 ~1.5 s |
| Latency, cold | 9.6 s round trip, 2.0 s of it in the handler |

Model-only comparison, before the `unclear` rule is applied:

| Approach | Correct | Median latency |
| --- | --- | --- |
| CTC candidate scoring (this service) | 24/28 (85.7%) | 1,408 ms |
| Whisper-small + text matching | 17/28 (60.7%) | 2,796 ms |

### The confusable pairs, honestly

Reading the pair table on its own would flatter this service, so here it is
with the caveat attached:

| Pair | Names differ by | Clean letters | Mispronunciation caught? |
| --- | --- | --- | --- |
| taa / Taa (ت, ط) | consonant only | both correct | yes |
| kaaf / qaaf (ك, ق) | consonant only | both correct | yes |
| **haa / Haa (ه, ح)** | consonant only | **collapsed** | yes (as unclear) |
| **seen / saad (س, ص)** | consonant, vowel and final letter | both correct | **no, both directions** |
| daal / Daad (د, ض) | consonant and final letter | both correct | yes |
| dhaal / DHaa (ذ, ظ) | consonant and final letter | both correct | yes |
| **alif / ayn (ا, ع)** | everything | both correct | **no** |

The service scores letter **names**, so it only catches a wrong
pronunciation when some *other* letter's name becomes a better match. Where
two names differ only in the consonant, that works. Where they differ
elsewhere too, as the names seen and saad do, a learner can say "saad" with a plain
English s and still be told they were correct, because the vowel and final
letter carry the match. This is the most important limitation of the approach and it is
argued in full in [WRITEUP.md](WRITEUP.md).

Full per-clip numbers: [eval/RESULTS.md](eval/RESULTS.md),
[eval/results.json](eval/results.json) and `eval/results_service.json`.

**Known failures**, in order of how much they matter:

- **It grades names, not sounds.** 3 of 9 deliberate mispronunciations were
  accepted as correct, including both halves of the س/ص pair.
- **The haa / Haa collapse.** Both come back as Haa, so a learner asked for
  Haa who says haa is told they were right.
- Two other single-clip errors: ث heard as ب, غ heard as ع.
- Confidence is not calibrated and should not be shown to a learner.
- A cough sits close to the `unclear` threshold.

---

## Run the evaluation yourself

### Against the deployed service (what you would actually call)

No dependencies beyond Python:

```bash
python eval/eval_service.py https://abdullahel11--alphabet-stt-checker-web.modal.run
```

Reports four things separately: clean letters, mispronunciations (right
name, wrong consonant), wrong letter entirely, and unclear clips. Add
`--cold` as the first run after 5+ minutes of inactivity to time a cold
start.

### Against the models directly (the approach comparison)

```bash
modal run eval/evaluate.py
```

Runs every clip through both the CTC scorer and Whisper, prints the
confusion and the confusable pairs, and writes `eval/RESULTS.md` and
`eval/results.json`.

---

## Deploy it from scratch

```bash
pip install modal
modal setup            # links this machine to your Modal account
modal deploy modal_app.py
```

The first deploy takes several minutes: it installs ffmpeg and PyTorch and
bakes the ~1.2 GB model into the image, so it is not downloaded per
request. `modal deploy` prints the URL at the end.

No API keys, no other services. Everything runs on CPU.

---

## The test set

`eval/audio/` with `eval/manifest.csv` describing every clip.

- 28 letters, spoken as **names**, one clip each
- silence, a cough and a full sentence, for the `unclear` path
- mispronunciations: the right letter name said with the wrong consonant
- wrong-letter cases need no recordings, because the evaluation pairs an
  existing clip with a mismatched prompt

**Where it is weak, stated plainly:**

- **One speaker** (me), one laptop microphone, one quiet room. Nothing here
  says how it behaves on a learner's voice, an accent, a child, or a phone
  in a noisy room, which is the actual target user.
- **One clip per letter**, so a single bad recording moves accuracy by 3.6
  points, and a systematic error is indistinguishable from an unlucky take.
- Partway through recording I turned off the microphone's noise
  suppression, because it was clipping the start of ف. Accuracy was 17/20
  with it on and 7/8 with it off, so it does not appear to be a confound,
  but the split is recorded in the manifest.

---

## Cost and hardware

The whole project cost a few pence of Modal's free credit, including
several multi-gigabyte image builds. Everything runs on **2 CPU cores, no
GPU**: a clip takes about 1.4 s, which was inside the budget, so a GPU was
never justified. `scaledown_window=300` keeps a container alive for five
minutes after a request, so a learner working through letters pays one cold
start rather than one per letter. Idle minutes are billed, which is the
trade-off.

---

## Layout

```
modal_app.py              the deployed service: /check and the demo page
demo/index.html           the demo page
eval/manifest.csv         every clip and what is in it
eval/audio/               the test set
eval/evaluate.py          compares both approaches on the same clips
eval/eval_service.py      evaluates the deployed endpoint end to end
eval/baseline_whisper.py  the Whisper baseline on its own
eval/approach2_ctc_scoring.py   the CTC scoring approach on its own
eval/RESULTS.md           per-clip results
WRITEUP.md                what I tried, what I rejected, what I would do next
```
