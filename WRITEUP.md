# Write-up

## What I built

A Modal service that takes a short clip and the letter the learner was asked
to say, and returns `correct`, `incorrect` or `unclear`. A demo page is
served from the same URL. It runs on CPU and the whole project cost a few
pence of Modal credit.

The core problem is that a single letter is not a word, and most speech
models will not commit to one of 28 options. My answer is to never ask for a
transcription at all: run an Arabic wav2vec2 CTC model once over the audio,
then score all 28 letter names against those frame probabilities with CTC
loss and rank them. The model cannot emit anything that is not one of the 28.

## What I tried, and what I rejected

**Whisper (rejected).** The obvious baseline, and the brief predicted it
would struggle. It does: 17/28 once its free text is matched to the nearest
letter name by edit distance, at roughly twice the latency. It turns letters
into words, so "baa" comes back as a real Arabic word. That is because it
generates plausible running speech, and one letter is not that. It also can't be asked
the 28-way question directly; you have to guess afterwards which letter its
text was aiming at, which is a whole extra source of error.

**CTC candidate scoring (kept).** 24/28 on the model alone, 23/28 through the
deployed service once the `unclear` rule applies. Twice as fast. The win is
structural rather than incidental: the output *is* a ranking over the 28
options, with scores I can threshold on.

**A phoneme recogniser (not attempted).** The third approach in the brief. I
skipped it deliberately: the scoring approach already solves the "commit to
one of 28" problem, and my remaining failures are specific pairs that I don't
think a different model fixes without training data. With hindsight, and for a
reason I only discovered on the last day, a phoneme model is exactly what I'd
try next. See the final section.

**Confidence as the `unclear` signal (rejected).** The obvious design, and it
does not work. Across my test set the worst *wrong* answer scored 0.98 and
the weakest *right* answer scored 0.40. They overlap almost entirely. The
reason is structural: confidence is a softmax over the 28 candidates, so it
only ever says which candidate won, never whether any of them fit. Given a
cough, one letter still wins.

**What I used instead.** Two signals that measure different things:
the raw CTC score, which says how well the best letter explains the audio at
all, and the margin between the top two. A spoken sentence scores −38.6 where
the worst real letter scores −18.8. Silence produces a near-tie: a 0.23 gap
between the top two, which is what "I heard no letter" should look like. The
rule is `unclear` if score < −25 or margin < 1.1.

I also found that the greedy output for the sentence was the only clip in the
set containing a space, so "more than one word" would have identified it
perfectly. I did not use it: one sample, no mechanism behind it, and it would
be fitting my test set rather than the problem.

## What the numbers say

Through the deployed endpoint, one speaker, each clip chosen from all 28:

| Case | Result |
| --- | --- |
| Clean letters | 23/28 correct (82.1%), 3 wrong, 2 abstained |
| Mispronunciations (right name, wrong consonant) | 6/9 caught |
| Wrong letter entirely | 5/6 caught |
| Unclear: silence, cough, sentence | 3/3 caught |
| Latency, warm | median ~1.4 s, p90 ~1.5 s |
| Latency, cold | 9.6 s round trip, 2.0 s of it in my code |

The threshold choice was made on Sunday from offline numbers, predicting it
would convert 24/28 into "23 correct, 3 wrong, 2 abstentions". The live
service did exactly that. `khaa`, previously a confidently wrong answer, now
abstains. `DHaa`, the weakest correct answer, abstains too. That was the one
correct answer I knowingly traded away. For a tutor that is the right direction: "I
didn't catch that" costs three seconds, "you said ط" when they said خ teaches
something false.

## Where it fails

**The big one: it scores letter names, not sounds.** This is the flaw in my
own design, and I only exposed it because I was recommended to test
mispronunciations properly.

The service catches a wrong pronunciation **only when some other letter's
name becomes a better match for what was said**:

- The names taa/Taa, kaaf/qaaf and haa/Haa differ *only* in the first
  consonant, so getting that consonant wrong immediately makes a different
  name fit better. All three were caught.
- The names seen and saad differ in the vowel and the final letter too.
  Saying "saad" with a plain English s still matches saad far better than
  seen, so the service says **correct**. Both directions of that pair fail.
- The name ayn has no near neighbour among the 28, so a weak ayn still wins.

**This means my own confusable-pair table overstated things.** I reported
س/ص as "both correct" on day 3. That only showed the service can tell سين
from صاد as words. It does not show it can hear the emphatic s, and the
mispronunciation test proves it cannot. The only honest entries in that table
are the pairs whose names differ solely by the consonant, and one of those,
ه/ح, fails outright.

**The ه/ح collapse.** Both come back as ح. It is one of the three wrong answers,
and it is the single case where a learner asked for ح who says ه is told they
were right.

**Other failures.** ث heard as ب, غ heard as ع. Confidence is not calibrated
and should not be shown to a learner. A cough sits close to the threshold.

**The test set is the weakest part of the work.** One speaker, one laptop
microphone, one quiet room, one clip per letter. A single bad recording moves
accuracy by 3.6 points, and I cannot distinguish a systematic error from an
unlucky take. Nothing here says how the service behaves on a learner's voice,
which is the actual user.

## A mistake worth recording

My first day-2 results were invalid and I nearly wrote them up. Audio was
silently failing to decode, because ffmpeg cannot read .m4a from a pipe when
the format keeps its index at the end of the file. Whisper pads every input to
30 seconds, so it was cheerfully transcribing silence and inventing text.
I had a confident write-up of "Whisper hallucinates on isolated letters"
based on measurements of nothing.

I only caught it because the wav2vec2 script crashed on an empty array where
Whisper had quietly padded it. The fix was one line; the lesson was that I
had computed clip duration and not printed it. Every evaluation now prints
duration on every row, and the service returns `duration_s` on every
response, so the same failure cannot hide again.

## What I would do with another two weeks

**Score the sound, not the name.** This is the first thing, and it follows
directly from the mispronunciation result. Either accept the bare sound
(/sˤ/ rather than "saad"), or keep the name for prompting but force-align it
and score only the consonant segment. Right now the vowels are doing work
they should not be doing.

**Try the phoneme recogniser after all.** A model that outputs phonemes
rather than spelling would let me compare the emphatic and plain consonants
directly, rather than inferring it from which whole name fits best. That is a
better fit for this problem than I judged on day 3.

**A test set that can carry the weight.** Several speakers including
non-native learners, several clips per letter, and 20 to 30 unclear clips of
different kinds. My thresholds are currently fitted to three unclear clips,
which is enough to show a threshold exists and roughly where, and not enough
to trust the exact number.

**Fix ه/ح specifically**, probably with a small classifier trained on just
that pair, since it is the failure that most directly harms a learner.

**Decide the keep-warm policy on data.** `scaledown_window=300` was a
judgement call: a learner works through letters in a burst, so one cold start
per session rather than one per letter. Real usage patterns would tell me
whether five minutes is right, and whether a permanently warm container is
worth the idle cost.

## Where AI helped and where it led me wrong

I used Claude throughout, and the brief asks me to be specific.

**Helped:** explaining Modal and CTC scoring, writing the evaluation
harnesses, and working through the threshold analysis. The idea of using raw
score and margin instead of confidence came out of looking at the data
together.

**Led me wrong:** the first version of the audio decode was piped into
ffmpeg, which silently broke everything and produced a confident, wrong
conclusion about Whisper that I nearly submitted. It also wrote an evaluation
script that swallowed errors, so when TLS failed on my work laptop it printed
46 identical "ERROR" lines instead of one useful message. Both were cases of
the failure being hidden rather than loud, which is the thing I would watch
for next time.
