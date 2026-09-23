"""
Evaluate the DEPLOYED service, not the model in isolation.

eval/evaluate.py measured the model. This measures the thing a reviewer
actually calls: every clip goes over the network to POST /check with an
expected_letter, and the verdict that comes back is what gets scored.

It covers four things the brief asks for, kept separate on purpose:

  1. Clean letters        - does it recognise the right letter?
  2. Mispronunciations    - does it CATCH a wrong attempt at the right
                            letter? (right name, wrong consonant)
  3. Wrong letter entirely- an existing clip sent against a mismatched
                            prompt. The easy version of (2).
  4. Unclear              - silence, a cough, a whole sentence.

Only stdlib, so there is nothing to install:

    python eval/eval_service.py https://YOUR-URL

Add --cold as the first run after 5+ minutes of inactivity to time a
cold start.
"""

import base64
import csv
import json
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EVAL_DIR = Path(__file__).parent
AUDIO_DIR = EVAL_DIR / "audio"

# The same clip sent against a prompt for a different letter. The learner
# said a real letter, just not the one asked for. No new audio needed -
# a wrong answer is defined by the pairing, not by the recording.
WRONG_PAIRINGS = [
    ("seen.m4a", "saad"), ("saad.m4a", "seen"),
    ("daal.m4a", "Daad"), ("taa.m4a", "Taa"),
    ("haa.m4a", "Haa"), ("kaef.m4a", "qaaf"),
]


def make_ssl_context():
    """Trust whatever the machine trusts.

    Python ships its own CA bundle and ignores the Windows certificate
    store, so on a managed laptop that inspects TLS, every request fails
    with CERTIFICATE_VERIFY_FAILED even though the browser and curl are
    fine. This loads the OS root certificates on Windows.
    """
    try:
        import truststore  # optional, the tidiest fix if it is installed
        truststore.inject_into_ssl()
        print("(using the OS certificate store via truststore)\n")
        return ssl.create_default_context()
    except ImportError:
        pass

    ctx = ssl.create_default_context()
    if sys.platform == "win32" and hasattr(ssl, "enum_certificates"):
        loaded = 0
        for store in ("ROOT", "CA"):
            try:
                for cert, encoding, trust in ssl.enum_certificates(store):
                    if encoding == "x509_asn" and trust:
                        try:
                            ctx.load_verify_locations(cadata=cert)
                            loaded += 1
                        except ssl.SSLError:
                            pass
            except Exception:
                pass
        if loaded:
            print(f"(using {loaded} certificates from the Windows store)\n")
    return ctx


SSL_CTX = make_ssl_context()


def call(url: str, audio_path: Path, expected: str, timeout=180):
    payload = json.dumps({
        "audio": base64.b64encode(audio_path.read_bytes()).decode(),
        "expected_letter": expected,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/check", data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        body = {"error": f"HTTP {exc.code}: {exc.read().decode()[:300]}", "verdict": "ERROR"}
    except Exception as exc:  # network, TLS, timeout
        body = {"error": f"{type(exc).__name__}: {exc}", "verdict": "ERROR"}
    body["round_trip_ms"] = round((time.time() - started) * 1000)
    return body


def fail_fast(out, where):
    """Stop on the first error instead of repeating it 46 times."""
    if out.get("verdict") != "ERROR":
        return False
    print(f"\nFAILED on the first call ({where}):\n  {out['error']}\n")
    if "CERTIFICATE_VERIFY" in out["error"] or "SSLError" in out["error"]:
        print("  That is a TLS trust problem on this machine, not a problem\n"
              "  with the service - the browser and curl.exe use the Windows\n"
              "  certificate store, Python does not. Try:\n"
              "    pip install truststore\n"
              "  then re-run. If that fails, use the curl commands in the\n"
              "  README instead and record the results by hand.")
    else:
        print("  Check the URL is right and that the service is deployed:\n"
              f"    curl.exe {sys.argv[1] if len(sys.argv) > 1 else '<URL>'}/letters")
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cold = "--cold" in sys.argv
    if not args:
        print("usage: python eval/eval_service.py https://YOUR-URL [--cold]")
        return 1
    url = args[0]

    rows = list(csv.DictReader(open(EVAL_DIR / "manifest.csv", encoding="utf-8")))
    present = {r["filename"] for r in rows if (AUDIO_DIR / r["filename"]).exists()}
    missing = [r["filename"] for r in rows if r["filename"] not in present]
    if missing:
        print(f"Not recorded yet, skipping {len(missing)}: {', '.join(missing)}\n")

    letters = [r for r in rows if r["kind"] == "letter" and r["filename"] in present]
    mispron = [r for r in rows if r["kind"] == "mispronunciation" and r["filename"] in present]
    unclear = [r for r in rows if r["kind"] == "unclear" and r["filename"] in present]

    results, latencies = {}, []

    # One call first, so a broken setup fails in 1 call rather than 46.
    probe = letters[0] if letters else (unclear[0] if unclear else None)
    if probe is None:
        print("No audio files found. Record the test set first.")
        return 1
    out = call(url, AUDIO_DIR / probe["filename"], probe.get("name_en") or "seen")
    if fail_fast(out, probe["filename"]):
        return 1
    if cold:
        print(f"COLD START: first call took {out['round_trip_ms']} ms "
              f"(server reported {out.get('latency_ms')} ms)\n")
    else:
        print(f"(service reachable; warm-up call {out['round_trip_ms']} ms)\n")

    # ---- 1. clean letters ----
    print("=" * 74)
    print(f"1. CLEAN LETTERS  ({len(letters)} clips, each chosen from all 28)")
    print("=" * 74)
    correct = wrong = unsure = 0
    for r in letters:
        out = call(url, AUDIO_DIR / r["filename"], r["name_en"])
        results[r["filename"]] = out
        latencies.append(out["round_trip_ms"])
        v = out.get("verdict")
        correct += v == "correct"
        wrong += v == "incorrect"
        unsure += v == "unclear"
        if v != "correct":
            print(f"  {r['name_en']:<7} -> {v:<9} heard {out.get('heard_name') or '-':<7} "
                  f"conf {out.get('confidence')} {out.get('reason') or ''}")
    n = len(letters) or 1
    print(f"\n  correct {correct}/{n} ({100*correct/n:.1f}%) | "
          f"wrong {wrong} | unclear {unsure}")

    # ---- 2. mispronunciations: the hard case ----
    print("\n" + "=" * 74)
    print(f"2. MISPRONUNCIATIONS  ({len(mispron)} clips) - right name, wrong consonant")
    print("   A pass here means NOT 'correct'. Saying 'correct' to a")
    print("   mispronunciation is the worst failure a pronunciation tutor can make.")
    print("=" * 74)
    caught = 0
    for r in mispron:
        out = call(url, AUDIO_DIR / r["filename"], r["name_en"])
        results[r["filename"]] = out
        latencies.append(out["round_trip_ms"])
        v = out.get("verdict")
        ok = v in ("incorrect", "unclear")
        caught += ok
        print(f"  {r['filename']:<26} asked {r['name_en']:<6} -> {v:<9} "
              f"heard {out.get('heard_name') or '-':<7} conf {out.get('confidence')} "
              f"{'CAUGHT' if ok else 'MISSED - told them they were right'}")
    if mispron:
        print(f"\n  caught {caught}/{len(mispron)}")

    # ---- 3. wrong letter entirely ----
    print("\n" + "=" * 74)
    print("3. WRONG LETTER ENTIRELY - a real clip against a mismatched prompt")
    print("=" * 74)
    caught_w = 0
    usable = [(f, e) for f, e in WRONG_PAIRINGS if f in present]
    for fn, pretend in usable:
        out = call(url, AUDIO_DIR / fn, pretend)
        latencies.append(out["round_trip_ms"])
        ok = out.get("verdict") in ("incorrect", "unclear")
        caught_w += ok
        print(f"  {fn:<14} asked {pretend:<6} -> {out.get('verdict'):<9} "
              f"heard {out.get('heard_name') or '-':<7} {'CAUGHT' if ok else 'MISSED'}")
    if usable:
        print(f"\n  caught {caught_w}/{len(usable)}")

    # ---- 4. unclear ----
    print("\n" + "=" * 74)
    print("4. UNCLEAR - these must not get a confident letter")
    print("=" * 74)
    caught_u = 0
    for r in unclear:
        # Prompt with any letter; the point is that it should abstain.
        out = call(url, AUDIO_DIR / r["filename"], "seen")
        results[r["filename"]] = out
        latencies.append(out["round_trip_ms"])
        ok = out.get("verdict") == "unclear"
        caught_u += ok
        print(f"  {r['filename']:<16} -> {out.get('verdict'):<9} "
              f"{out.get('reason') or ''} {'OK' if ok else 'NOT CAUGHT'}")
    if unclear:
        print(f"\n  caught {caught_u}/{len(unclear)}")

    # ---- latency ----
    if latencies:
        warm = sorted(latencies)[:-1] if cold else latencies
        print("\n" + "=" * 74)
        print("LATENCY over the network, end to end")
        print("=" * 74)
        print(f"  median {statistics.median(warm):.0f} ms | "
              f"p90 {sorted(warm)[int(len(warm)*0.9)-1]:.0f} ms | max {max(warm)} ms")

    out_path = EVAL_DIR / "results_service.json"
    out_path.write_text(json.dumps({
        "endpoint": url,
        "clean_letters": {"correct": correct, "incorrect": wrong,
                          "unclear": unsure, "total": len(letters)},
        "mispronunciations": {"caught": caught, "total": len(mispron)},
        "wrong_letter": {"caught": caught_w, "total": len(usable)},
        "unclear": {"caught": caught_u, "total": len(unclear)},
        "latency_ms": {"median": statistics.median(latencies) if latencies else None,
                       "max": max(latencies) if latencies else None},
        "raw": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
