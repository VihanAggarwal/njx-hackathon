"""Build the REAL public-dataset files the loader consumes, from LLMail-Inject.

Source: microsoft/llmail-inject-challenge (HuggingFace) — the dataset from the
LLMail-Inject Adaptive Prompt Injection Challenge (real attacker submissions
against a simulated LLM email assistant, plus a held-out benign email set).

This writes:
  eval/datasets/llmailinjections/data.json  — real indirect-injection ATTACK emails
  eval/datasets/benign/data.json            — real BENIGN emails (FP test set)

so loader.load_all() picks them up as provenance="real". Downloads what it needs
(small FP set + a head sample of the raw submissions) if not already present.

    python eval/datasets/build_real.py
"""

from __future__ import annotations

import json
import os
import random
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "_downloads")
BASE = "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/main/data"
FP_URL = f"{BASE}/emails_for_fp_tests.json"
RAW_URL = f"{BASE}/raw_submissions_phase2.jsonl"

OUT_ATK = os.path.join(HERE, "llmailinjections", "data.json")
OUT_BEN = os.path.join(HERE, "benign", "data.json")


def _ensure(path, url, rng_bytes=None):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.environ.setdefault("SSL_CERT_FILE",
                          os.path.join(os.path.expanduser("~"), ".local", "cacert.pem"))
    headers = {"Range": f"bytes=0-{rng_bytes}"} if rng_bytes else {}
    req = urllib.request.Request(url, headers=headers)
    print(f"  downloading {os.path.basename(path)} ...")
    with urllib.request.urlopen(req, timeout=300) as r:
        open(path, "wb").write(r.read())
    return path


def build(n_attacks=150, n_benign=120, seed=42):
    rng = random.Random(seed)
    fp_path = _ensure(os.path.join(DL, "emails_for_fp_tests.json"), FP_URL)
    raw_path = _ensure(os.path.join(DL, "raw_p2_head.jsonl"), RAW_URL, rng_bytes=8_000_000)

    # --- real attacks (indirect injection email submissions) ------------ #
    attacks, seen = [], set()
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # final line of a range download may be truncated
            subj = (r.get("subject") or "").strip()
            body = (r.get("body") or "").strip()
            if not body:
                continue
            content = f"Subject of the email: {subj}.   Body: {body}"
            if content in seen:
                continue
            seen.add(content)
            attacks.append({"content": content, "is_attack": True,
                            "attack_class": "indirect", "content_type": "email",
                            "source": "real"})
    rng.shuffle(attacks)
    attacks = attacks[:n_attacks]

    # --- real benign emails (false-positive test set) ------------------- #
    fp = json.load(open(fp_path, "r", encoding="utf-8"))
    rng.shuffle(fp)
    benign = [{"content": e, "is_attack": False, "attack_class": "benign",
               "content_type": "email", "source": "real"} for e in fp[:n_benign]]

    os.makedirs(os.path.dirname(OUT_ATK), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_BEN), exist_ok=True)
    json.dump(attacks, open(OUT_ATK, "w", encoding="utf-8"), indent=0)
    json.dump(benign, open(OUT_BEN, "w", encoding="utf-8"), indent=0)
    print(f"  wrote {len(attacks)} REAL attacks -> {OUT_ATK}")
    print(f"  wrote {len(benign)} REAL benign  -> {OUT_BEN}")
    print("  loader.load_all() will now report llmail_inject + benign as provenance='real'.")


if __name__ == "__main__":
    print("### Building real LLMail-Inject dataset files ###")
    build()
