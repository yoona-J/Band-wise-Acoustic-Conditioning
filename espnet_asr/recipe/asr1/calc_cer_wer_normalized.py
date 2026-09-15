#!/usr/bin/env python3
import sys
import re
from pathlib import Path

def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost
            )
            prev = temp
    return dp[m]

def load_text(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            data[utt] = text
    return data

def load_hyp_from_outputs(logdir):
    data = {}
    for text_file in Path(logdir).glob("output.*/1best_recog/text"):
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                utt = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                data[utt] = text

    # whisper direct path fallback
    direct_text = Path(logdir) / "1best_recog" / "text"
    if direct_text.exists():
        with open(direct_text, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                utt = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                data[utt] = text

    return data

def normalize_text(text):
    text = text.lower()

    # punctuation / special symbol 제거
    text = re.sub(r"[^\w\s가-힣]", " ", text)

    # 다중 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalize_char(text):
    return text.replace(" ", "")

def main():
    if len(sys.argv) != 3:
        print("Usage: python calc_cer_wer_normalized.py <ref_text> <decode_logdir>")
        sys.exit(1)

    ref_path = sys.argv[1]
    logdir = sys.argv[2]

    refs = load_text(ref_path)
    hyps = load_hyp_from_outputs(logdir)

    common = sorted(set(refs) & set(hyps))

    if not common:
        print("No matching utterance IDs found.")
        print(f"refs: {len(refs)}, hyps: {len(hyps)}")
        sys.exit(1)

    char_err = 0
    char_total = 0
    word_err = 0
    word_total = 0

    for utt in common:
        r = normalize_text(refs[utt])
        h = normalize_text(hyps[utt])

        r_chars = list(normalize_char(r))
        h_chars = list(normalize_char(h))

        r_words = r.split()
        h_words = h.split()

        char_err += edit_distance(r_chars, h_chars)
        char_total += len(r_chars)

        word_err += edit_distance(r_words, h_words)
        word_total += len(r_words)

    cer = char_err / char_total if char_total else 0
    wer = word_err / word_total if word_total else 0

    print(f"Matched utterances: {len(common)} / refs={len(refs)}, hyps={len(hyps)}")
    print(f"Normalized CER: {cer * 100:.2f}% ({char_err}/{char_total})")
    print(f"Normalized WER: {wer * 100:.2f}% ({word_err}/{word_total})")

if __name__ == "__main__":
    main()
