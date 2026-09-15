#!/usr/bin/env python3
import sys
from pathlib import Path
from collections import Counter

def edit_ops(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]  # S, D, I

    for i in range(1, n + 1):
        dp[i][0] = i
        ops[i][0] = (0, i, 0)

    for j in range(1, m + 1):
        dp[0][j] = j
        ops[0][j] = (0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = []

            # correct or substitution
            s0, d0, i0 = ops[i - 1][j - 1]
            if ref[i - 1] == hyp[j - 1]:
                candidates.append((dp[i - 1][j - 1], (s0, d0, i0)))
            else:
                candidates.append((dp[i - 1][j - 1] + 1, (s0 + 1, d0, i0)))

            # deletion
            s0, d0, i0 = ops[i - 1][j]
            candidates.append((dp[i - 1][j] + 1, (s0, d0 + 1, i0)))

            # insertion
            s0, d0, i0 = ops[i][j - 1]
            candidates.append((dp[i][j - 1] + 1, (s0, d0, i0 + 1)))

            best_dist, best_ops = min(candidates, key=lambda x: x[0])
            dp[i][j] = best_dist
            ops[i][j] = best_ops

    return ops[n][m]  # S, D, I

def load_text(path):
    path = Path(path)
    data = {}

    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("output.*/1best_recog/text"))
        if not files and (path / "1best_recog/text").exists():
            files = [path / "1best_recog/text"]

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split(maxsplit=1)
                utt = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                data[utt] = text

    return data

def repetition_ratio(tokens):
    if not tokens:
        return 0.0
    c = Counter(tokens)
    return max(c.values()) / len(tokens)

def is_hallucination(ref, hyp):
    ref_chars = ref.replace(" ", "")
    hyp_chars = hyp.replace(" ", "")
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    s, d, i = edit_ops(list(ref_chars), list(hyp_chars))
    cer = (s + d + i) / max(len(ref_chars), 1)

    ws, wd, wi = edit_ops(ref_tokens, hyp_tokens)
    wer = (ws + wd + wi) / max(len(ref_tokens), 1)

    hyp_ref_char_ratio = len(hyp_chars) / max(len(ref_chars), 1)
    hyp_ref_tok_ratio = len(hyp_tokens) / max(len(ref_tokens), 1)

    rep_tok = repetition_ratio(hyp_tokens)
    rep_char = repetition_ratio(list(hyp_chars))

    if cer >= 1.0 and hyp_ref_char_ratio >= 1.5:
        return True
    if wer >= 1.0 and hyp_ref_tok_ratio >= 1.5:
        return True
    if len(ref_tokens) <= 2 and len(hyp_tokens) >= 8:
        return True
    if rep_tok >= 0.45 and len(hyp_tokens) >= 6:
        return True
    if rep_char >= 0.35 and len(hyp_chars) >= 15:
        return True

    return False

def weighted_cer_utt(s, d, i, L):
    # Formula:
    # Weighted CER_i = w_s * s/L + w_d * d/L + w_i * i/L
    # w_s = (s/L) / T_i, w_d = (d/L) / T_i, w_i = (i/L) / T_i
    # T_i = (s+d+i)/L
    E = s + d + i
    if L <= 0 or E == 0:
        return 0.0

    T = E / L
    ws = (s / L) / T
    wd = (d / L) / T
    wi = (i / L) / T

    return ws * (s / L) + wd * (d / L) + wi * (i / L)

def main():
    if len(sys.argv) < 3:
        print("Usage: python calc_weighted_cer_hall.py REF_TEXT HYP_DIR_OR_TEXT")
        sys.exit(1)

    refs = load_text(sys.argv[1])
    hyps = load_text(sys.argv[2])

    common = sorted(set(refs) & set(hyps))

    if not common:
        print(f"No matching utterance IDs found. refs={len(refs)}, hyps={len(hyps)}")
        sys.exit(1)

    total_s = total_d = total_i = total_L = 0
    weighted_sum = 0.0
    hall_count = 0

    for utt in common:
        ref = refs[utt].replace(" ", "")
        hyp = hyps[utt].replace(" ", "")

        s, d, i = edit_ops(list(ref), list(hyp))
        L = max(len(ref), 1)

        total_s += s
        total_d += d
        total_i += i
        total_L += L

        weighted_sum += weighted_cer_utt(s, d, i, L)

        if is_hallucination(refs[utt], hyps[utt]):
            hall_count += 1

    cer = (total_s + total_d + total_i) / total_L * 100
    corpus_weighted_cer = weighted_sum / len(common) * 100
    hall_rate = hall_count / len(common) * 100

    print(f"Matched utterances: {len(common)} / refs={len(refs)}, hyps={len(hyps)}")
    print(f"S/D/I: {total_s}/{total_d}/{total_i}")
    print(f"CER: {cer:.2f}% ({total_s + total_d + total_i}/{total_L})")
    print(f"Weighted-CER: {corpus_weighted_cer:.2f}%")
    print(f"Hall.: {hall_rate:.2f}% ({hall_count}/{len(common)})")

if __name__ == "__main__":
    main()
