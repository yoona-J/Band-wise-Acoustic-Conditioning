#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# ============================================================
# Config
# ============================================================

SRC_DIR = Path("dump/raw/test_1k")

GATE_CSV = Path("logs/test_1k_instability_gate.csv")

OUT_ROOT = Path("dump/raw")

# ============================================================
# Load gate csv
# ============================================================

df = pd.read_csv(GATE_CSV)

bucket_map = {
    "stable": set(df[df["bucket"] == "stable"]["utt"]),
    "mid": set(df[df["bucket"] == "mid"]["utt"]),
    "unstable": set(df[df["bucket"] == "unstable"]["utt"]),
}

# ============================================================
# Create subsets
# ============================================================

FILES = [
    "wav.scp",
    "text",
    "utt2spk",
    "utt2num_samples",
]

for bucket, utts in bucket_map.items():

    out_dir = OUT_ROOT / f"test_1k_{bucket}"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{bucket}] {len(utts)} utterances")

    for fname in FILES:

        src_path = SRC_DIR / fname
        dst_path = out_dir / fname

        if not src_path.exists():
            continue

        kept = 0

        with open(src_path, encoding="utf-8") as fin, \
             open(dst_path, "w", encoding="utf-8") as fout:

            for line in fin:

                if not line.strip():
                    continue

                utt = line.split(maxsplit=1)[0]

                if utt in utts:
                    fout.write(line)
                    kept += 1

        print(f"  {fname}: {kept}")

    # keys.txt 생성
    with open(out_dir / "keys.txt", "w", encoding="utf-8") as f:
        for utt in sorted(utts):
            f.write(utt + "\n")

print("\nDone.")
