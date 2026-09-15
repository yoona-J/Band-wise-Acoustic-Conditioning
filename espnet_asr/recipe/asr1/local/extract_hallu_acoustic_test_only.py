import os
import json
import argparse
import numpy as np
from extract_hallu_acoustic_features import (
    extract_one,
    load_text,
    read_wav_scp,
)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wav_scp", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--stats_json", required=True)
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.stats_json, "r") as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    text_dict = load_text(args.text)
    items = read_wav_scp(args.wav_scp)

    scp_path = os.path.join(args.out_dir, "acoustic_feats.scp")

    with open(scp_path, "w") as scp:
        for utt_id, wav_path in items:

            try:
                feat = extract_one(
                    wav_path,
                    text=text_dict.get(utt_id, "")
                )
            except Exception as e:
                print(f"[SKIP] {utt_id}: {e}")
                continue

            feat = (feat - mean) / std
            feat = np.nan_to_num(feat)

            out_path = os.path.abspath(
                os.path.join(args.out_dir, f"{utt_id}.npy")
            )

            np.save(out_path, feat.astype(np.float32))

            scp.write(f"{utt_id} {out_path}\n")


if __name__ == "__main__":
    main()
