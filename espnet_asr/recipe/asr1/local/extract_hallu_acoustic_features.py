import os
import json
import argparse
import numpy as np
import librosa
from tqdm import tqdm

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PRAAT = True
except Exception:
    HAS_PRAAT = False


FEATURE_NAMES = [
    "f0_std",
    "f0_range",
    "token_per_sec",
    "duration_per_token",
    "silence_ratio",
    "num_pauses",
    "mean_pause",
    "rms_energy",
    "hnr_mean",
    "jitter_local",
    "shimmer_local",
]


def load_text(text_path):
    text_dict = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, text = parts
                text_dict[utt_id] = text
    return text_dict


def count_tokens(text):
    if text is None:
        return 0

    text = text.strip()
    if not text:
        return 0

    # 한국어 char-level 기준
    return len(text.replace(" ", ""))


def praat_voice_quality(wav_path):
    if not HAS_PRAAT:
        return 0.0, 0.0, 0.0

    try:
        snd = parselmouth.Sound(wav_path)

        harmonicity = call(
            snd,
            "To Harmonicity (cc)",
            0.01,
            75,
            0.1,
            1.0,
        )
        hnr = call(harmonicity, "Get mean", 0, 0)

        point_process = call(
            snd,
            "To PointProcess (periodic, cc)",
            75,
            500,
        )

        jitter = call(
            point_process,
            "Get jitter (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
        )

        shimmer = call(
            [snd, point_process],
            "Get shimmer (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )

        if not np.isfinite(hnr):
            hnr = 0.0
        if not np.isfinite(jitter):
            jitter = 0.0
        if not np.isfinite(shimmer):
            shimmer = 0.0

        return float(hnr), float(jitter), float(shimmer)

    except Exception:
        return 0.0, 0.0, 0.0


def extract_one(wav_path, text=None, sr=16000):
    y, sr = librosa.load(wav_path, sr=sr, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    duration = max(duration, 1e-6)

    # F0
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0 = f0[f0 > 0]

        if len(f0) == 0:
            f0_std = 0.0
            f0_range = 0.0
        else:
            f0_std = float(np.std(f0))
            f0_range = float(np.max(f0) - np.min(f0))
    except Exception:
        f0_std = 0.0
        f0_range = 0.0

    # Token rate
    token_count = count_tokens(text)
    token_per_sec = token_count / duration if token_count > 0 else 0.0
    duration_per_token = duration / token_count if token_count > 0 else duration

    # Silence / pause
    try:
        intervals = librosa.effects.split(y, top_db=30)

        voiced_samples = sum(end - start for start, end in intervals)
        silence_ratio = 1.0 - voiced_samples / max(len(y), 1)

        pause_durations = []
        for i in range(len(intervals) - 1):
            prev_end = intervals[i][1]
            next_start = intervals[i + 1][0]
            pause = max(0, next_start - prev_end) / sr
            pause_durations.append(pause)

        num_pauses = len(pause_durations)
        mean_pause = float(np.mean(pause_durations)) if pause_durations else 0.0

    except Exception:
        silence_ratio = 0.0
        num_pauses = 0.0
        mean_pause = 0.0

    # RMS energy
    try:
        rms_energy = float(np.mean(librosa.feature.rms(y=y)))
    except Exception:
        rms_energy = 0.0

    # Voice quality
    hnr_mean, jitter_local, shimmer_local = praat_voice_quality(wav_path)

    feat = np.array([
        f0_std,
        f0_range,
        token_per_sec,
        duration_per_token,
        silence_ratio,
        num_pauses,
        mean_pause,
        rms_energy,
        hnr_mean,
        jitter_local,
        shimmer_local,
    ], dtype=np.float32)

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def read_wav_scp(wav_scp):
    items = []
    with open(wav_scp, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                items.append((parts[0], parts[1]))
    return items


def extract_split(wav_scp, text_path, out_dir, sr=16000):
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    text_dict = load_text(text_path)
    items = read_wav_scp(wav_scp)

    feats = []
    utt_ids = []

    for utt_id, wav_path in tqdm(items, desc=f"Extracting {out_dir}"):
        text = text_dict.get(utt_id, "")
        feat = extract_one(wav_path, text=text, sr=sr)

        np.save(os.path.join(raw_dir, f"{utt_id}.npy"), feat)

        feats.append(feat)
        utt_ids.append(utt_id)

    feats = np.stack(feats, axis=0)

    return utt_ids, feats


def save_normalized(utt_ids, feats, mean, std, out_dir):
    norm_dir = os.path.join(out_dir, "norm")
    os.makedirs(norm_dir, exist_ok=True)

    scp_path = os.path.join(out_dir, "acoustic_feats.scp")

    with open(scp_path, "w", encoding="utf-8") as scp:
        for utt_id, feat in zip(utt_ids, feats):
            norm_feat = (feat - mean) / std
            norm_feat = np.nan_to_num(norm_feat, nan=0.0, posinf=0.0, neginf=0.0)

            out_path = os.path.abspath(os.path.join(norm_dir, f"{utt_id}.npy"))
            np.save(out_path, norm_feat.astype(np.float32))

            scp.write(f"{utt_id} {out_path}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_wav_scp", required=True)
    parser.add_argument("--train_text", required=True)
    parser.add_argument("--dev_wav_scp", required=True)
    parser.add_argument("--dev_text", required=True)
    parser.add_argument("--test_wav_scp", required=True)
    parser.add_argument("--test_text", required=True)

    parser.add_argument("--out_root", required=True)
    parser.add_argument("--sr", type=int, default=16000)

    args = parser.parse_args()

    train_dir = os.path.join(args.out_root, "train")
    dev_dir = os.path.join(args.out_root, "dev")
    test_dir = os.path.join(args.out_root, "test")

    train_ids, train_feats = extract_split(
        args.train_wav_scp,
        args.train_text,
        train_dir,
        sr=args.sr,
    )

    dev_ids, dev_feats = extract_split(
        args.dev_wav_scp,
        args.dev_text,
        dev_dir,
        sr=args.sr,
    )

    test_ids, test_feats = extract_split(
        args.test_wav_scp,
        args.test_text,
        test_dir,
        sr=args.sr,
    )

    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    stats = {
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "acoustic_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    save_normalized(train_ids, train_feats, mean, std, train_dir)
    save_normalized(dev_ids, dev_feats, mean, std, dev_dir)
    save_normalized(test_ids, test_feats, mean, std, test_dir)


if __name__ == "__main__":
    main()
