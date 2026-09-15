#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def read_events(event_dir):
    event_dir = Path(event_dir)
    event_files = sorted(event_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No tensorboard event files found in {event_dir}")

    ea = EventAccumulator(str(event_dir))
    ea.Reload()
    return ea

def main():
    if len(sys.argv) != 3:
        print("Usage: python export_espnet_metrics.py <exp_dir> <out_csv>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    rows = {}

    for split in ["train", "valid"]:
        tb_dir = exp_dir / "tensorboard" / split
        if not tb_dir.exists():
            continue

        ea = read_events(tb_dir)

        for tag in ea.Tags().get("scalars", []):
            for ev in ea.Scalars(tag):
                epoch = ev.step
                rows.setdefault(epoch, {"step": epoch})
                rows[epoch][f"{split}/{tag}"] = ev.value

    if not rows:
        raise RuntimeError(f"No scalar metrics found in {exp_dir}/tensorboard")

    fieldnames = ["epoch"]
    for row in rows.values():
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in sorted(rows):
            writer.writerow(rows[epoch])

    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
