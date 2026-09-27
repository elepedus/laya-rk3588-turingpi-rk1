"""Aggregate per-call RKNN host and device phases from a trace log."""

import argparse
from collections import defaultdict
from pathlib import Path


PHASES = ("input_ms", "run_ms", "output_ms", "copy_ms", "release_ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    totals = defaultdict(lambda: defaultdict(float))
    rows = []
    for line in args.log.read_text(errors="replace").splitlines():
        if not line.startswith("RKNN_PHASE "):
            continue
        values = dict(part.split("=", 1) for part in line.split()
                      if "=" in part)
        stage = values["stage"]
        group = "encoder" if stage.startswith("encoder_") else stage
        row = {name: float(values[name]) for name in PHASES}
        for name, value in row.items():
            totals[group][name] += value
        rows.append((stage, row))
    for group, phases in totals.items():
        print(group, " ".join(f"{name}={phases[name]:.3f}"
                              for name in PHASES),
              f"total_ms={sum(phases.values()):.3f}")
    print("slowest host phases:")
    for stage, row in sorted(rows, key=lambda item:
                             item[1]["input_ms"] + item[1]["output_ms"] +
                             item[1]["copy_ms"] + item[1]["release_ms"],
                             reverse=True)[:10]:
        print(stage, " ".join(f"{name}={row[name]:.3f}" for name in PHASES))


if __name__ == "__main__":
    main()
