"""Summarize LAYA_NPU_TIMING stage logs from the Rust NPU service."""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path


def fields(line):
    return dict(part.split("=", 1) for part in line.split()
                if "=" in part)


def median(values):
    return statistics.median(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--bucket", type=int)
    parser.add_argument("--tail-lines", type=int,
                        help="analyze only the latest log lines")
    args = parser.parse_args()
    if args.tail_lines is not None and args.tail_lines <= 0:
        parser.error("--tail-lines must be positive")
    requests = defaultdict(lambda: defaultdict(list))
    stages = defaultdict(lambda: defaultdict(list))
    lines = args.log.read_text(errors="replace").splitlines()
    if args.tail_lines is not None:
        lines = lines[-args.tail_lines:]
    for line in lines:
        if line.startswith("NPU_REQUEST "):
            item = fields(line)
            key = item["model"], int(item["bucket"])
            if args.model and key[0] != args.model:
                continue
            if args.bucket and key[1] != args.bucket:
                continue
            for name in ("lookup_ms", "mask_prepare_ms", "encoder_wall_ms",
                         "encoder_npu_us", "total_ms"):
                if name in item:
                    requests[key][name].append(float(item[name]))
        elif line.startswith("NPU_STAGE "):
            item = fields(line)
            if "bucket" not in item:
                continue
            key = item["model"], int(item["bucket"])
            if args.model and key[0] != args.model:
                continue
            if args.bucket and key[1] != args.bucket:
                continue
            stages[key][item["stage"]].append((float(item["wall_ms"]),
                                                 float(item["npu_us"]) / 1000.0))
    for (model, bucket), data in sorted(requests.items()):
        encoder_wall = median(data["encoder_wall_ms"])
        encoder_npu = median(data["encoder_npu_us"]) / 1000.0
        print(f"{model} bucket={bucket} samples={len(data['total_ms'])} "
              f"total_ms={median(data['total_ms']):.2f} "
              f"lookup_ms={median(data['lookup_ms']):.2f} "
              f"mask_ms={median(data['mask_prepare_ms']):.2f} "
              f"encoder_wall_ms={encoder_wall:.2f} "
              f"encoder_npu_ms={encoder_npu:.2f} "
              f"encoder_host_gap_ms={encoder_wall-encoder_npu:.2f}")
        ranked = sorted(((name, median([wall for wall, _ in values]),
                          median([npu for _, npu in values]))
                         for name, values in stages[(model, bucket)].items()),
                        key=lambda value: value[1], reverse=True)
        for name, wall, npu in ranked[:10]:
            print(f"  {name:18} wall_ms={wall:8.2f} npu_ms={npu:8.2f} "
                  f"host_gap_ms={wall-npu:7.2f}")


if __name__ == "__main__":
    main()
