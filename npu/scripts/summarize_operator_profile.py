"""Summarize RKNN operator ranking tables emitted by LAYA_RKNN_PROFILE."""

import argparse
import re
from collections import defaultdict
from pathlib import Path


STAGE = re.compile(r"^NPU_OPERATOR_PROFILE model=(\S+) stage=(\S+)$")
ROW = re.compile(
    r"^(\w+)\s+(\d+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+"
    r"([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)%?\s*$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--stage")
    args = parser.parse_args()
    totals = defaultdict(lambda: defaultdict(float))
    calls = defaultdict(lambda: defaultdict(int))
    stage = None
    for line in args.log.read_text(errors="replace").splitlines():
        match = STAGE.match(line)
        if match:
            stage = f"{match[1]}/{match[2]}"
            continue
        if stage is None or (args.stage and args.stage not in stage):
            continue
        match = ROW.match(line)
        if match and match[1] != "Total":
            op = match[1]
            calls[stage][op] += int(match[2])
            totals[stage][op] += float(match[5])
    for stage in sorted(totals):
        overall = sum(totals[stage].values())
        print(f"{stage} npu_ms={overall / 1000:.3f}")
        for op, microseconds in sorted(totals[stage].items(),
                                       key=lambda item: item[1], reverse=True):
            print(f"  {op:18} calls={calls[stage][op]:3} "
                  f"npu_ms={microseconds / 1000:8.3f} "
                  f"share={100 * microseconds / overall:5.1f}%")


if __name__ == "__main__":
    main()
