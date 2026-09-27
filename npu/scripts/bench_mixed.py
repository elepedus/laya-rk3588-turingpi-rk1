"""Measure alternating checkpoint requests and report RKNN bucket cache state."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

HOST = os.environ.get("LAYA_BENCH_HOST", "127.0.0.1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    fixtures = Path(os.environ.get("LAYA_FIXTURES_DIR", Path(__file__).resolve().parents[1] / "fixtures"))
    if not (fixtures / "billing.json").is_file():
        fixtures = Path(__file__).resolve().parents[2] / "fixtures"
    payloads = {name: (fixtures / f"{name}.json").read_bytes()
                for name in ("billing", "spanish", "typed")}
    samples = {name: [] for name in payloads}
    for round_number in range(args.rounds):
        for name, body in payloads.items():
            request = Request(f"http://{HOST}:{args.port}/v1/systemone",
                              data=body, headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            with urlopen(request, timeout=120) as response:
                value = json.load(response)
            elapsed = time.perf_counter() - started
            samples[name].append(elapsed)
            print(f"round={round_number} model={value['routing']['model']} "
                  f"fixture={name} seconds={elapsed:.3f}", flush=True)
    with urlopen(f"http://{HOST}:{args.port}/health") as response:
        health = json.load(response)
    print("median_after_first_round", json.dumps({
        name: round(statistics.median(values[1:]), 3)
        for name, values in samples.items()}), flush=True)
    print("cached_buckets", json.dumps(health["cached_buckets"]), flush=True)


if __name__ == "__main__":
    main()
