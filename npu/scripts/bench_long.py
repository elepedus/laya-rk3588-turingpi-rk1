"""Measure warm LAN latency of one repeated-state request across services."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

HOST = os.environ.get("LAYA_BENCH_HOST", "127.0.0.1")


def post(port, body):
    request = Request(
        f"http://{HOST}:{port}/v1/systemone", data=body,
        headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urlopen(request, timeout=240) as response:
        value = json.load(response)
    return value, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--repeat-state", type=int, default=1)
    parser.add_argument("--ports", type=int, nargs="+", default=[8003])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeat_state < 1:
        parser.error("rounds and repeat-state must be positive")
    payload = json.loads(args.fixture.read_text())
    if args.repeat_state > 1:
        payload["state"] = (payload["state"] + " ") * args.repeat_state
    body = json.dumps(payload, ensure_ascii=False).encode()
    for port in args.ports:
        _, first = post(port, body)
        samples = []
        for _ in range(args.rounds):
            value, elapsed = post(port, body)
            samples.append(elapsed)
        print(f"fixture={args.fixture.stem} repeat={args.repeat_state} "
              f"port={port} model={value['routing']['model']} "
              f"tokens={value['usage']['input_tokens']} "
              f"first_s={first:.3f} warm_median_s={statistics.median(samples):.3f} "
              f"warm_samples_s={[round(sample, 3) for sample in samples]}",
              flush=True)


if __name__ == "__main__":
    main()
