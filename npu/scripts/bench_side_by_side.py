"""Compare warmed CPU, Mali, and NPU API latency on matched requests."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

from verify_api import differences


PORTS = (8000, 8002, 8003)
HOST = os.environ.get("LAYA_BENCH_HOST", "127.0.0.1")
ROOT = Path(os.environ.get("LAYA_FIXTURES_DIR", Path(__file__).resolve().parents[1] / "fixtures"))
CASES = (
    ("English short", "billing", 1),
    ("Multilingual short", "spanish", 1),
    ("Typed short", "typed", 1),
    ("English medium", "long-english", 1),
    ("English max", "billing", 50),
    ("Multilingual long", "spanish", 30),
    ("Typed long", "typed", 30),
)


def post(port, body):
    request = Request(
        f"http://{HOST}:{port}/v1/systemone",
        data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlopen(request, timeout=240) as response:
        value = json.load(response)
    return value, time.perf_counter() - started


def payload(name, repeat_state):
    fixture = ROOT / f"{name}.json"
    if not fixture.is_file():
        fixture = Path(__file__).resolve().parents[2] / "fixtures" / f"{name}.json"
    value = json.loads(fixture.read_text())
    if repeat_state > 1:
        value["state"] = (value["state"] + " ") * repeat_state
    return json.dumps(value, ensure_ascii=False).encode()


def run_case(label, name, repeat_state, rounds):
    body = payload(name, repeat_state)
    samples = {port: [] for port in PORTS}
    first = {}
    latest = {}
    for port in PORTS:
        latest[port], first[port] = post(port, body)
    for round_number in range(rounds):
        order = PORTS[round_number % len(PORTS):] + PORTS[:round_number % len(PORTS)]
        for port in order:
            latest[port], elapsed = post(port, body)
            samples[port].append(elapsed)
    reference = latest[8000]
    tokens = reference["usage"]["input_tokens"]
    route = reference["routing"]["model"]
    comparisons = {}
    for port in PORTS[1:]:
        candidate = latest[port]
        items = list(differences(reference["answers"], candidate["answers"], "answers"))
        comparisons[port] = {
            "selected_agreement": all(item[1] == "numeric" for item in items)
                                  and candidate["routing"]["model"] == route,
            "max_numeric_delta": max((abs(item[2] - item[3]) for item in items
                                      if item[1] == "numeric"), default=0),
            "tokens_match": candidate["usage"]["input_tokens"] == tokens,
        }
    result = {
        "case": label,
        "tokens": tokens,
        "model": route,
        "median_s": {port: round(statistics.median(values), 3)
                     for port, values in samples.items()},
        "samples_s": {port: [round(value, 3) for value in values]
                      for port, values in samples.items()},
        "first_s": {port: round(value, 3) for port, value in first.items()},
        "comparisons": comparisons,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not all(item["selected_agreement"] and item["tokens_match"]
               for item in comparisons.values()):
        raise SystemExit(f"response mismatch in {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    for case in CASES:
        run_case(*case, args.rounds)


if __name__ == "__main__":
    main()
