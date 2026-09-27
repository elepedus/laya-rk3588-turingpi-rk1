"""Check NPU decisions against the running CPU oracle at FP16 tolerance."""

import argparse
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

HOST = os.environ.get("LAYA_BENCH_HOST", "127.0.0.1")


def request(port, value):
    message = Request(
        f"http://{HOST}:{port}/v1/systemone",
        data=json.dumps(value, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urlopen(message, timeout=240) as response:
        result = json.load(response)
    return result, time.perf_counter() - start


def differences(reference, actual, prefix=""):
    if isinstance(reference, dict) and isinstance(actual, dict):
        if reference.keys() != actual.keys():
            yield prefix, "keys", sorted(reference.keys()), sorted(actual.keys())
        for key in reference.keys() & actual.keys():
            yield from differences(reference[key], actual[key], f"{prefix}.{key}")
    elif isinstance(reference, (int, float)) and not isinstance(reference, bool) \
            and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        delta = abs(reference - actual)
        if delta > 0:
            yield prefix, "numeric", reference, actual
    elif reference != actual:
        yield prefix, "discrete", reference, actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--candidate-port", type=int, default=8003)
    parser.add_argument("--names", nargs="*", help="fixture names without .json")
    parser.add_argument("--repeat-state", type=int, default=1,
                        help="repeat a string state to exercise longer token buckets")
    args = parser.parse_args()
    failed = 0
    files = ([args.fixtures / f"{name}.json" for name in args.names]
             if args.names else sorted(args.fixtures.glob("*.json")))
    for file in files:
        value = json.loads(file.read_text())
        if args.repeat_state > 1:
            if not isinstance(value.get("state"), str):
                raise ValueError("--repeat-state requires a string fixture state")
            value["state"] = (value["state"] + " ") * args.repeat_state
        cpu, cpu_s = request(8000, value)
        npu, npu_s = request(args.candidate_port, value)
        fields = ("answers", "usage")
        items = [item for field in fields
                 for item in differences(cpu[field], npu[field], field)]
        items += list(differences(cpu["routing"]["model"], npu["routing"]["model"], "routing.model"))
        major = [item for item in items if item[1] != "numeric" or abs(item[2] - item[3]) > args.tolerance]
        numeric = [abs(item[2] - item[3]) for item in items if item[1] == "numeric"]
        failed += bool(major)
        print(f"{file.stem:24} tokens={npu['usage']['input_tokens']} "
              f"CPU={cpu_s:.3f}s NPU={npu_s:.3f}s "
              f"max_numeric_delta={max(numeric, default=0):.4f} "
              f"{'PASS' if not major else 'FAIL '+repr(major)}", flush=True)
    if failed:
        raise SystemExit(f"{failed} oracle comparisons exceeded tolerance")


if __name__ == "__main__":
    main()
