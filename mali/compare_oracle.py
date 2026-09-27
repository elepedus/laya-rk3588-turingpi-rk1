"""Compare the Rust Mali API to the running Laya Python oracle."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def post(url, payload):
    request = Request(url + "/v1/systemone", data=json.dumps(payload, ensure_ascii=False).encode(),
                      headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlopen(request, timeout=240) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default="http://127.0.0.1:8000")
    parser.add_argument("--candidate", default="http://127.0.0.1:8002")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    fixtures = [here / "fixtures" / (name + ".json")
                for name in ("billing", "spanish", "typed", "score-noul", "short", "hindi")]
    failures = 0
    for file in fixtures:
        payload = json.loads(file.read_text())
        oracle, cpu_time = post(args.oracle, payload)
        candidate, gpu_time = post(args.candidate, payload)
        fields = ("answers", "usage")
        errors = [field for field in fields if oracle[field] != candidate[field]]
        if oracle["routing"]["model"] != candidate["routing"]["model"]:
            errors.append("routing.model")
        failures += bool(errors)
        print(f"{file.stem:12} CPU={cpu_time:6.2f}s Mali={gpu_time:6.2f}s "
              f"{'MATCH' if not errors else 'DIFFER: ' + ', '.join(errors)}", flush=True)
    if failures:
        raise SystemExit(f"{failures} fixtures differ from oracle")


if __name__ == "__main__":
    main()
