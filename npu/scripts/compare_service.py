"""Print concrete response differences between CPU, Mali and RKNN services."""

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

HOST = os.environ.get("LAYA_BENCH_HOST", "127.0.0.1")


def post(port, value):
    request = Request(
        f"http://{HOST}:{port}/v1/systemone",
        data=json.dumps(value, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    fixtures = Path(os.environ.get("LAYA_FIXTURES_DIR", Path(__file__).resolve().parents[2] / "mali/fixtures"))
    for name in ("billing", "score-noul", "short"):
        payload = json.loads((fixtures / f"{name}.json").read_text())
        print("FIXTURE", name, flush=True)
        for port in (8000, 8002, 8003):
            result = post(port, payload)
            print(port, "usage", result.get("usage"), "answers", json.dumps(result.get("answers"), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
