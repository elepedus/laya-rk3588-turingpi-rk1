# CPU, Mali GPU, and NPU benchmark — 2026-09-27

All measurements are end-to-end `POST /v1/systemone` response times on one
Turing Pi RK1 running Ubuntu 22.04. The client ran on the board and addressed
its LAN interface. Ports 8000, 8002, and 8003 served the upstream CPU, Mali
GPU, and RK3588 NPU paths respectively.

Each path received one warm-up request per case, followed by three measured
requests. Requests ran sequentially. The port order rotated each measured
round to reduce order and thermal bias. The table reports the median of those
three warmed samples; it measures latency, not concurrent throughput.

| Request | Tokens | CPU | Mali GPU | NPU | NPU speedup vs GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| English short | 48 | 2.464 s | 0.409 s | **0.221 s** | 1.85× |
| Multilingual short | 58 | 0.950 s | 0.213 s | **0.122 s** | 1.75× |
| Typed short | 58 | 2.991 s | 0.467 s | **0.257 s** | 1.82× |
| English medium | 154 | 4.833 s | 1.233 s | **1.076 s** | 1.15× |
| English maximum | 512 | 8.073 s | 6.137 s | **1.738 s** | 3.53× |
| Multilingual long | 726 | 6.160 s | 5.394 s | **1.381 s** | 3.91× |
| Typed long | 639 | 8.790 s | 8.827 s | **2.253 s** | 3.92× |

All three paths used the same token count and selected the same answers in all
seven cases. GPU answers matched the CPU numeric fields in these fixtures.
The largest CPU–NPU numeric difference was 0.0163 on the 512-token English
stress request; all other cases were at or below 0.0049. FP16 NPU confidence
values can differ even when the selected answer agrees.

The short cases use `billing.json`, `spanish.json`, and `typed.json`; the
154-token case uses `long-english.json`. The long cases repeat the state of
`billing.json` 50 times, `spanish.json` 30 times, and `typed.json` 30 times.
The benchmark source is [bench_side_by_side.py](../scripts/bench_side_by_side.py).
The [raw JSONL results](2026-09-27-side-by-side.jsonl) include every measured
sample, the warm-up call, response agreement, and token counts.
The benchmark script defaults to `127.0.0.1`; set `LAYA_BENCH_HOST` to an
RK1 LAN hostname or address when running it from another machine.
