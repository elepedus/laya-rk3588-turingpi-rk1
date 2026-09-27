# Laya on the RK3588 NPU

For a new Turing Pi RK1 installation, use the prebuilt ARM64
[`laya-rknpu` release](https://github.com/elepedus/laya-rk3588-turingpi-rk1/releases)
and the [runtime setup](../README.md). Rust and Cargo are not required on the
RK1 to run the release binary. The graph-conversion commands below use the
tested `/mnt/warm` layout and can run on a separate ARM64 build host.

An RK1 service bound at `http://RK1_LAN_IP:8003` implements `GET /health` and
`POST /v1/systemone` for all three Laya checkpoints. It runs embedding
normalization, every ModernBERT encoder layer, the typed decision head, marker
scoring, and action inference through the [Rockchip RKNN Runtime](https://github.com/airockchip/rknn-toolkit2).
The existing CPU and Mali GPU services remain at ports 8000 and 8002.

Rust still constructs prompts, tokenizes, looks up FP16 vocabulary rows, and
decodes scalar answers. RKNN's vocabulary `Gather` runs on CPU and rounded
float token IDs incorrectly, so the Rust lookup supplies gathered vectors to
an NPU normalization graph. A one-hot type input keeps the decision head's
type projection on the NPU. No Python, Torch, ONNX Runtime, or Mali OpenCL is
used by this service at inference time.

## Lengths and memory

The service selects the smallest compiled bucket that holds each question:
64, 128, 256, 512, 768, or 1024 tokens, plus a 640-token typed bucket.
English supports its native 1–512 tokens; multilingual and typed-decisions
support 1–1024. The input is padded and
masked, including the 64-token half-window in sliding-attention layers.
Requests do not fall back to the GPU. The service processes one request at a
time and retains one bucket per checkpoint. All three checkpoints can remain
loaded at once; an RKNN allocation failure evicts the other checkpoints' bucket
caches and retries the request. Switching buckets loads graphs from `/mnt/warm`
and adds a cold-load delay. The 512-token English, 768/1024-token multilingual,
and 640/768/1024-token typed buckets use two fused encoder graphs each; smaller
buckets use individual layer graphs. The 768-token multilingual and 640/768-token
typed graphs tile sliding attention internally, calculating only each query
block's 64-token neighborhood.

The RK1 runtime needs only the binary, vendor runtime, model snapshot,
compiled graphs, and logs under `/mnt/warm`. Rust and Cargo remain on the
workstation or GitHub runner. No model binary or Rockchip library is committed
to Git. On the graph-build host, `env-rk1.sh` redirects Python, uv, pip,
Torch, Hugging Face, and temporary files to the warm build directory. The
converter creates a unique working directory because Toolkit2 writes
intermediate ONNX files in its current directory; it removes them afterward.

The staged Toolkit2 wheel and runtime came from official Rockchip repository
commit `59a913d172e7f5ff03c9076e2ec7b1b1288ffd08`:

| Artifact | SHA-256 |
| --- | --- |
| Toolkit2 2.3.2 ARM64 Python 3.10 wheel | `73d0bb55d30ee00bbd9603331a4118da73a4e12fa9d9196200460d07d67c6a12` |
| `librknnrt.so` | `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8` |

The tested RK1 reported RKNN API 2.3.2 and RKNPU driver 0.9.2. Python is used
only for offline oracle export and graph compilation. The Rust executable
loads the staged runtime dynamically. On that RK1, graphs and retained CPU
traces used about 13 GB. After installing released binaries and removing the
node's Rust toolchain/build caches, `/mnt/warm` had about 5.8 GB free and the
eMMC root about 12 GB free.
The service uses about 263 MB RSS before loading buckets and about 2.4 GB RSS
with all three long-request buckets cached. The node has 31 GiB RAM.

## Build and deploy

The resumable build scripts export each layer and head at every bucket,
check each graph against a CPU oracle, convert it with Toolkit2, and remove
regenerable ONNX intermediates. Run these commands on an **ARM64 Linux build
host**, with Laya/Torch in one Python environment and Rockchip Toolkit2 2.3.2
in another. Point `LAYA_MODEL_SNAPSHOT` at the pinned checkpoint tree. The
tested scripts write all graph files and caches under `NPU_ROOT`.

```sh
export NPU_ROOT=/mnt/warm/laya-build
export LAYA_NPU_ENV="$PWD/npu/env-rk1.sh"
export LAYA_MODEL_SNAPSHOT=/path/to/pinned/laya-checkpoint
export LAYA_CPU_PYTHON=/path/to/laya-torch-venv/bin/python
export LAYA_RKNN_PYTHON=/path/to/rknn-toolkit2-venv/bin/python
mkdir -p "$NPU_ROOT/scripts"
rsync -a npu/scripts/ "$NPU_ROOT/scripts/"
bash "$NPU_ROOT/scripts/build_full_npu.sh" english
bash "$NPU_ROOT/scripts/build_full_npu.sh" multilingual
bash "$NPU_ROOT/scripts/build_full_npu.sh" typed-decisions
bash "$NPU_ROOT/scripts/build_onehot_heads.sh" english
bash "$NPU_ROOT/scripts/build_onehot_heads.sh" multilingual
bash "$NPU_ROOT/scripts/build_onehot_heads.sh" typed-decisions
bash "$NPU_ROOT/scripts/build_encoder_blocks.sh" english 512 14
bash "$NPU_ROOT/scripts/build_encoder_blocks.sh" multilingual 1024 14
bash "$NPU_ROOT/scripts/build_encoder_blocks.sh" typed-decisions 1024 14
bash "$NPU_ROOT/scripts/build_fused_bucket.sh" multilingual 768
bash "$NPU_ROOT/scripts/build_fused_bucket.sh" typed-decisions 640
bash "$NPU_ROOT/scripts/build_fused_bucket.sh" typed-decisions 768
bash "$NPU_ROOT/scripts/build_encoder_blocks.sh" multilingual 768 14 256
bash "$NPU_ROOT/scripts/build_encoder_blocks.sh" typed-decisions 640 14 256
bash "$NPU_ROOT/scripts/build_encoder_blocks.sh" typed-decisions 768 14 256
```

Copy only the compiled graphs to the RK1's warm drive; oracle traces and build
caches can stay on the build host. Stage `librknnrt.so`, the pinned model
snapshot, and a configured copy of [`runtime.env.example`](runtime.env.example)
there as well. From this repository on the workstation, install the downloaded
release binary without running Cargo on the RK1:

```sh
rsync -a --include='*/' --include='*.rknn' --exclude='*' \
  "$NPU_ROOT/models/" ubuntu@RK1_HOST:/mnt/warm/laya-service/npu-backend/models/
export LAYA_RK1_NODE=ubuntu@RK1_HOST
./npu/deploy-rk1.sh /path/to/extracted/laya-rknpu
```

`deploy-rk1.sh` copies only the binary and runtime scripts, restarts the full
NPU service on port 8003, and checks health. The transient systemd unit can be
started after reboot with `/mnt/warm/laya-service/npu-backend/start-full-rk1.sh`.
`LAYA_NPU_WINDOW_QUERY=256` enables tiled graphs where a complete set exists;
the loader uses the standard fused graphs for other buckets. The old seven-layer
English and typed 1024-token binaries were removed to keep warm-drive space
available; `build_encoder_blocks.sh` can regenerate them.

## Oracle accuracy

All ten root API fixtures and all six Mali API fixtures kept the same selected
answers and routing as the CPU oracle, with numeric differences below 0.01.
Additional long requests also kept their selected answers: typed 639 tokens
had maximum numeric difference 0.0020, multilingual 726 had 0.0049, and a
512-token English stress request had 0.0163. RKNN executes the heavy graph in
FP16, so close decisions can change; use the CPU or Mali service where
CPU-equivalent confidence is required.

The latest [side-by-side CPU/GPU/NPU benchmark](benchmarks/README.md) covers
seven matched requests, from 48 to 726 tokens. Each path received one warm-up
call and three measured calls per case, with rotated run order. The NPU was
fastest on every case: warm medians ranged from 0.122 s for short multilingual
to 2.253 s for typed 639 tokens. Switching a length bucket adds graph-loading
time; all three large buckets fit in memory together. An NPU allocation failure
clears other checkpoints' caches and retries once.

`trace-full` checks head states, logits, and action logits against exact-length
CPU traces. Set `LAYA_ORACLE_DIR` to print error after every encoder layer.
The 1024-token multilingual trace showed mean encoder error growing from
0.0055 at layer 0 to 0.36 at layer 21; the FP16 decision head amplifies that
error. The flash-attention Toolkit2 flag did not improve the profiled
1024-token layer.

## Profiling

`LAYA_NPU_TIMING=1` is enabled in `serve-full-rk1.sh`. It writes prompt
preparation, host embedding lookup, RKNN wall and device time for every layer,
the decision head, scoring, and action inference to
`/mnt/warm/laya-service/npu-backend/logs/full-server.log`. Summarize the live
log with:

```sh
ssh ubuntu@RK1_HOST 'source /mnt/warm/laya-service/npu-backend/env.sh; "$NPU_ROOT/venv/bin/python" "$NPU_ROOT/scripts/summarize_profile.py" "$NPU_ROOT/logs/full-server.log" --model typed --bucket 640 --tail-lines 200'
```

Set `LAYA_RKNN_PROFILE=1` before loading a graph for Rockchip's detailed
operator tables; use it for a trace, since collecting them reduces throughput.
On a 1024-token sliding layer, `exSDPAttention` used 83 ms (48% of operator
time), transposes 26 ms (15%), and dense projections 37 ms (22%). Direct
FP16 inter-layer transfers preserved oracle output and cut typed 1024-token
encoder host overhead from 2.36 s to 1.72 s in the measured trace.
Before internal attention tiling, typed encoder execution took 2.64 s of
2.72 s on the NPU at 768 tokens. Detailed operator profiling put attention at
1.282 s (48.5%) and dense `Conv` plus `ConvAdd` at about 0.824 s (31%).
After tiling at 768, attention fell to 0.876 s (about 38%); dense projection
stayed near 0.824 s (about 35%). At 640 tokens, attention takes 0.648 s (35%)
and dense projection 0.685 s (37%). The node
reported 100% NPU load at 1 GHz during a long request. Summarize detailed
traces with `scripts/summarize_operator_profile.py`.

## Optimization record

The Rust runtime records RKNN input, run, output, copy, and release time for
each call when `LAYA_RKNN_PHASES=1`. `summarize_phases.py` aggregates a trace.
`block_probe` runs alternating fused and split graphs on the same input and
checks their FP16 outputs bit for bit. The iterations below used the typed
1024-token bucket unless stated otherwise.

| Iteration | Measured bottleneck | Change and result |
| --- | --- | --- |
| Input transfer | 28 separate encoder calls spent 1.626 s in `rknn_inputs_set`; device runs took 4.699 s. | A shared device mask reduced input time to 0.525 s but increased device time to 6.081 s. Rejected. An FP16 mask path was also slower. |
| Attention | A single sliding layer spent 83 ms (48% of device operator time) in `exSDPAttention`, 26 ms in transposes, and 37 ms in dense projections. | Running whole-layer graphs on overlapping windows improved oracle error but took 245–289 ms versus 223–227 ms for one full layer. Rejected; the later internal tiling computes projections only once. |
| Graph boundaries | Separate 1024-token layers cost roughly 6.4 s for the typed encoder. | Four seven-layer graphs cut the measured encoder wall time to roughly 4.4–4.8 s with the same RKNN output. The multilingual and English long buckets also improved. Superseded by fourteen-layer graphs. |
| Model reloads | Alternating short English, Spanish, and typed requests repeatedly reloaded graphs. | Retaining one bucket per checkpoint cut median responses after the first round from 2.006/1.021/1.502 s to 0.229/0.118/0.237 s. Kept, with allocation-error eviction and retry. |
| Device execution | After seven-layer fusion, the typed encoder spent about 4.1 s of 4.4 s on the NPU. | Two fourteen-layer graphs cut encoder time to 4.10–4.17 s in the paired full traces. Their FP16 output and final logits matched the seven-layer path exactly. Kept. |
| Padding | A 768-token input ran in a 1024-token graph. | Compiling a 768-token bucket cut typed encoder time from 4.13–4.17 s to 2.73–2.78 s and multilingual from 2.21–2.25 s to 1.35–1.47 s for the same inputs. Kept. |
| Sliding attention inside one graph | Attention consumed 1.282 s, about 49% of typed 768-token encoder operator time. | Query tiles with a 64-token key halo cut attention to 0.876 s. Warm typed 639-token API latency fell from 3.092 to 2.661 s and multilingual 726 from 1.625 to 1.446 s. Both retained the CPU-selected answer and improved their numeric agreement. Kept. |
| Core assignment | The windowed graph spent about 2.35 s in the typed 768-token encoder with all three NPU cores. | Two cores took 2.70 s and one took 3.68 s. All three cores remain enabled. |
| Residual padding | A 639-token typed request still ran the 768-token graph. | A 640-token tiled graph cut same-input encoder time from 2.34–2.36 s to 1.89–1.99 s. Warm API median fell from 2.661 to 2.165 s, with the same selected answer and maximum numeric difference improving from 0.0027 to 0.0020. Kept. |

On an isolated 1024-token sliding layer, internal 256-token query tiles cut
median wall time from about 212 to 164 ms and maximum CPU-oracle error from
0.738 to 0.207. Smaller 128-token tiles gave a modest, less stable speed gain
and were not deployed. `single_graph_probe` and `block_probe` make those
comparisons reproducible.

Binding a graph's output memory directly to the next graph's input returned
zero output with this runtime, so it is not used. The remaining long-request
latency is mostly NPU computation, especially attention and dense projection.
The probes live in `src/bin/` and the API still uses standard RKNN execution.
An isolated weight-only quantization test was rejected by Toolkit2 before
conversion: its `w8a16` format is unsupported for RK3588 in version 2.3.2.
