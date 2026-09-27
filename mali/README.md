# Laya on the Turing RK1 Mali GPU

For a new RK3588/Turing Pi RK1 installation, use the prebuilt ARM64
[`laya-mali` release](https://github.com/elepedus/laya-rk3588-turingpi-rk1/releases)
and the [runtime setup](../README.md). Build Rust on a workstation or GitHub
ARM runner; Rust and Cargo are not required on the RK1 to run a release binary.

This is a Rust/OpenCL inference backend for the three checkpoints in
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).
Rust constructs the prompt, tokenizes it, selects a checkpoint, runs its
ModernBERT encoder and two decision-head layers on the Mali-G610, then decodes
typed answers on the CPU. The OpenCL kernels for FP16-weight linear layers,
normalization, RoPE, attention, activations, and residuals are in
[`src/kernels.cl`](src/kernels.cl). No Python, PyTorch, ONNX Runtime, or CLBlast
is used by this server at inference time.

An RK1 service bound at `http://RK1_LAN_IP:8002` implements
`GET /health` and `POST /v1/systemone`. The original Python CPU server remains
at port 8000. All three checkpoints preload, including typed-decisions, so an
explicit `"model": "typed-decisions"` request avoids a first-use load. The server
binds the LAN address with no API key, as requested, and handles requests one
at a time.

While the GPU service runs, its startup script selects Mali's `always_on`
power policy and warms the device before accepting requests. The transient
systemd unit restores `coarse_demand` when stopped. This avoids the roughly
2.4-second first-dispatch delay observed after GPU idle; it may increase idle
power draw. This node exposes no GPU wattage sensor, so that cost is unmeasured.

The RK1 runtime needs only the release binary, matching Mali libraries and
firmware, the Laya model snapshot, scripts, and logs under `/mnt/warm`. The
tested installation left the eMMC root untouched. Build tools and Cargo caches
remain off the RK1. The firmware search path is a temporary kernel setting;
it must be set again after reboot.

## Run and inspect

```sh
ssh ubuntu@RK1_HOST /mnt/warm/laya-service/mali-backend/start-rk1.sh
curl http://RK1_LAN_IP:8002/health
curl http://RK1_LAN_IP:8002/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was billed twice; please refund me","questions":{"billing":{"type":"noul","instructions":"Does this request concern billing?"}}}'
```

Before starting, copy and edit [`runtime.env.example`](runtime.env.example)
as `/mnt/warm/laya-service/mali-backend/runtime.env`. `start-rk1.sh` is safe to
run when the service is already active. It accepts `LAYA_MALI_FIRMWARE` and an
optional `LAYA_MALI_FIRMWARE_SHA256`, sets the kernel firmware lookup to the
warm-drive directory, and starts a transient systemd unit. The unit, firmware setting,
and power-policy change do not survive reboot. Stop it with
`sudo systemctl stop laya-mali`; the stop hook restores `coarse_demand`.

The node's kernel requires firmware header `0.3`. The older G13/G15 firmware
from the configured Ubuntu PPA failed with `Firmware header version 0.1/0.2 not
understood`. The staged `mali_csffw.bin` came from the
[Rockchip libmali mirror](https://github.com/JeffyCN/mirrors/tree/libmali/firmware/g610),
and the tested firmware SHA-256 was
`60ffa376edec8c402dc0ee9357c685d835ec2e0fb3d0778f303d08a9802d57f1`.
The Mali G610 userspace was extracted
from the node's configured `libmali-g610-x11` package on `/mnt/warm`, together
with its four missing shared-library dependencies. No package was installed
system-wide.

## Build off the RK1

The GitHub Release already contains the ARM64 binary. To build from source on
an ARM64 Linux workstation instead, use the pinned Rust toolchain and lockfile:

```sh
cargo build --locked --release --manifest-path mali/Cargo.toml --bin laya-mali
scp mali/target/release/laya-mali \
  ubuntu@RK1_HOST:/mnt/warm/laya-service/mali-backend/bin/
```

`LAYA_MALI_OPENCL` must point to the staged `libOpenCL.so.1`, and
`LD_LIBRARY_PATH` must include both the staged library directories. The
deployed `serve-rk1.sh` sets these. The Rust binary also supports `predict`
(one JSON request on standard input), `selftest`, `trace`, and `bench-linear`
modes. `bench-linear` compares the original, four-column, and eight-column
matrix layouts on a real Wi tensor.

## Oracle checks and speed

The synthetic oracle trace in `oracle_export.py` captures embeddings, every
encoder layer, both head layers, and final logits. On the original scalar
path, logits for a 51-token English request differed from the CPU oracle by at
most `9e-6`; on a 48-token multilingual trace, at most `4e-6`. The API matched
the Python
server's four-decimal answers, routing, and token counts for English, Spanish,
Hindi, typed-decisions, score, yes/no, and a 12-token padded input. Run
`python3 mali/compare_oracle.py` while both services are up to repeat the API
comparison.

The six-fixture LAN oracle check after the fourth optimization produced these
wall times. Both Mali columns use packed eight-column weights and `always_on`;
the current version also vectorizes attention score dots and layer norms.
These are individual observations and vary with node activity.

| Fixture | CPU Python | Previous Mali | Current Mali |
| --- | ---: | ---: | ---: |
| English choice, 48 tokens | 2.68 s | 0.62 s | 0.52 s |
| Spanish choice, 58 tokens | 1.07 s | 0.25 s | 0.21 s |
| Typed decisions choice, 58 tokens | 3.53 s | 0.54 s | 0.49 s |
| Score + yes/no, 101 input tokens | 4.59 s | 0.96 s | 0.87 s |
| Short choice, 12 tokens | 1.57 s | 0.41 s | 0.25 s |
| Hindi choice, 60 tokens | 1.74 s | 0.41 s | 0.24 s |

Five further English requests separated by five seconds of idle time took
0.441–0.465 s (median 0.456 s), down from the previous median of 0.509 s.

## Profiling and optimization

Set `LAYA_MALI_PROFILE=1` for a one-shot `predict` run to print OpenCL kernel
times, GPU event span, and host call times. This uses the
[Khronos event profiling API](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetEventProfilingInfo.html).
Set `LAYA_MALI_TIMING=1` for wall timing without event profiling.

On the 48-token English fixture, the original scalar backend spent 12.61 s of
12.71 s GPU execution time in dense linear operations (11.70 s in the four
encoder matrix classes). Reading eight adjacent FP16 weights and FP32 input
values per loop cut GPU execution to 2.19 s. Reusing each weight vector across
four input rows cut it to 0.94–1.05 s.
Tests of 16-wide vectors, paired output columns, and eight-row reuse were
slower and were removed from the deployed code.

After that change, English still spent 0.98 s of GPU event time in 390 kernels;
Wi, Wqkv and Wo projections dominated. A representative Wi projection took
about 13–30 ms with its original weight layout, versus 6–7 ms when transposed
to `[input_feature, output_column]`; outputs differed by at most `7e-6`.
The deployed backend packs 120 English and 96 multilingual matrices during
preload, plus the typed-decisions matrices. GPU event time fell to 0.54 s for
English and 0.26 s for multilingual. Preloading all three takes longer at
service startup (9.02 s measured) but removes the observed 4.00-second first
typed request; the first typed request after restart took 0.73 s.

Reprofiling the packed backend showed the four encoder projection classes
still used about 388 ms of 507 ms English GPU execution; host allocation and
enqueue calls took about 45 ms. Changing the packed kernel from four to eight
adjacent output columns cut its real Wi matrix from 5.83–6.45 ms to
3.39–3.82 ms, with a `7e-6` maximum difference from the original layout.
Full-model GPU event time fell from 0.508 to 0.388 s for English, 0.248 to
0.196 s for multilingual, and 0.586 to 0.452 s for typed-decisions. The
eight-column kernel is the deployed default.

Reprofiling after eight-column packing showed matrix projections still used
about 274 of 389 ms English GPU time, while attention scores used 37 ms and
normalization 33 ms. Vectorizing the 64-value attention dot product cut its
English total from 36.9 to 9.0 ms (multilingual 29.7 to 6.8 ms). Vectorizing
layer norm cut its English total from 32.6 to 9.2 ms (multilingual 19.7 to
5.8 ms). Combined, warm English GPU event time fell from 388 to 337 ms and
host inference from 475 to 418 ms; multilingual fell from 196 to 159 ms GPU
and 262 to 223 ms host. A four-step matrix unroll and a simple one-work-item
fused-attention kernel were tested and were slower.

A profiled four-row request spent about 2.8 s in GPU completion. A separate
`clFinish` experiment moved that wait out of `clEnqueueReadBuffer`; the final
transfer then took 0.3 ms. Earlier queue flushes did not improve wall time.
The first dispatch in a fresh process under `coarse_demand` took 2.4–2.6 s;
with `always_on` and a prior warmup, repeated packed-model inference took
about 0.6 s for English and 0.3 s for multilingual. Fusion of Wi and SwiGLU
and a float4 weight kernel were also tested and were slower.

Requests execute each question separately; the GPU backend does not yet batch
questions. Inputs are capped at 4,000 state characters, 16 questions, and 32
options per question. It uses each checkpoint's configured limit (512 tokens
for English, 1,024 for multilingual and typed-decisions); the multilingual
8,192-token override is not implemented. As with the upstream checkpoint,
confidence values affected by invalid stored temperatures should be treated
as uncalibrated.

During a pre-optimization 51-token English request, Mali utilization had a
99% median across 54 samples and the Rust service used 0.15 CPU seconds.
This confirms the model's heavy work ran on the GPU.
