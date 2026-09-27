# Laya on Turing Pi RK1 (RK3588)

Rust inference backends for the [Laya decision model](https://huggingface.co/convaiinnovations/laya)
on the **RK3588 chipset in Turing Pi RK1 modules**. This repository contains
two hardware paths for the same `POST /v1/systemone` API.

**Measured on one RK1, warmed end-to-end response time (median of three):**

| Request | Tokens | CPU reference | Mali GPU | RK3588 NPU |
| --- | ---: | ---: | ---: | ---: |
| English short | 48 | 2.464 s | 0.409 s | **0.221 s** |
| Multilingual short | 58 | 0.950 s | 0.213 s | **0.122 s** |
| Typed short | 58 | 2.991 s | 0.467 s | **0.257 s** |
| English medium | 154 | 4.833 s | 1.233 s | **1.076 s** |
| English maximum | 512 | 8.073 s | 6.137 s | **1.738 s** |
| Multilingual long | 726 | 6.160 s | 5.394 s | **1.381 s** |
| Typed long | 639 | 8.790 s | 8.827 s | **2.253 s** |

The CPU column is the upstream Laya server on the same RK1. All three paths
selected the same answers. Each path received a warm-up call; measured
requests ran sequentially with rotated port order. The [benchmark report and
raw samples](npu/benchmarks/README.md) show the method and numeric differences.

## Backends

| Executable | Engine | Additional runtime files |
| --- | --- | --- |
| `laya-rknpu` | RK3588 NPU | Laya safetensors, compiled RKNN graphs, `librknnrt.so` |
| `laya-mali` | Mali-G610 OpenCL GPU | Laya safetensors, matching Mali OpenCL driver and firmware |

Both release binaries target **ARM64 Ubuntu 22.04**, the tested RK1 system.
The CPU reference implementation is maintained upstream and is not included
here.

## Download a release

Open [Releases](https://github.com/elepedus/laya-rk3588-turingpi-rk1/releases)
and download the archive for your engine. Verify it against `SHA256SUMS`, then
extract the executable onto the RK1's warm drive:

```sh
mkdir -p /mnt/warm/laya/bin
tar -xzf laya-rknpu-linux-aarch64-glibc2.35.tar.gz -C /mnt/warm/laya/bin
```

**Rust and Cargo are not needed on the RK1 to run a release binary.**
[GitHub Actions](.github/workflows/rust.yml) builds and tests both binaries
on native Ubuntu 22.04 ARM64 runners. A pushed `v*` tag publishes the binaries
and checksums as a GitHub Release.

Release archives contain executable code and the Apache 2.0 license. Model
weights, RKNN graphs, Rockchip runtime libraries, and Mali drivers are
separate. They are not embedded in the binaries or stored in this Git repo.

## Prepare the model files

Download [the Laya model](https://huggingface.co/convaiinnovations/laya) at
revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. The repository root
contains English; `multilingual/` and `typed-decisions/` contain the other
checkpoints. Both Rust backends read the safetensors, tokenizer, and config
files from this tree. Downloading on a workstation and copying the snapshot to
`/mnt/warm` avoids Python install caches on the RK1. One way to do that on a
workstation is:

```sh
python3 -m pip install huggingface_hub
hf download convaiinnovations/laya \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 \
  --local-dir ./laya-checkpoint
rsync -a ./laya-checkpoint/ ubuntu@RK1_HOST:/mnt/warm/laya/checkpoints/laya/
```

The NPU also requires fixed-shape `.rknn` files under
`LAYA_RKNN_GRAPH_ROOT`. The [NPU guide](npu/README.md) lists the tested graph
layout, Rockchip Toolkit2 version, and resumable conversion scripts. Convert
the graphs on a build machine, then copy only their output and the Rockchip
runtime library to the RK1. The [Mali guide](mali/README.md) documents the
matching OpenCL driver and firmware required by that binary.

## Run on an RK1

Set paths for your installation. This NPU example binds port 8003 to the LAN;
use `127.0.0.1:8003` for a local-only service:

```sh
export LAYA_MODEL_DIR=/mnt/warm/laya/checkpoints/laya
export LAYA_RKNN_GRAPH_ROOT=/mnt/warm/laya/rknn-models
export LAYA_RKNNRT=/mnt/warm/laya/lib/librknnrt.so
export LAYA_NPU_FULL_BIND=0.0.0.0:8003
export LAYA_NPU_PRELOAD=english,multilingual,typed-decisions
export LAYA_NPU_CACHE_BUCKETS=1
export LAYA_NPU_KEEP_CACHES=1
export LAYA_NPU_FUSED_BLOCKS=1
export LAYA_NPU_FUSED_GROUP=14
export LAYA_NPU_WINDOW_QUERY=256
/mnt/warm/laya/bin/laya-rknpu serve-full
```

For a managed NPU service, copy and edit
[`npu/runtime.env.example`](npu/runtime.env.example) as
`/mnt/warm/laya-service/npu-backend/runtime.env` on the RK1. From this repo on
your workstation, deploy the extracted release binary:

```sh
LAYA_RK1_NODE=ubuntu@RK1_HOST \
  ./npu/deploy-rk1.sh /path/to/extracted/laya-rknpu
```

The deploy helper does not install or run Cargo on the RK1.

For Mali, set `LAYA_MODEL_DIR`, set `LAYA_MALI_OPENCL` to the installed
`libOpenCL.so.1`, set `LAYA_MALI_BIND=0.0.0.0:8002`, and run
`/mnt/warm/laya/bin/laya-mali serve`. The [Mali guide](mali/README.md) covers
driver, firmware, power-policy setup, and its
[`runtime.env.example`](mali/runtime.env.example).

Both services expose `GET /health` and `POST /v1/systemone`:

```sh
curl http://RK1_LAN_IP:8003/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"We were billed twice for March.","questions":{"billing":{"type":"noul","instructions":"Is this about billing?"}}}'
```

The services have no built-in API key. Bind them to a trusted network or put
authentication in front of them. Rust handles tokenization, vocabulary lookup,
and response assembly; the NPU runs its supported neural operators in FP16, so
confidence values can differ from the CPU reference.

## Build and release from source

On an ARM64 Linux workstation, install Rust 1.98.1 and build either independent
crate. The pinned lockfiles and CI use the same commands:

```sh
cargo build --locked --release --manifest-path mali/Cargo.toml --bin laya-mali
cargo build --locked --release --manifest-path npu/Cargo.toml --bin laya-rknpu
```

The CI uses GitHub's Ubuntu 22.04 ARM runners, matching the RK1's glibc 2.35
baseline. After CI passes, push a `v*` tag to publish two ARM64 archives and
`SHA256SUMS`. The binary archives require no Rust installation on the RK1.

## Accuracy, benchmarks, and license

The [NPU runbook](npu/README.md) records oracle checks, bucket limits, and
profiling results. The [side-by-side benchmark](npu/benchmarks/README.md)
includes the method and raw timings for the upstream CPU, Mali, and NPU paths.

This code is licensed under [Apache 2.0](LICENSE). The upstream
[Laya model card](https://huggingface.co/convaiinnovations/laya) lists the model
under the same license. This is an independent inference backend; model
weights and hardware runtimes remain upstream projects.
