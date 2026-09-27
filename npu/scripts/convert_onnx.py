"""Compile a fixed-shape ONNX graph for the RK3588 NPU."""

import argparse
import os
import tempfile
from pathlib import Path

from rknn.api import RKNN


def compile_model(source: Path, destination: Path, **config):
    optimization_level = config.pop("optimization_level", 3)
    source = source.resolve(strict=True)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(os.environ["TMPDIR"]).resolve(strict=True)
    if not temporary_root.is_relative_to("/mnt/warm"):
        raise ValueError("RKNN conversion TMPDIR must be on /mnt/warm")
    previous_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="rknn-convert-", dir=temporary_root) as work:
        try:
            os.chdir(work)
            compiler = RKNN(verbose=True)
            try:
                assert compiler.config(target_platform="rk3588", optimization_level=optimization_level,
                                       **config) == 0
                assert compiler.load_onnx(model=str(source)) == 0
                assert compiler.build(do_quantization=False) == 0
                assert compiler.export_rknn(str(destination)) == 0
            finally:
                compiler.release()
        finally:
            os.chdir(previous_directory)
    print(f"compiled {source} -> {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--flash-attention", action="store_true")
    args = parser.parse_args()
    compile_model(args.source, args.destination,
                  **({"enable_flash_attention": True} if args.flash_attention else {}))


if __name__ == "__main__":
    main()
