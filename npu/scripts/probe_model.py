"""Compile a small deterministic MatMul + Add graph for the RK3588 NPU."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from convert_onnx import compile_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x = (np.arange(16, dtype=np.float32).reshape(1, 16) - 7) / 8
    w = ((np.arange(16 * 32, dtype=np.float32).reshape(16, 32) % 23) - 11) / 32
    bias = (np.arange(32, dtype=np.float32) - 15) / 64
    expected = x @ w + bias

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weight"], ["projected"]),
            helper.make_node("Add", ["projected", "bias"], ["output"]),
        ],
        "laya_npu_probe",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32])],
        [numpy_helper.from_array(w, "weight"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx_path = args.output_dir / "linear_probe.onnx"
    onnx.save(model, onnx_path)

    compile_model(onnx_path, args.output_dir / "linear_probe.rknn")

    (args.output_dir / "linear_probe.json").write_text(
        json.dumps({"input": x.reshape(-1).tolist(), "expected": expected.reshape(-1).tolist()}, indent=2)
    )
    print(f"compiled {args.output_dir / 'linear_probe.rknn'}")


if __name__ == "__main__":
    main()
