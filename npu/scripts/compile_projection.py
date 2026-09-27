"""Compile one pinned Laya linear projection and write a deterministic oracle."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open
from convert_onnx import compile_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--weight", default="encoder.layers.0.mlp.Wi.weight")
    parser.add_argument("--rows", type=int, default=51)
    parser.add_argument("--layout", choices=("matmul", "conv"), default="matmul")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with safe_open(args.model_dir / "model.safetensors", framework="pt", device="cpu") as weights:
        matrix = weights.get_tensor(args.weight).numpy().astype(np.float16)
    output_features, input_features = matrix.shape
    x = ((np.arange(args.rows * input_features, dtype=np.float32).reshape(args.rows, input_features) * 17) % 101 - 50) / 32
    expected = x @ matrix.astype(np.float32).T
    if args.layout == "conv":
        input_array = x.T.reshape(1, input_features, args.rows, 1)
        expected_array = expected.T.reshape(1, output_features, args.rows, 1)
    else:
        input_array = x
        expected_array = expected
    # RKNN's input API accepts NHWC source buffers for an NCHW model.
    # Flattened x is [row, feature, 1], so this is the buffer we pass at runtime.
    (x if args.layout == "conv" else input_array).astype("<f4").tofile(args.output_dir / "input.f32")
    expected_array.astype("<f4").tofile(args.output_dir / "expected.f32")
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "weight": args.weight,
        "input_shape": [args.rows, input_features],
        "output_shape": [args.rows, output_features],
        "layout": args.layout,
        "expected_min": float(expected.min()),
        "expected_max": float(expected.max()),
    }, indent=2))

    if args.layout == "conv":
        nodes = [helper.make_node("Conv", ["input", "weight"], ["output"], kernel_shape=[1, 1])]
        input_shape = [1, input_features, args.rows, 1]
        output_shape = [1, output_features, args.rows, 1]
        weight = matrix[:, :, None, None].copy()
    else:
        nodes = [helper.make_node("MatMul", ["input", "weight"], ["output"])]
        input_shape = [args.rows, input_features]
        output_shape = [args.rows, output_features]
        weight = matrix.T.copy()
    graph = helper.make_graph(
        nodes,
        "laya_projection",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, output_shape)],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx_path = args.output_dir / "projection.onnx"
    onnx.save(model, onnx_path)

    compile_model(onnx_path, args.output_dir / "projection.rknn")
    print(f"compiled {args.weight}: {args.rows} × {input_features} → {output_features}")


if __name__ == "__main__":
    main()
