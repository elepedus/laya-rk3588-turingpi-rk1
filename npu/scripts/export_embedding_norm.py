"""Export the embedding layer norm; token table lookup is supplied by Rust."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from laya.agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("oracle_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--length", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    agent = Agent(str(args.model_dir), device="cpu")
    model = agent.model.eval()
    ids = json.loads((args.oracle_dir / "input.json").read_text())["ids"]
    if len(ids) > args.length:
        raise ValueError("oracle input does not fit graph")
    padded = ids + [model.encoder.config.pad_token_id] * (args.length - len(ids))
    with torch.no_grad():
        source = model.encoder.embeddings.tok_embeddings(
            torch.tensor([padded], dtype=torch.long))
        actual = model.encoder.embeddings.norm(source)
    expected = np.fromfile(args.oracle_dir / "embedding.f32", dtype="<f4")
    delta = np.abs(actual[:, :len(ids)].numpy().reshape(-1) - expected)
    print("tokens", len(ids), "length", args.length,
          "oracle_max_abs", float(delta.max()), flush=True)
    if delta.max() > 0.001:
        raise ValueError("embedding norm differs from CPU oracle")
    source.numpy().astype("<f4").tofile(args.output_dir / "gathered.f32")
    actual.numpy().astype("<f4").tofile(args.output_dir / "expected.f32")
    destination = args.output_dir / "embedding_norm.onnx"
    with torch.no_grad():
        torch.onnx.export(model.encoder.embeddings.norm, (source,), str(destination),
                          opset_version=17, input_names=["gathered"],
                          output_names=["hidden"], do_constant_folding=True,
                          dynamo=False)
    graph = onnx.load(destination, load_external_data=False)
    onnx.checker.check_model(graph)
    print("onnx_bytes", destination.stat().st_size,
          "ops", [node.op_type for node in graph.graph.node], flush=True)


if __name__ == "__main__":
    main()
