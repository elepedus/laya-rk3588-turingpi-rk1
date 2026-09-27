"""Export final encoder norm, type embedding, and decision Transformer head."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from laya.agent import Agent


class HeadExport(torch.nn.Module):
    def __init__(self, model, onehot=False):
        super().__init__()
        self.final_norm = model.encoder.final_norm
        self.type_emb = model.type_emb
        self.layers = model.head.layers
        self.onehot = onehot

    def forward(self, hidden, attention_mask, qtype_input):
        hidden = self.final_norm(hidden)
        type_vector = (qtype_input @ self.type_emb.weight if self.onehot
                       else self.type_emb(qtype_input.long()))
        hidden = hidden + type_vector[:, None, :]
        for layer in self.layers:
            hidden = layer(hidden, src_mask=attention_mask)
        return hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("oracle_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--onehot", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    agent = Agent(str(args.model_dir), device="cpu")
    model = agent.model.eval()
    dim = model.encoder.config.hidden_size
    last = model.encoder.config.num_hidden_layers - 1
    input_data = np.fromfile(args.oracle_dir / f"encoder_{last:02d}.f32", dtype="<f4")
    expected = np.fromfile(args.oracle_dir / f"head_{len(model.head.layers)-1:02d}.f32", dtype="<f4")
    valid = input_data.size // dim
    assert input_data.size == expected.size == valid * dim and valid <= args.length
    hidden = torch.zeros((1, args.length, dim), dtype=torch.float32)
    hidden[:, :valid] = torch.from_numpy(input_data.reshape(1, valid, dim).copy())
    mask = torch.zeros((args.length, args.length), dtype=torch.float32)
    mask[:, valid:] = -10000.0
    qtype = (torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
             if args.onehot else torch.zeros((1,), dtype=torch.float32))
    with torch.no_grad():
        wrapper = HeadExport(model, args.onehot).eval()
        actual = wrapper(hidden, mask, qtype).numpy()
    delta = np.abs(actual[:, :valid].reshape(-1) - expected)
    print("valid", valid, "length", args.length,
          "head_oracle_max_abs", float(delta.max()), flush=True)
    if delta.max() > 0.001:
        raise ValueError("head export differs from CPU oracle")
    stem = "head_onehot" if args.onehot else "head"
    for name, array in (("hidden", hidden), ("mask", mask),
                        ("qtype_onehot" if args.onehot else "qtype", qtype),
                        ("expected_onehot" if args.onehot else "expected", torch.from_numpy(actual))):
        array.numpy().astype("<f4").tofile(args.output_dir / f"{name}.f32")
    destination = args.output_dir / f"{stem}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (hidden, mask, qtype), str(destination),
            opset_version=17,
            input_names=["hidden", "attention_mask", "qtype"],
            output_names=["head_hidden"],
            do_constant_folding=True, dynamo=False,
        )
    graph = onnx.load(destination, load_external_data=False)
    onnx.checker.check_model(graph)
    print("onnx_bytes", destination.stat().st_size,
          "ops", dict(Counter(node.op_type for node in graph.graph.node)), flush=True)


if __name__ == "__main__":
    main()
