"""Export one pinned ModernBERT encoder layer with fixed sequence length."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from laya.agent import Agent


class LayerExport(torch.nn.Module):
    def __init__(self, layer, cos, sin, layout):
        super().__init__()
        self.layer = layer
        self.layout = layout
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, hidden, attention_mask):
        if self.layout == "nchw":
            hidden = hidden.squeeze(-1).transpose(1, 2)
        output = self.layer(hidden, attention_mask=attention_mask, position_embeddings=(self.cos, self.sin))
        return output.transpose(1, 2).unsqueeze(-1) if self.layout == "nchw" else output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("oracle_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--layout", choices=("sequence", "nchw"), default="sequence")
    parser.add_argument("--length", type=int, default=None, help="fixed sequence length; pad the oracle input")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    agent = Agent(str(args.model_dir), device="cpu")
    model = agent.model.eval()
    model.encoder.config._attn_implementation = "eager"
    layer = model.encoder.layers[args.layer]
    layer.config._attn_implementation = "eager"
    input_name = "embedding" if args.layer == 0 else f"encoder_{args.layer-1:02d}"
    expected_name = f"encoder_{args.layer:02d}"
    dim = model.encoder.config.hidden_size
    inputs = np.fromfile(args.oracle_dir / f"{input_name}.f32", dtype="<f4")
    expected = np.fromfile(args.oracle_dir / f"{expected_name}.f32", dtype="<f4")
    valid = inputs.size // dim
    assert inputs.size == expected.size == valid * dim
    seq = args.length or valid
    if seq < valid:
        raise ValueError("fixed sequence length must cover the oracle input")
    padded = np.zeros((1, seq, dim), dtype=np.float32)
    padded[:, :valid, :] = inputs.reshape(1, valid, dim)
    hidden = torch.from_numpy(padded)
    model_input = hidden.transpose(1, 2).unsqueeze(-1) if args.layout == "nchw" else hidden
    padded.astype("<f4").tofile(args.output_dir / "input_nhwc.f32")
    if seq > 64:
        mask = torch.zeros((1, 1, seq, seq), dtype=torch.float32)
        mask[..., valid:] = -10000.0
        if layer.attention_type == "sliding_attention":
            positions = torch.arange(seq)
            half_window = model.encoder.config.local_attention // 2
            outside = (positions[:, None] - positions[None, :]).abs() > half_window
            mask[0, 0][outside] = -10000.0
    else:
        mask = torch.zeros((1, 1, 1, seq), dtype=torch.float32)
        mask[..., valid:] = -10000.0
    mask.numpy().astype("<f4").tofile(args.output_dir / "mask.f32")
    position_ids = torch.arange(seq).unsqueeze(0)
    with torch.no_grad():
        cos, sin = model.encoder.rotary_emb(hidden, position_ids, layer.attention_type)
        wrapper = LayerExport(layer, cos, sin, args.layout).eval()
        model_output = wrapper(model_input, mask).detach().numpy()
        sequence_output = model_output.squeeze(-1).transpose(0, 2, 1) if args.layout == "nchw" else model_output
        error = np.abs(sequence_output[:, :valid, :] - expected.reshape(1, valid, dim))
        max_abs = float(error.max())
        mean_abs = float(error.mean())
        scale = float(np.max(np.abs(expected)))
    model_output.astype("<f4").tofile(args.output_dir / "expected.f32")
    print("layer", args.layer, "sequence", seq, "valid", valid, "hidden", dim,
          "attention", layer.attention_type, "mask_shape", list(mask.shape),
          "oracle_max_abs", max_abs, "oracle_mean_abs", mean_abs,
          "reference_abs_max", scale, flush=True)
    if max_abs > max(0.001, scale * 1e-6) or mean_abs > 5e-5:
        raise ValueError("layer wrapper does not match the CPU oracle")

    destination = args.output_dir / f"encoder_{args.layer:02d}.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, (model_input, mask), str(destination),
                          opset_version=17, input_names=["hidden", "attention_mask"],
                          output_names=["next_hidden"],
                          do_constant_folding=True, dynamo=False)
    graph = onnx.load(destination, load_external_data=False)
    onnx.checker.check_model(graph)
    print("onnx_bytes", destination.stat().st_size, "ops", dict(Counter(n.op_type for n in graph.graph.node)), flush=True)
    print("exported", destination, flush=True)


if __name__ == "__main__":
    main()
