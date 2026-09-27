"""Export adjacent ModernBERT layers as one RKNN graph to avoid host handoffs."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from laya.agent import Agent
from export_windowed_layer import WindowedAttention


class EncoderBlock(torch.nn.Module):
    def __init__(self, layers, full_rope, local_rope):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.register_buffer("cos_full", full_rope[0])
        self.register_buffer("sin_full", full_rope[1])
        self.register_buffer("cos_local", local_rope[0])
        self.register_buffer("sin_local", local_rope[1])

    def forward(self, hidden, full_mask, local_mask):
        for layer in self.layers:
            if layer.attention_type == "full_attention":
                hidden = layer(hidden, attention_mask=full_mask,
                               position_embeddings=(self.cos_full, self.sin_full))
            else:
                hidden = layer(hidden, attention_mask=local_mask,
                               position_embeddings=(self.cos_local, self.sin_local))
        return hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("oracle_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--window-query", type=int,
                        help="tile sliding attention inside the fused graph")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    agent = Agent(str(args.model_dir), device="cpu")
    model = agent.model.eval()
    model.encoder.config._attn_implementation = "eager"
    layers = list(model.encoder.layers[args.start:args.start + args.count])
    if len(layers) != args.count:
        raise ValueError("block runs past last encoder layer")
    for layer in layers:
        layer.config._attn_implementation = "eager"
        if args.window_query and layer.attention_type == "sliding_attention":
            if args.window_query < 64:
                raise ValueError("window query must be at least 64")
            layer.attn = WindowedAttention(
                layer.attn, args.window_query,
                model.encoder.config.local_attention // 2)
    dim = model.encoder.config.hidden_size
    previous = "embedding" if args.start == 0 else f"encoder_{args.start-1:02d}"
    last = args.start + args.count - 1
    input_data = np.fromfile(args.oracle_dir / f"{previous}.f32", dtype="<f4")
    expected = np.fromfile(args.oracle_dir / f"encoder_{last:02d}.f32", dtype="<f4")
    valid = input_data.size // dim
    assert input_data.size == expected.size == valid * dim and valid <= args.length
    hidden = torch.zeros((1, args.length, dim), dtype=torch.float32)
    hidden[:, :valid] = torch.from_numpy(input_data.reshape(1, valid, dim).copy())
    full_mask = torch.zeros((1, 1, args.length, args.length), dtype=torch.float32)
    full_mask[..., valid:] = -10000.0
    local_mask = full_mask.clone()
    position = torch.arange(args.length)
    half_window = model.encoder.config.local_attention // 2
    local_mask[0, 0][(position[:, None] - position[None, :]).abs() > half_window] = -10000.0
    position = position.unsqueeze(0)
    with torch.no_grad():
        full_rope = model.encoder.rotary_emb(hidden, position, "full_attention")
        local_rope = model.encoder.rotary_emb(hidden, position, "sliding_attention")
        wrapper = EncoderBlock(layers, full_rope, local_rope).eval()
        output = wrapper(hidden, full_mask, local_mask).numpy()
    delta = np.abs(output[:, :valid] - expected.reshape(1, valid, dim))
    scale = float(np.max(np.abs(expected)))
    print("start", args.start, "count", args.count, "length", args.length,
          "window_query", args.window_query,
          "valid", valid, "oracle_max_abs", float(delta.max()),
          "oracle_mean_abs", float(delta.mean()), "reference_abs_max", scale,
          flush=True)
    if delta.max() > max(0.001, scale * 2e-5) or delta.mean() > 5e-5:
        raise ValueError("block differs from CPU oracle")
    for name, array in (("hidden", hidden), ("mask_full", full_mask),
                        ("mask_local", local_mask),
                        ("expected", torch.from_numpy(output))):
        array.numpy().astype("<f4").tofile(args.output_dir / f"{name}.f32")
    destination = args.output_dir / f"encoder_{args.start:02}_{last:02}.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, (hidden, full_mask, local_mask), str(destination),
                          opset_version=17,
                          input_names=["hidden", "full_mask", "local_mask"],
                          output_names=["next_hidden"],
                          do_constant_folding=True, dynamo=False)
    graph = onnx.load(destination, load_external_data=False)
    onnx.checker.check_model(graph)
    print("onnx_bytes", destination.stat().st_size,
          "ops", dict(Counter(node.op_type for node in graph.graph.node)),
          flush=True)


if __name__ == "__main__":
    main()
