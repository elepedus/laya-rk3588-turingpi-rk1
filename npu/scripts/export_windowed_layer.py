"""Export one ModernBERT sliding layer with attention tiled inside one graph."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from laya.agent import Agent
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


class WindowedAttention(torch.nn.Module):
    def __init__(self, attention, query_block, half_window):
        super().__init__()
        self.attention = attention
        self.query_block = query_block
        self.half_window = half_window

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **_kwargs):
        batch, length, _ = hidden_states.shape
        head_dim = self.attention.head_dim
        qkv = self.attention.Wqkv(hidden_states)
        qkv = qkv.view(batch, length, 3, -1, head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
        scaling = head_dim ** -0.5
        pieces = []
        for start in range(0, length, self.query_block):
            end = min(start + self.query_block, length)
            key_start = max(0, start - self.half_window)
            key_end = min(length, end + self.half_window)
            scores = torch.matmul(
                query[:, :, start:end],
                key[:, :, key_start:key_end].transpose(2, 3),
            ) * scaling
            scores = scores + attention_mask[:, :, start:end, key_start:key_end]
            weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
            weights = weights.to(query.dtype)
            pieces.append(torch.matmul(weights, value[:, :, key_start:key_end]))
        output = torch.cat(pieces, dim=2).transpose(1, 2)
        output = output.reshape(batch, length, -1).contiguous()
        return self.attention.out_drop(self.attention.Wo(output)), None


class Layer(torch.nn.Module):
    def __init__(self, layer, cos, sin):
        super().__init__()
        self.layer = layer
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, hidden, mask):
        return self.layer(hidden, attention_mask=mask,
                          position_embeddings=(self.cos, self.sin))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("oracle_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--query-block", type=int, required=True)
    args = parser.parse_args()
    if args.query_block < 64:
        parser.error("query block must be at least 64")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    agent = Agent(str(args.model_dir), device="cpu")
    model = agent.model.eval()
    model.encoder.config._attn_implementation = "eager"
    layer = model.encoder.layers[args.layer]
    layer.config._attn_implementation = "eager"
    if layer.attention_type != "sliding_attention":
        raise ValueError("windowed export requires a sliding-attention layer")
    dim = model.encoder.config.hidden_size
    previous = "embedding" if args.layer == 0 else f"encoder_{args.layer - 1:02d}"
    source = np.fromfile(args.oracle_dir / f"{previous}.f32", dtype="<f4")
    reference = np.fromfile(args.oracle_dir / f"encoder_{args.layer:02d}.f32", dtype="<f4")
    valid = source.size // dim
    assert source.size == reference.size == valid * dim and valid <= args.length
    hidden = torch.zeros((1, args.length, dim), dtype=torch.float32)
    hidden[:, :valid] = torch.from_numpy(source.reshape(1, valid, dim).copy())
    mask = torch.zeros((1, 1, args.length, args.length), dtype=torch.float32)
    mask[..., valid:] = -10000.0
    position = torch.arange(args.length)
    half_window = model.encoder.config.local_attention // 2
    mask[0, 0][(position[:, None] - position[None, :]).abs() > half_window] = -10000.0
    with torch.no_grad():
        cosine, sine = model.encoder.rotary_emb(
            hidden, position.unsqueeze(0), layer.attention_type)
        layer.attn = WindowedAttention(layer.attn, args.query_block, half_window)
        wrapper = Layer(layer, cosine, sine).eval()
        actual = wrapper(hidden, mask).numpy()
    error = np.abs(actual[:, :valid] - reference.reshape(1, valid, dim))
    print("layer", args.layer, "length", args.length,
          "query_block", args.query_block, "valid", valid,
          "oracle_max_abs", float(error.max()),
          "oracle_mean_abs", float(error.mean()), flush=True)
    if error.max() > 0.001 or error.mean() > 5e-5:
        raise ValueError("windowed CPU graph differs from oracle")
    hidden.numpy().astype("<f4").tofile(args.output_dir / "hidden.f32")
    mask.numpy().astype("<f4").tofile(args.output_dir / "mask.f32")
    actual.astype("<f4").tofile(args.output_dir / "expected.f32")
    destination = args.output_dir / f"encoder_{args.layer:02d}_window{args.query_block}.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, (hidden, mask), str(destination),
                          opset_version=17, input_names=["hidden", "attention_mask"],
                          output_names=["next_hidden"], do_constant_folding=True,
                          dynamo=False)
    graph = onnx.load(destination, load_external_data=False)
    onnx.checker.check_model(graph)
    print("onnx_bytes", destination.stat().st_size,
          "ops", dict(Counter(node.op_type for node in graph.graph.node)), flush=True)


if __name__ == "__main__":
    main()
