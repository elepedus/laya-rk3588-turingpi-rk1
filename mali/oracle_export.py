"""Capture exact CPU reference tensors for one English Laya question.

Run with the installed Laya package and a local checkpoint. All output files go
under the requested directory; no personal data or network calls are involved.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from laya.agent import Agent
from laya.common import build_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    agent = Agent(str(args.model_dir), device="cpu")
    model = agent.model
    question = {
        "t": "choice",
        "ins": "Which department should handle this?",
        "crit": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "other": "everything else",
        },
    }
    state = "I was billed twice for March. Please refund the duplicate charge."
    ids, markers = build_sequence(agent.tok, state, question,
                                  max_len=agent.cfg["max_len"],
                                  head_max_len=agent.cfg["head_max_len"])
    sequence = {
        "ids": ids,
        "markers": markers,
        "state": state,
        "question": question,
        "qtype": 0,
    }
    (args.output_dir / "input.json").write_text(json.dumps(sequence, indent=2))
    manifest = {}

    def save(name, value):
        if isinstance(value, tuple):
            value = value[0]
        array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
        array.astype("<f4").tofile(args.output_dir / (name + ".f32"))
        manifest[name] = {"shape": list(array.shape), "first": array.reshape(-1)[:8].tolist()}

    hooks = []
    for name, module in [("embedding", model.encoder.embeddings),
                         ("encoder_final", model.encoder.final_norm),
                         ("encoder_00_attention", model.encoder.layers[0].attn),
                         ("encoder_00_mlp", model.encoder.layers[0].mlp)]:
        hooks.append(module.register_forward_hook(lambda _module, _input, output, name=name: save(name, output)))
    for i, layer in enumerate(model.encoder.layers):
        hooks.append(layer.register_forward_hook(
            lambda _module, _input, output, i=i: save(f"encoder_{i:02d}", output)))
    for i, layer in enumerate(model.head.layers):
        hooks.append(layer.register_forward_hook(
            lambda _module, _input, output, i=i: save(f"head_{i:02d}", output)))
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention = torch.ones_like(input_ids)
    marker_pos = torch.tensor([markers + [0] * (4 - len(markers))], dtype=torch.long)
    marker_mask = torch.tensor([[True] * len(markers) + [False] * (4 - len(markers))])
    with torch.no_grad():
        logits, act_logits = model(input_ids, attention, marker_pos, marker_mask,
                                   torch.tensor([0]))
    save("logits", logits)
    save("act_logits", act_logits)
    for hook in hooks:
        hook.remove()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("sequence length", len(ids), "markers", markers)
    print("logits", logits.tolist(), "act_logits", act_logits.tolist())


if __name__ == "__main__":
    main()
