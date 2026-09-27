"""Export a deterministic full Laya trace at an exact valid token length."""

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
    parser.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    agent = Agent(str(args.model_dir), device="cpu")
    model = agent.model.eval()
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
    if not len(ids) <= args.tokens <= agent.cfg["max_len"]:
        raise ValueError(f"requested {args.tokens} tokens outside {len(ids)}..{agent.cfg['max_len']}")
    pattern = agent.tok(" Please review the duplicate invoice and refund request.",
                        add_special_tokens=False)["input_ids"]
    extra = args.tokens - len(ids)
    ids = ids[:-1] + (pattern * ((extra + len(pattern) - 1) // len(pattern)))[:extra] + ids[-1:]
    assert len(ids) == args.tokens
    (args.output_dir / "input.json").write_text(json.dumps({
        "ids": ids, "markers": markers, "state": state,
        "question": question, "qtype": 0,
    }, indent=2))
    manifest = {}

    def save(name, value):
        if isinstance(value, tuple):
            value = value[0]
        array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
        array.astype("<f4").tofile(args.output_dir / f"{name}.f32")
        manifest[name] = {"shape": list(array.shape)}

    hooks = []
    for name, module in [("embedding", model.encoder.embeddings),
                         ("encoder_final", model.encoder.final_norm)]:
        hooks.append(module.register_forward_hook(
            lambda _module, _input, output, name=name: save(name, output)))
    for index, layer in enumerate(model.encoder.layers):
        hooks.append(layer.register_forward_hook(
            lambda _module, _input, output, index=index: save(f"encoder_{index:02d}", output)))
    for index, layer in enumerate(model.head.layers):
        hooks.append(layer.register_forward_hook(
            lambda _module, _input, output, index=index: save(f"head_{index:02d}", output)))
    marker_pos = markers + [0] * (4 - len(markers))
    marker_mask = [True] * len(markers) + [False] * (4 - len(markers))
    with torch.no_grad():
        logits, act_logits = model(
            torch.tensor([ids], dtype=torch.long),
            torch.ones((1, len(ids)), dtype=torch.long),
            torch.tensor([marker_pos], dtype=torch.long),
            torch.tensor([marker_mask]),
            torch.tensor([0]),
        )
    save("logits", logits)
    save("act_logits", act_logits)
    for hook in hooks:
        hook.remove()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("tokens", len(ids), "markers", markers,
          "logits", logits.tolist(), "act_logits", act_logits.tolist(), flush=True)


if __name__ == "__main__":
    main()
