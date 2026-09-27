"""Export marker scorer and action head as separate RKNN graphs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnx
import torch
from laya.agent import Agent


class ScorerExport(torch.nn.Module):
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, marker_hidden):
        return self.scorer(marker_hidden).squeeze(-1)


def export(module, source, path, input_name, output_name):
    with torch.no_grad():
        torch.onnx.export(module, (source,), str(path), opset_version=17,
                          input_names=[input_name], output_names=[output_name],
                          do_constant_folding=True, dynamo=False)
    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(graph)
    print(path.name, "bytes", path.stat().st_size,
          "ops", [node.op_type for node in graph.graph.node], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("oracle_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    agent = Agent(str(args.model_dir), device="cpu")
    model = agent.model.eval()
    dim = model.encoder.config.hidden_size
    markers = json.loads((args.oracle_dir / "input.json").read_text())["markers"]
    hidden_data = np.fromfile(args.oracle_dir / f"head_{len(model.head.layers)-1:02d}.f32", dtype="<f4")
    hidden = torch.from_numpy(hidden_data.reshape(1, -1, dim).copy())
    marker_input = torch.zeros((1, 32, dim), dtype=torch.float32)
    marker_input[:, :len(markers)] = hidden[:, markers]
    with torch.no_grad():
        scorer = ScorerExport(model.scorer).eval()
        scores = scorer(marker_input)
    oracle_scores = np.fromfile(args.oracle_dir / "logits.f32", dtype="<f4")[:len(markers)]
    score_error = np.max(np.abs(scores.numpy().reshape(-1)[:len(markers)] - oracle_scores))
    print("scorer_oracle_max_abs", score_error, flush=True)
    if score_error > 0.001:
        raise ValueError("scorer disagrees with oracle")
    marker_input.numpy().astype("<f4").tofile(args.output_dir / "marker_hidden.f32")
    scores.numpy().astype("<f4").tofile(args.output_dir / "expected_scores.f32")
    export(scorer, marker_input, args.output_dir / "scorer.onnx", "marker_hidden", "scores")

    full_logits = torch.tensor(np.fromfile(args.oracle_dir / "logits.f32", dtype="<f4"))
    probabilities = torch.softmax(full_logits, dim=-1)
    top2 = probabilities.topk(2).values
    entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum() / math.log(max(2, len(markers)))
    features = torch.stack((top2[0], top2[0] - top2[1], entropy,
                            torch.tensor(len(markers) / 255.0)))
    action_input = torch.cat((hidden[:, 0], features[None]), dim=-1)
    with torch.no_grad():
        action = model.act_head(action_input)
    oracle_action = np.fromfile(args.oracle_dir / "act_logits.f32", dtype="<f4")
    action_error = np.max(np.abs(action.numpy().reshape(-1) - oracle_action))
    print("action_oracle_max_abs", action_error, flush=True)
    if action_error > 0.01:
        raise ValueError("action head disagrees with oracle")
    action_input.numpy().astype("<f4").tofile(args.output_dir / "action_input.f32")
    action.numpy().astype("<f4").tofile(args.output_dir / "expected_action.f32")
    export(model.act_head, action_input, args.output_dir / "action.onnx", "features", "action")


if __name__ == "__main__":
    main()
