"""Replay native PF with stacked training tapes; no model or optimizer.

Set WEFTMOE_TENSOR_EXTENSION to the old or fixed artifact, and put WeftMoE
on PYTHONPATH. Unlike repro.cu this diagnostic requires PyTorch/WeftMoE.
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from weftmoe.runtime import TensorRuntime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rounds", type=int, default=200)
parser.add_argument("--layers", type=int, default=8)
parser.add_argument("--seed", type=int, default=8123)
args = parser.parse_args()

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(args.seed + rank)
experts, hidden, intermediate, topk, capacity = 256, 2048, 512, 8, 32768
weights = [
    (
        torch.randn(experts // world, 2 * intermediate, hidden,
                    device="cuda", dtype=torch.bfloat16) * 0.02,
        torch.randn(experts // world, hidden, intermediate,
                    device="cuda", dtype=torch.bfloat16) * 0.02,
    ) for _ in range(args.layers)
]
runtime = TensorRuntime(
    group=dist.group.WORLD, experts=experts, top_k=topk, hidden=hidden,
    intermediate=intermediate, max_tokens=capacity,
    max_rows=capacity * topk * 2 + experts // world * 128,
    training=True, max_saved=args.layers,
)
if rank == 0:
    print(json.dumps(dict(event="start", extension=os.environ["WEFTMOE_TENSOR_EXTENSION"],
                          seed=args.seed, rounds=args.rounds, layers=args.layers,
                          world=world)), flush=True)
start = time.monotonic()
for step in range(args.rounds):
    tapes = []
    for layer, (w1, w2) in enumerate(weights):
        tokens = (8191, 20003, 32767, 17)[(step + layer + rank) % 4]
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x) * 0.0001
        scores = torch.randn(tokens, experts, device="cuda")
        # Rank-independent shifting popularity, with a tail of nearly idle experts.
        bias = torch.linspace(-2, 2, experts, device="cuda").roll((step * 7 + layer) % experts)
        routes = (scores + bias).topk(topk, dim=-1).indices.int()
        gates = torch.rand(tokens, topk, device="cuda")
        gates /= gates.sum(-1, keepdim=True)
        tape = runtime.forward(x, routes, gates, w1, w2)
        tapes.append((tape, dy, tokens))

    for layer in reversed(range(args.layers)):
        tape, dy, tokens = tapes.pop()
        values = (tape.output, *runtime.backward(tape, dy))
        maxima = torch.stack([v.float().abs().max() for v in values])
        # The small finite inputs cannot legitimately produce 1e12 gradients;
        # include large finite corruption, not just NaN/Inf.
        bad = (~torch.isfinite(maxima)) | (maxima > 1e12)
        failed = bad.any().int()
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            for name, value, broken in zip(("output", "dx", "dg", "dw1", "dw2"), values, bad.tolist()):
                if not broken:
                    continue
                mask = (~torch.isfinite(value)) | (value.abs() > 1e12)
                indices = mask.nonzero()[:128]
                print(json.dumps(dict(event="corruption", rank=rank, step=step,
                    layer=layer, tokens=tokens, tensor=name, count=mask.sum().item(),
                    indices=indices.tolist(), values=value[tuple(indices.T)].float().tolist())), flush=True)
            raise RuntimeError("native PF corruption captured before optimizer")
        del tape, dy, values
    if step % 10 == 0 and rank == 0:
        print(json.dumps(dict(event="pass", round=step, backward_calls=(step + 1) * args.layers,
                              elapsed_s=round(time.monotonic() - start, 2))), flush=True)
runtime.close()
dist.destroy_process_group()
if rank == 0:
    print(json.dumps(dict(event="complete", backward_calls=args.rounds * args.layers,
                          elapsed_s=round(time.monotonic() - start, 2))), flush=True)
