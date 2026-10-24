"""Synthetic checks of drafter.uniform_match, the match under llama.cpp's device sampler (one shared uniform per
position, inverse CDF over each side's tokens in descending order): equal distributions match with probability 1, a
swapped top pair with the share of the rest, a one-hot guess with Flint's probability of that token.

usage: python check_coupling.py
"""
import torch

import drafter


def main():
    V = 50
    ids = torch.tensor([[7, 3, 11, 20] + [V] * 16])
    pk = torch.tensor([[0.6, 0.25, 0.1, 0.05] + [0.0] * 16])
    logits = torch.full((1, V + 1), -30.0)
    for t, pr in zip([7, 3, 11, 20], [0.6, 0.25, 0.1, 0.05]):
        logits[0, t] = torch.log(torch.tensor(pr))
    swapped = logits.clone()
    swapped[0, 3], swapped[0, 7] = logits[0, 7], logits[0, 3]
    onehot = torch.full((1, V + 1), -30.0)
    onehot[0, 7] = 0.0
    for name, lg, want in (("q equal to p", logits, 1.0), ("top two swapped", swapped, 0.15),
                           ("one-hot on Flint's top token", onehot, 0.6)):
        got = float(drafter.uniform_match(lg, ids, pk))
        print(f"{name}: {got:.4f} (expected {want})")
        assert abs(got - want) < 1e-3


if __name__ == "__main__":
    main()
