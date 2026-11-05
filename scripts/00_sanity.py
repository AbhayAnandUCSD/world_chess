"""Hand-check encoding, target masks, labels; MPS vs CPU logits agree."""
import torch
from wc.data import load_games
from wc.labels import render, white_dots
from wc.model import encode, load_model
from wc.score import target_mask

for col in ("w", "b"):
    g = next(x for x in load_games("policy", "karpov") if x.color == col and len(x.sans) > 6)
    T = 70
    m = target_mask(g, "karpov", T)
    tgt = g.text[1:T + 1]
    print(f"Karpov as {col}: {g.white} vs {g.black}")
    print("text  :", repr(tgt))
    print("target:", repr("".join(c if m[i] else "_" for i, c in enumerate(tgt))))
    assert encode(g.text) and all(g.text[a:b] == s for (a, b), s in zip(g.spans, g.sans))
    assert not target_mask(g, "karpov", len(g.text) - 1)[g.spans[-1][0] - 1:].any(), "final move must be unmasked"

g = load_games("probe_fit")[0]
dot, mv, ply, lab = list(white_dots(g.text, g.sans, 16))[0]
print("dot context:", repr(g.text[dot - 20:dot + 1]), "move", mv); print(render(lab))

x = torch.tensor([encode(g.text[:500])])
a = load_model(device="cpu")(x).log_softmax(-1); b = load_model(device="mps")(x.to("mps")).log_softmax(-1).cpu()
print("max |logp cpu - mps| =", (a - b).abs().max().item())
