"""
02_holographic_gate.py — SwiGLU holographic gate phenomenology.

Demonstrates key results from T4-MLP on a constructed transformer:

  1. φ-boundary identity: σ(±log φ) = 1/φ, 1/φ² (exact)
  2. Dark-fringe energy at deep gate bias
  3. Integer-arithmetic property of the residual stream

Run: python3 output/code/02_holographic_gate.py
"""

import math

import numpy as np

PHI = (1 + 5 ** 0.5) / 2


def phi_sigmoid(x):
    """Exact φ-form of sigmoid."""
    return 1.0 / (1.0 + PHI ** (-x / math.log(PHI)))


def phi_swiglu(x):
    """SiLU gate in φ-form."""
    return phi_sigmoid(x) * x


# ── 1. φ-boundary identity ────────────────────────────────────

print("=" * 58)
print("  Holographic Gate on Constructed Transformer")
print("=" * 58)

print("\n── 1. φ-boundary identity ──")
print("  σ(+log φ) = 1/φ,  σ(-log φ) = 1/φ²  (exact)\n")

all_ok = True
for x, label in [(math.log(PHI), "+log φ"), (-math.log(PHI), "-log φ")]:
    actual = phi_sigmoid(x)
    expected = 1 / PHI if x > 0 else 1 / PHI ** 2
    ok = abs(actual - expected) < 1e-15
    all_ok = all_ok and ok
    print(f"  σ({label:>7s}) = {actual:.16f}  (expected {expected:.16f})  "
          f"[{'PASS' if ok else 'FAIL'}]")

# ── 2. Dark-fringe energy ────────────────────────────────────

print("\n── 2. Gate saturation at extreme bias ──")
print("  Deep negative bias → gate near 0 ('dead' channel)")
print("  Deep positive bias → gate passes through\n")

for bias in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
    g = phi_swiglu(bias)
    print(f"  bias = {bias:+5.1f}  →  gate_out = {g:+8.5f}  "
          f"({'dead/contract' if bias < 0 else 'live/expand' if bias > 0 else 'neutral'})")


# ── 3. Integer arithmetic ─────────────────────────────────────

print("\n── 3. Integer-arithmetic property ──")
print("  With hard attention, all logits are exact integers.\n")

# Residual stream: 3 axes × 6 dims = 18
# NUMBER, LEX_CLASS, TENSE
# each block: [sign, mag, any_state, flag1, flag2, flag3]

def build_resid(number, lex_class, tense, agree_flag=False):
    r = np.zeros(18)
    # NUMBER block (dims 0-5)
    r[0:6] = [number[0], number[1], 1.0, 0, 0, 0]
    # LEX_CLASS block (dims 6-11)
    r[6:12] = [lex_class[0], lex_class[1], 1.0,
               2.0 if agree_flag else 0.0, 0, 0]
    # TENSE block (dims 12-17)
    r[12:18] = [tense[0], tense[1], 1.0, 0, 0, 0]
    return r

# Verb candidates
VERBS = {
    "is":   build_resid((+1, +1), (-1, +1), (+1, +1)),
    "are":  build_resid((-1, +1), (-1, +1), (+1, +1)),
    "was":  build_resid((+1, +1), (-1, +1), (-1, +1)),
    "were": build_resid((-1, +1), (-1, +1), (-1, +1)),
}

# Test: [boys, AGREE_BE_PRESENT] → should decode to "are"
# After attention copies NUMBER from subject to op position:
boys_resid = build_resid((-1, +1), (-1, +1), (+1, +1), agree_flag=True)

print("  Residual at op position for [boys, AGREE_BE_PRESENT]:")
for name, vec in sorted(VERBS.items()):
    logit = int(boys_resid @ vec)
    print(f"    {name:>4s} · residual = {logit:+d}")

margin = int(boys_resid @ VERBS["are"] - max(boys_resid @ VERBS[v]
                                              for v in VERBS if v != "are"))
print(f"\n  Margin for 'are': +{margin}")
print(f"  All logits are exact integers by construction.")

print(f"\nDone.  φ-boundary identity verified: {'ALL PASS' if all_ok else 'ERRORS'}.")
