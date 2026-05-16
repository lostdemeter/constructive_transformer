"""
01_4state_alphabet.py — The 4-state per-axis alphabet.

Demonstrates the core TruthSpace substrate:
  Each semantic axis is a 2-bit register with 4 distinguishable states.
  States are encoded as (sign, magnitude) ∈ {+1, −1}² and live in a
  6-dimensional block: [sign, mag, any_state, FLIP_SIGN_flag, COLLAPSE_flag, EXPAND_flag].

Run: python3 output/code/01_4state_alphabet.py
"""

import numpy as np

# ── The 4 states ──────────────────────────────────────────────
STATES = {
    "+1": np.array([+1, +1]),   # bright pole
    "+0": np.array([+1, -1]),   # bright fringe  (positive zero)
    "-0": np.array([-1, -1]),   # dark fringe     (negative zero)
    "-1": np.array([-1, +1]),   # dark pole
}

print("=" * 60)
print("The 4-State Alphabet: (sign, magnitude)")
print("=" * 60)
for name, vec in STATES.items():
    print(f"  {name:>4s}  →  ({vec[0]:+d}, {vec[1]:+d})")

# ── Dot products: the key geometric fact ──────────────────────
print("\n── Dot-product matrix (sign·sign + mag·mag) ──")
names = list(STATES.keys())
print(f"       ", end="")
for n in names:
    print(f"{n:>6s}", end="")
print()
for n1 in names:
    print(f"  {n1:>4s} ", end="")
    for n2 in names:
        d = int(STATES[n1] @ STATES[n2])
        print(f"{d:>+6d}", end="")
    print()

# ── Key insight: +0 and -0 are distinguishable ───────────────
print("\n── Key insight: +0 and -0 are DISTINGUISHABLE ──")
dot_00 = STATES["+0"] @ STATES["+0"]
dot_0neg0 = STATES["+0"] @ STATES["-0"]
print(f"  +0 · +0 = {int(dot_00):+d}")
print(f"  +0 · -0 = {int(dot_0neg0):+d}")
print(f"  Margin = {int(dot_00 - dot_0neg0)}  (IEEE-754 would give 0)")

# ── Per-axis operators ────────────────────────────────────────
print("\n── Per-axis operators ──")

def flip_sign(s):
    return np.array([-s[0], s[1]])

def collapse(s):
    return np.array([s[0], -1])

def expand(s):
    return np.array([s[0], +1])

OPS = {
    "FLIP_SIGN": flip_sign,
    "COLLAPSE": collapse,
    "EXPAND": expand,
}

for op_name, op_fn in OPS.items():
    print(f"\n  {op_name}:")
    for s_name, s_vec in STATES.items():
        result = op_fn(s_vec)
        r_name = [n for n, v in STATES.items() if np.array_equal(v, result)][0]
        print(f"    {s_name} → {r_name}")

# ── Integer margins from the 6-dim block representation ──────
print("\n── 6-Dim Block Representation (per axis) ──")
BLOCK_DIMS = ["sign", "mag", "any_state",
              "FLIP_SIGN_flag", "COLLAPSE_flag", "EXPAND_flag"]

def embed(state_name, op_name=None):
    """Build a 6-dim block vector for a state, optionally with an op flag."""
    s = STATES[state_name]
    block = np.zeros(6)
    block[0] = s[0]   # sign
    block[1] = s[1]   # magnitude
    block[2] = 1.0    # any_state (1 if loaded)
    if op_name == "FLIP_SIGN":
        block[3] = 1.0
    elif op_name == "COLLAPSE":
        block[4] = 1.0
    elif op_name == "EXPAND":
        block[5] = 1.0
    return block

# Show that cross-state dot products within the full block
# preserve the integer-arithmetic property
print("\n── Block dot products (with any_state=+1) ──")
for s1 in names:
    for s2 in names:
        b1 = embed(s1)
        b2 = embed(s2)
        d = int(b1 @ b2)
        print(f"  {s1:>4s} · {s2:>4s} = {d:+d}  (block={b1[:3].astype(int)} · {b2[:3].astype(int)})")

print("\n── Verification: All dot products are exact integers ──")
all_int = True
for s1 in names:
    b1 = embed(s1)
    for s2 in names:
        dp = embed(s1) @ embed(s2)
        if dp != int(dp):
            all_int = False
            print(f"  FAIL: {s1} · {s2} = {dp} (not integer)")
if all_int:
    print("  PASS: Every cross-product is an exact integer")
print("\n  Self vs. cross margins:")
for s1 in names:
    self_dot = int(embed(s1) @ embed(s1))
    for s2 in names:
        if s1 >= s2:
            continue
        cross_dot = int(embed(s1) @ embed(s2))
        margin = self_dot - cross_dot
        print(f"    margin({s1}, {s2}) = {self_dot} − {cross_dot} = {margin:+d}")

print("\nDone. The 4-state alphabet supports exact integer arithmetic.")
