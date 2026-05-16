# F-CT-T4MLPCROSS — CROSS_COPY: the simplest MLP-required primitive

**Status: PASS, with exact integer margins, and a clean
attention/MLP separation:**

- **Without MLP**: 0/32 cases pass (every margin exactly 0.000)
- **With MLP**: 32/32 cases pass (every margin exactly +2.000)
- **T4-NEG0 invariants survive**: 24/24 (with and without MLP)

This locates the attention/MLP expressivity boundary precisely: per-axis
operators (`FLIP`, `COLLAPSE`, `EXPAND`) live in attention; cross-axis
operators (`COPY`, agreement, content-conditional writes) require MLP.

## What CROSS_COPY does

`CROSS_COPY_a→b` applied to source `(s_a, s_b)` produces `(s_a, s_a)`.
Axis `a` is preserved; axis `b` is overwritten with axis `a`'s state.
Concretely on our 16-concept vocab, with two axes:

- `(+1, +0)` → `(+1, +1)` for direction a0→a1
- `(-0, +1)` → `(-0, -0)` for direction a0→a1
- `(+1, -1)` → `(-1, -1)` for direction a1→a0

This is *content-conditional routing across blocks* — the value
written to block `b` depends on what's in block `a` of the source.

## Why pure attention cannot do this

T4-NEG0's per-axis operator algebra is *abelian on disjoint blocks*:
every attention head's `W_O` is a fixed linear map that touches only
its own axis's block.  Composing attention heads gives the direct
product `G_a × G_b` of per-axis groups — there is no way to express
"the value written to block `b` should equal the source's block-`a`
content" without exponentially many heads (one per concrete state).

Empirically: with attention alone (the 6 T4-NEG0 heads + 2 new
COPY_BLOCK heads that copy block-`a` to the op position), every
CROSS_COPY case has logit margin **exactly 0.000** — the residual
contains the right block-`a` content but the LM-head dot product
cannot tell the difference between the 4 candidates that share that
block-`a` content.  Attention has done all it can; the routing into
block `b` is *missing*.

## How the MLP solves it

A SwiGLU MLP with 6 channels (3 per direction) is sufficient.  For
each (src_axis, dst_axis) direction and each routable per-block dim
`d ∈ {sign, magnitude, any_state}`:

- `gate_k = M_MLP · cross_copy_flag_src` (≈10 when the flag is +2,
  ≈0 otherwise)
- `up_k = source_block_src.dim_d` (sign, magnitude, or any_state value)
- `down_k → dst_block.dim_d` with scale `1 / SiLU(M_MLP · 2)` so the
  delta is `+1 · source_value`

When the CROSS_COPY flag is set, the gate fires, SiLU saturates near
linear, the up-projection carries the actual content, and the
down-projection routes it into the destination block.  When the flag
is not set, the gate is zero, SiLU is zero, the channel is silent.

This is the canonical "key/value gating" pattern that real
transformer MLPs use — the gate is a content-independent switch (the
op flag), the up is the content (source block), the down is the
routing (destination block).

## Why the margins are exactly ±2 / 0

- **Margin 0 without MLP**: the residual at the op position has block
  `a` set to source's block-`a` content (via the COPY_BLOCK head), but
  block `b` is zero.  Concept tokens that share the source's block-`a`
  state with various block-`b` states all score identically because
  the residual contributes 0 in block-`b`'s sign/mag dims (which are
  the only dims that distinguish them).  Tie → margin = 0.

- **Margin +2 with MLP**: the MLP writes block-`a`'s content into
  block-`b` exactly, producing a residual that matches the target
  concept in *both* blocks.  Margin equals the per-block contribution
  of the matching state minus the per-block contribution of the
  closest distractor — the same +2 we saw in T4-NEG0's single-axis
  tests, but now arising from MLP routing rather than attention W_O.

## Architecture extension

Added to T4-NEG0:

- 2 new dims at the end of the residual:
  - dim 12 = `CROSS_COPY_FLAG_a0_to_a1`
  - dim 13 = `CROSS_COPY_FLAG_a1_to_a0`
- 2 new attention heads (`COPY_BLOCK`, one per source axis) that fire
  on the corresponding flag and write the source's block content into
  the op position with `W_O = identity_on_block`.
- 1 SwiGLU MLP with 6 inter-dim channels.

Hidden size grew from 12 to 14.  The 6 original T4-NEG0 attention
heads and the per-axis operator algebra are completely unchanged.

## What this proves

Three sharp claims:

1. **There exists a primitive that pure abelian attention cannot
   express.**  The 0/32 result is not "barely fails" — every margin is
   exactly 0.000.  The structure literally cannot be encoded without
   nonlinear routing.

2. **A minimal SwiGLU MLP suffices to break the abelian barrier.**
   6 channels and one MLP layer turn 0/32 into 32/32 with margin +2
   on every case.  The MLP buys *expressivity*, not just stability.

3. **The attention/MLP boundary maps to a known algebraic distinction.**
   Attention realises the abelian product algebra `G_a × G_b × ...`.
   MLP realises content-conditional cross-block routing.  Together
   they realise non-abelian operator algebras over the residual
   stream.  The constructed transformer now hosts both regimes
   coexisting.

## Why this matters for the larger story

T4-NEG0 + T4-MLP + T4-OLMO2 + T4-MLP-CROSS together establish a
coherent picture:

- **The 4-state alphabet** is the universal substrate (T4-NEG0:
  exact integer arithmetic; T4-OLMO2: replicates on real OLMo2-1B).
- **Attention realises the per-axis abelian algebra** (T4-NEG0: 288
  chain-composition cases byte-equal across orderings).
- **SwiGLU MLP adds cross-block routing** (T4-MLP-CROSS: 0→32 jump
  with 6 inter-dim channels).
- **Together, attention + MLP host the same holographic phenomenology
  as a real production transformer** (T4-OLMO2: 3 of 4 paper findings
  replicate on OLMo2-1B).

The attention/MLP boundary in our construction is the same boundary
real LLMs use: per-axis content-independent transformations are cheap
(attention); cross-axis content-conditional writes are nonlinear and
require MLP.

## Files

- `experiments/t4_mlp_cross_copy.py` — full script (~360 lines)
- This finding: `docs/findings/F-CT-T4MLPCROSS.md`
- Companions: F-CT-T4NEG0.md, F-CT-T4MLP.md, F-CT-T4OLMO2.md

## One-line summary

CROSS_COPY proves the attention/MLP separation: 0/32 with attention
alone (margins exactly 0), 32/32 with a 6-channel SwiGLU MLP added
(margins exactly +2).  The simplest primitive that strictly requires
MLP routing, realised in 6 channels of nonlinearity.
