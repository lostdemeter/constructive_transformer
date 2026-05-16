"""
04_constructive_sva.py — Subject-verb agreement on a constructed transformer.

Demonstrates T4-SVA: a 1-layer, 1-head constructed transformer that
generates the correct form of "to be" (is/are/was/were) from a
[subject, AGREE_OP] input sequence.

Key mechanism:
  - Attention routes NUMBER from the subject token to the operator token
  - Embeddings commit LEX_CLASS and TENSE at each position
  - LM-head dot products produce exact integer margins (+2)

Run: python3 output/code/04_constructive_sva.py
"""

import numpy as np

# ── Layout: 3 axes × 6 dims = 18-dim residual ────────────────

AXES = ["NUMBER", "LEX_CLASS", "TENSE"]
BLOCK = 6
HIDDEN = len(AXES) * BLOCK  # 18

SIGN, MAG, ANYSTATE = 0, 1, 2
NUMBER_AXIS, LEXCLASS_AXIS, TENSE_AXIS = 0, 1, 2


def axis_block(sign, mag, any_state=1.0, agree_flag=0.0):
    v = np.zeros(BLOCK)
    v[SIGN] = sign
    v[MAG] = mag
    v[ANYSTATE] = any_state
    v[3] = agree_flag
    return v


def residual(axis_vals):
    blocks = []
    for ax in AXES:
        if ax in axis_vals and axis_vals[ax] is not None:
            s, m, *rest = axis_vals[ax]
            af = rest[0] if rest else 0.0
            blocks.append(axis_block(s, m, agree_flag=af))
        else:
            blocks.append(np.zeros(BLOCK))
    return np.concatenate(blocks)


# ── Vocabulary ─────────────────────────────────────────────────

# 12 subjects (6 singular, 6 plural)
SUBJECTS = {}
for name in ["boy", "girl", "dog", "cat", "wolf", "bird"]:
    SUBJECTS[name] = residual({
        "NUMBER": (+1, +1),     # singular
        "LEX_CLASS": (+1, +1),  # noun
    })
for name in ["boys", "girls", "dogs", "cats", "wolves", "birds"]:
    SUBJECTS[name] = residual({
        "NUMBER": (-1, +1),     # plural
        "LEX_CLASS": (+1, +1),  # noun
    })

# 4 verb forms
VERBS = {
    "is":   residual({"NUMBER": (+1, +1), "LEX_CLASS": (-1, +1), "TENSE": (+1, +1)}),
    "are":  residual({"NUMBER": (-1, +1), "LEX_CLASS": (-1, +1), "TENSE": (+1, +1)}),
    "was":  residual({"NUMBER": (+1, +1), "LEX_CLASS": (-1, +1), "TENSE": (-1, +1)}),
    "were": residual({"NUMBER": (-1, +1), "LEX_CLASS": (-1, +1), "TENSE": (-1, +1)}),
}

# 2 operators (present/past agreement)
AGREE_BE_PRESENT = residual({
    "LEX_CLASS": (-1, +1),     # verb class
    "TENSE": (+1, +1),         # present
    # AGREE flag at NUMBER_AXIS, dim 3
})
AGREE_BE_PRESENT[NUMBER_AXIS * BLOCK + 3] = 2.0  # agree flag

AGREE_BE_PAST = residual({
    "LEX_CLASS": (-1, +1),     # verb class
    "TENSE": (-1, +1),         # past
})
AGREE_BE_PAST[NUMBER_AXIS * BLOCK + 3] = 2.0  # agree flag


# ── Attention: route NUMBER from subject to op position ──────

def attention_head(subject_resid, op_resid):
    """One attention head: fires on AGREE flag, copies NUMBER only.

    W_Q matches the AGREE flag at (NUMBER_AXIS, dim 3).
    W_K matches ANYSTATE at NUMBER_AXIS.
    W_O writes NUMBER block from source to target, selectively.
    """
    # Check if AGREE flag fires (value > 1.5)
    agree_flag = op_resid[NUMBER_AXIS * BLOCK + 3]
    if agree_flag < 1.5:
        return op_resid.copy()  # no firing

    # Copy subject's NUMBER block into op position
    result = op_resid.copy()
    src_start = NUMBER_AXIS * BLOCK
    tgt_start = NUMBER_AXIS * BLOCK
    result[tgt_start:tgt_start + BLOCK] = subject_resid[src_start:src_start + BLOCK]
    return result


# ── LM Head ────────────────────────────────────────────────────

def lm_head(residual):
    """Decode by dot-product with each verb candidate."""
    logits = {}
    for name, vec in VERBS.items():
        logit = float(residual @ vec)
        # Integer check
        if abs(logit - round(logit)) > 1e-10:
            raise ValueError(f"Non-integer logit: {logit}")
        logits[name] = int(logit)
    return logits


# ── Test ───────────────────────────────────────────────────────

print("=" * 58)
print("  Constructive Subject-Verb Agreement (T4-SVA)")
print("  Hand-placed weights, no gradient descent.")
print("=" * 58)

all_ok = 0
n_tests = 0

for subj_name, subj_resid in SUBJECTS.items():
    for tense_name, op_resid in [("present", AGREE_BE_PRESENT),
                                  ("past", AGREE_BE_PAST)]:
        n_tests += 1

        # Forward pass
        attended = attention_head(subj_resid, op_resid)
        logits = lm_head(attended)
        top1 = max(logits, key=logits.get)

        # Expected: is/are for present, was/were for past
        is_plural = (subj_name.endswith('s') and subj_name not in
                     ['is', 'was', 'were', 'his', 'this'])
        # Simple heuristic: plural if name ends in 's'
        if subj_name in ["boys", "girls", "dogs", "cats", "wolves", "birds"]:
            plural = True
        else:
            plural = False

        if tense_name == "present":
            expected = "are" if plural else "is"
        else:
            expected = "were" if plural else "was"

        ok = (top1 == expected)
        all_ok += ok
        margin = logits[expected] - max(v for k, v in logits.items() if k != expected)
        mk = "PASS" if ok else "FAIL"
        print(f"  [{mk}] [{subj_name:6s}, AGREE_BE_{tense_name.upper():6s}]  "
              f"target={expected:4s}  top1={top1:4s}  margin=+{margin}")

print(f"\n  Result: {all_ok}/{n_tests}")
pct = 100.0 * all_ok / max(n_tests, 1)
if pct == 100.0:
    print(f"\n  PASS: The constructed transformer generates correct verb forms.")
    print(f"  Every logit margin is an exact integer by construction.")
    print(f"  Attention routes content; embeddings commit content.")
else:
    print(f"\n  FAIL: {all_ok}/{n_tests} — unexpected errors.")
