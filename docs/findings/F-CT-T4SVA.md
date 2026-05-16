# F-CT-T4SVA — Subject-verb agreement on a constructed transformer

**Status: 24/24 PASS, every case with exact integer margin +2.  First
generative task on the constructed transformer — given a subject, the
network produces the correct present- or past-tense form of "to be"
with hand-built weights and no training of any kind.**

## What the task does

Given a 2-token sequence `[subject, AGREE_OP]`, produce the correct
inflected form of "to be":

- `[boy,  AGREE_BE_PRESENT]` → `is`
- `[boys, AGREE_BE_PRESENT]` → `are`
- `[boy,  AGREE_BE_PAST]`    → `was`
- `[boys, AGREE_BE_PAST]`    → `were`

12 subjects × 2 tenses = 24 cases.  All pass.

## Architecture

3 axes, hidden = 18, **one attention layer, no MLP**.

| Axis | Encoding | Used by |
|---|---|---|
| **NUMBER** | +1 sing / -1 plur | Subjects + verbs |
| **LEX_CLASS** | +1 noun-class / -1 verb-class | Subjects (+1) + verbs (-1) + op (-1) |
| **TENSE** | +1 present / -1 past | Verbs + op |

**Subjects** carry NUMBER and LEX_CLASS, but no TENSE (the block stays
zero — they are tense-neutral).

**Verbs** carry NUMBER, LEX_CLASS = -1, and TENSE.  Each of `{is, are,
was, were}` lives at a unique (NUMBER, TENSE) corner of the verb
sub-space.

**Op tokens** (`AGREE_BE_PRESENT`, `AGREE_BE_PAST`) carry:
- LEX_CLASS = -1 (pre-commit to verb-class)
- TENSE = +1 or -1 (pre-commit to the desired tense)
- An agree-flag dim in the LEX_CLASS block
- NUMBER block left empty (will be filled by attention)

**One attention head** with:
- Q matches the agree-flag at the current position
- K matches the previous token's NUMBER any-state
- V is identity-minus-flag
- W_O is **selective**: writes only the NUMBER block of the source
  into the current position

## Mechanism

After running `[subject, op]` through one attention layer, the
residual at the op position is:

```
NUMBER   block: subject's NUMBER content      (copied by attention)
LEX_CLASS block: -1, verb-class               (from op embedding)
TENSE    block: op's chosen tense             (from op embedding)
```

This residual matches exactly one verb in the vocab (the one with the
matching NUMBER and TENSE) and decodes against it with margin +2 over
the closest distractor.

The constructed transformer is doing real generation: the *output
token* (`were`) is not present anywhere in the input sequence (`boys`,
`AGREE_BE_PAST`).  It is *constructed* by combining NUMBER routed
from the subject with LEX_CLASS and TENSE committed by the op
embedding.

## Why this is the canonical example

This is the simplest possible transformer task that is genuinely
generative:

- **Not retrieval** — the answer is not present in the input.
- **Not a single-axis flip** — requires combining two different
  content sources (attention from subject + embedding from op).
- **Not a chain** — only one operator; the cleverness is in the
  multi-source combination.

It is also the simplest task where you can clearly see *why* a
transformer is composed of attention + embeddings:

- Attention provides **content routing** across positions
  (subject's NUMBER → op position's NUMBER block).
- Embeddings provide **content commitment** at a position
  (op embedding writes LEX_CLASS and TENSE directly).

Both are needed.  Attention alone could not commit LEX_CLASS = verb;
embeddings alone could not see what number the subject was.

## Margins

Every case has logit margin exactly +2 between the correct verb and
the closest distractor.  The decomposition:

For `[boys, AGREE_BE_PAST] → were`:

| candidate | NUMBER match | LEX_CLASS match | TENSE match | total |
|---|---|---|---|---|
| **were**  | (-1)·(-1) + 1·1 + 1·1 = +3 | (-1)·(-1) + 1·1 + 1·1 = +3 | (-1)·(-1) + 1·1 + 1·1 = +3 | **+9** |
| was       | wrong sign on NUMBER       | match LEX = +3              | match TENSE = +3            | +5 + 1 + 1? wait... |

Actually the table simplifies: each axis contributes
(sign·sign + mag·mag + 1·1) = +3 for matching, +1 for opposing-sign
(sign·sign = -1 but the other 2 dims still match), giving 9 vs 7 vs
7 vs 5 across the 4 verbs in the printed output.  Margin between top
and runner-up = 2.

## What this completes

With T4-SVA, the constructive theory of language models has now been
demonstrated end-to-end on a real linguistic task:

| Component | Status | Validation |
|---|---|---|
| Substrate (4-state alphabet) | T4-NEG0 | 288/288 byte-equal compositions |
| Algebra (per-axis ops) | T4-NEG0, T4-MLP | exact integer margins |
| Cross-axis routing | T4-MLP-CROSS | 0→32 with 6 SwiGLU channels |
| Phenomenology on real models | T4-OLMo2 | 3/4 paper findings on OLMo2-1B |
| Labelling map | T4-OLLAMA + hand | 24/24 self-decode, 12/14 analogy |
| **Generation** | **T4-SVA** | **24/24 SVA with margin +2** |

The constructed transformer is no longer just a static labelled space
— it generates output tokens that are not in the input.  It does so
by combining attention-routed content with embedding-committed
content, the same recipe trained transformers use, but with weights
chosen by hand rather than gradient descent.

## Files

- `experiments/t4_sva.py` — script (~280 lines)
- This finding: `docs/findings/F-CT-T4SVA.md`

## One-line summary

Given `[subject, AGREE_BE_TENSE]`, the constructed transformer
generates the correct inflected form of "to be" 24 out of 24 times,
margin +2, no MLP, no training — just one attention head with a
selective W_O routing the subject's NUMBER, plus an op embedding
committing the LEX_CLASS and TENSE.
