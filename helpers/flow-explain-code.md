---
description: Flow: Explain one function or file well enough to defend it
argument-hint: "<function or file>"
---
<!-- flow helper: installed by install.py -->

Explain $ARGUMENTS to me so I can defend it in review.

1. What it does, in three sentences.
2. Its inputs and outputs, including shapes, units, or invariants.
3. What callers must guarantee, and what it guarantees in return.
4. A concrete worked example with small values.
5. The inputs most likely to break it.

Cite line numbers. Tell me where you are inferring intent rather than
reading it from the code.
