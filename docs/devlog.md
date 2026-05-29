# Development Log

This log records design decisions, dead ends, and dates throughout the project.
The marking brief explicitly rewards documented iteration — reconstructing this
log at the end is not possible, so keep it current.

**How to use this file:**
- Add a new dated entry whenever you make a non-trivial decision, hit a dead end,
  or change direction.
- Be honest about what didn't work. Negative results are valuable evidence of
  rigorous process.
- Reference commits, files, or test results where relevant.

---

## 2026-MM-DD — Phase 0 scaffold initialised

**Decision:** Project structure follows the six development phases (data,
features, models, backtest, explain, web). Each phase is a separable subpackage
to allow swapping components without breaking the pipeline.

**Decision:** Python 3.13 selected. Noted TensorFlow compatibility risk; mitigation
is to fall back to a separate Python 3.12 venv for Phase 2 LSTM work if needed.

**Decision:** RF baseline chosen as both the comparison benchmark AND the primary
SHAP target. Rationale: SHAP works cleanly on tree models; SHAP on LSTM is
genuinely awkward and the project should not depend on it working perfectly.

**Decision:** Walk-forward validation will be the standard evaluation method
from Phase 2 onwards (not a single train/test split). Rationale: financial
time-series data is non-stationary, so a single split risks lucky-period bias.

---

## Template for future entries

```
## YYYY-MM-DD — [Short title of decision or event]

**Context:** What was happening, what triggered the decision.

**Options considered:** What alternatives were on the table.

**Decision:** What you chose and why.

**Outcome:** (Fill in later) What actually happened. Did it work?
```
