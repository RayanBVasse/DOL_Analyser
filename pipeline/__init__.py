"""
Chat Analyser — pipeline package.

Stages:
    0. precheck   — format detection, field validation, minimum requirements
    1. parse      — convert conversations.json to canonical internal schema
    2. profile    — cognitive style markers (Steps 3–4)
    3. topics     — TF-IDF topic modelling, cluster labelling (Steps 5–6)
    4. alignment  — dyadic JS divergence (Step 7)
    5. domains    — macro-domain mapping (Steps 8.5–8.6)
    6. robustness — null model permutation tests (Step 8r)
    7. coupling   — weekly/monthly lead-lag coupling (Steps 9–9.1)
    8. dynamics   — scale separation, state segmentation, rolling entropy,
                    episode initiation (Steps 10a/b)
"""
