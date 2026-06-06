---
title: Clinical Benefit-Risk Analyzer
emoji: ⚖️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: Benefit-risk decision plane for clinical drug evaluation
---

# ⚖️ Clinical Benefit–Risk Analyzer

A **domain-specific benefit–risk decision visualization** that encodes clinical trial evidence, regulatory classification rules, and HTA decision logic into a single interactive decision surface.

Built with [Dash](https://dash.plotly.com/) + [Plotly](https://plotly.com/) — deployable on **Hugging Face Spaces** (Docker SDK).

---

## Proposed Plotly Express API

This tool implements a domain-specific primitive that does not yet exist in Plotly Express. The equivalent high-level call, once contributed to the Plotly library, would look like:

```python
import plotly.express as px

fig = px.clinical_benefit_risk(
    df,
    x="efficacy",        # signal strength: response rate, HR_inverted, symptom improvement
    y="risk",            # risk axis: AE rate, hospitalization risk, toxicity score
    size="sample_size",  # evidence weight: n patients, inverse CI width
    color="decision_class",  # "preferred" | "conditional" | "avoid" | "uncertain"
    hover_data=["drug", "indication", "efficacy", "risk",
                "sample_size", "br_ratio", "net_benefit"],
    thresholds={
        "efficacy_min": 0.50,   # minimum efficacy to be considered beneficial
        "risk_max":     0.30,   # maximum acceptable risk for preferred status
        "n_min":        50,     # minimum sample size for non-uncertain classification
    },
)
fig.show()
```

### Decision logic (abstracted internally)

```
n < n_min               →  uncertain   ⚪  (insufficient evidence)
eff ≥ t  AND risk ≤ t   →  preferred   🟢  (clear benefit, low harm)
eff ≥ t  OR  risk ≤ t   →  conditional 🟡  (moderate tradeoff)
eff < t  AND risk > t   →  avoid       🔴  (unfavorable profile)
```

### Standalone usage (today)

```python
from app import classify_benefit_risk

df = classify_benefit_risk(
    df,
    efficacy_thresh=0.50,
    risk_thresh=0.30,
    n_min=50,
)
# df now has: decision_class, net_benefit, br_ratio columns
```

---

## What this replaces

| Traditional format | This tool |
|--------------------|-----------|
| HTA benefit–risk table (tabular) | Visual decision surface |
| Oncology waterfall plots | Comparative scatter with evidence weight |
| Forest plots (meta-analysis) | Benefit–risk plane with classification |
| Committee decision grids | Interactive threshold-adjustable quadrants |

---

## Why it's different from a scatter plot

A standard scatter plot shows **correlation**. This tool shows a **clinical recommendation**:

- **Axes are semantically directional**: higher X = more efficacious, lower Y = safer
- **Quadrant logic is embedded**: the threshold lines are regulatory decision boundaries, not aesthetic guides
- **Size encodes evidence**: large points = robust evidence (high n), small points = fragile evidence
- **Color = recommendation**: green/yellow/red/gray map to the EMA benefit-risk framework classes
- **All thresholds are adjustable live** — the decision surface updates in real time

---

## Input data format

Upload a CSV with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `drug` | string | Drug / intervention name |
| `indication` | string | Disease or indication |
| `efficacy` | float [0–1] | Response rate, or normalized efficacy measure |
| `risk` | float [0–1] | Adverse event rate, or normalized risk measure |
| `sample_size` | int | Total number of patients in evidence base |
| `ci_width` | float | Optional — 95% CI width on efficacy (auto-computed if absent) |

Efficacy and risk must be normalized to [0, 1]. For hazard ratios: use `1 − HR` as efficacy. For toxicity scores: normalize to [0, 1] range.

---

## Metrics computed

| Metric | Formula | Meaning |
|--------|---------|---------|
| `br_ratio` | `efficacy / risk` | Benefit-risk ratio |
| `net_benefit` | `(eff − risk) × log(n)` | Composite score weighted by evidence |
| `inv_ci_width` | `1 / ci_width` | Precision (inverse CI width) |
| `decision_class` | Rule-based | preferred / conditional / avoid / uncertain |

---

## Therapeutic areas covered in sample data

- **Oncology** — immunotherapy, targeted therapy, chemotherapy
- **Cardiology** — heart failure, anticoagulation, lipid management
- **Psychiatry / Neurology** — depression, schizophrenia, ADHD, Alzheimer's
- **Infectious disease** — HIV, Hepatitis C, COVID-19, hospital infections

55 drug-indication pairs with realistic efficacy and risk values from published trials.

---

## Local development

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:7860
```

---

## Real-world usage

- HTA committees (EMA, NICE, HAS, IQWiG)
- Oncology tumor boards
- Formulary decisions
- Clinical pharmacology and safety reviews
- Comparative effectiveness research
