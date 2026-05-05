# Peering Inside the Overton Window: A Computational Analysis of Discourse Shift in US News Headlines (2001 - 2024)

**Paris Heard** · School of Information, University of Michigan · SI 630: Natural Language Processing · Winter 2026

---

## Overview

This project operationalizes the political science concept of the **Overton Window** — the range of ideas considered acceptable in mainstream discourse — using computational methods applied to over 6 million U.S. news headlines spanning two decades. By treating the 2001–2015 period as a baseline and measuring how the 2016–2024 corpus drifts from it in TF-IDF space, the project quantifies how the boundaries of mainstream political discourse have shifted across five major outlets.

**Research question:** Can the drift of the Overton Window be measured and characterized using NLP methods applied to large-scale news headline corpora?

---

## Data

### Primary Dataset
**US Multi-Outlet News Headlines 2001–2024**  
Source: [`dess-mannheim/US_Multi_Outlet_News_Headlines2001_2024`](https://huggingface.co/datasets/dess-mannheim/US_Multi_Outlet_News_Headlines2001_2024) on HuggingFace  
Access: **Restricted — research use only.** Request access on HuggingFace, then authenticate via `huggingface-hub` before running the main analysis notebook.

| Outlet | Headlines |
|---|---|
| New York Times | 2,156,490 |
| Los Angeles Times | 1,674,433 |
| Fox News | 1,583,499 |
| Wall Street Journal | 724,275 |
| MSNBC | 97,184 |
| **Total Headlines (Raw)** | **6,235,881** |
| **Total Headlines (Filtered)** | **5,804,010** |

### Secondary Dataset (Baseline Exploration)
**Babel Briefings** — multilingual news dataset used in early exploratory analysis (`01_baseline_exploration.ipynb`). 54 JSON files by country/language. See notebook for schema details.

Raw data files are not included in this repository. See notebooks for loading instructions.

---

## Methods

- **Corpus split:** Baseline (2001–2015) vs. Comparison (2016–2024)
- **Vectorization:** TF-IDF with unigrams and bigrams (`max_features=10,000`, `min_df=5`, `ngram_range=(1,2)`)
- **Drift measurement:** Monthly cosine distance from the baseline centroid
- **Statistical validation:** Permutation test (n=1,000) to confirm observed drift exceeds chance
- **Sentiment analysis:** VADER sentiment scoring with intra-month volatility and sentiment-conditioned drift analysis

---

## Key Findings

- The comparison corpus (2016–2024) drifted significantly from the baseline centroid, with a mean cosine distance of **0.3426** vs. a permutation-test random baseline of **0.2341** — meaning observed drift is ~46% greater than chance
- Drift varied by sentiment valence: **+0.0945** for negative headlines, **+0.1238** for positive, and **+0.1723** for neutral
- Mean headline sentiment shifted from **−0.0124** (baseline) to **−0.0642** (comparison), a statistically significant negative shift (t = 20.78, p < 0.001, Cohen's d = −2.52)
- Top TF-IDF terms distinguishing the comparison corpus: *trump, biden, police, election, court*

---

## Repository Structure

```
overton-window-nlp/
│
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
│
├── notebooks/
│   ├── 01_baseline_exploration.ipynb   # Early exploration using Babel Briefings dataset
│   └── 02_main_analysis.ipynb          # Full pipeline: TF-IDF, cosine drift, permutation test, sentiment
│
├── src/
│   └── overton_pipeline.py             # Reusable utilities (vectorization, plotting helpers)
│
├── outputs/
│   └── figures/                        # All report figures (fig1–fig7, 300 DPI)
│
└── data/
    └── README.md                       # Data source documentation and access instructions
```

---

## Reproduction

### Requirements
```bash
pip install -r requirements.txt
```

### Authentication
The primary dataset requires a HuggingFace account with approved access. Before running `02_main_analysis.ipynb`, authenticate:
```python
from huggingface_hub import login
login()
```

### Run order
1. (Optional) `01_baseline_exploration.ipynb` — exploratory analysis on Babel Briefings
2. `02_main_analysis.ipynb` — full analysis pipeline; produces all figures

---

## Dependencies

Core libraries: `datasets`, `huggingface-hub`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `scipy`, `vaderSentiment`, `tqdm`

See `requirements.txt` for pinned versions.

---

## Citation

If you reference this work, please cite:

```bibtex
@misc{heard2026overton,
  author       = {Heard, Paris},
  title        = {Mapping the {Overton} {Window}: Discourse Shift in {U.S.} News Headlines, 2001--2024},
  year         = {2026},
  institution  = {School of Information, University of Michigan},
  note         = {Course project, SI 630: Natural Language Processing},
  url          = {https://github.com/YOUR_USERNAME/overton-window-nlp}
}
```

---

*School of Information · University of Michigan · Winter 2026*
