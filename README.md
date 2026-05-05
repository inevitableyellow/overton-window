# Peering Inside the Overton Window: A Computational Analysis of Discourse Shift in US News Headlines (2001 - 2024)

**Paris Heard** · School of Information, University of Michigan · SI 630: Natural Language Processing · Winter 2026

---

## Overview
The **Overton Window** is a political concept that describes the range of political ideas considered acceptable within public discourse, yet it is rarely operationalized using empirical or computational methods. This study presents a replicable computational framework toward quantifying Overton Window shifts through distributional analysis of mainstream US news
headlines.

Originally developed by Joseph P. Overton in the 1990s, the concept describes not just what is politically controversial, but what is uncontroversial, capturing the background assumptions that go unmarked in ordinary discourse. Rather than representing a fixed boundary, the window is recognized as dynamic, shifting constantly in response to social, political, economic, and cultural influences. Despite its widespread use as an analytical concept in political science and journalism, it has rarely been treated as an empirical object that can be measured, tracked, and validated across time.

This project treats the Overton Window as an observable linguistic phenomenon, approximated through patterns in political language used in mainstream news headlines. Headlines are chosen as the primary feature space because they are highly visible, intentionally framed by editorial decisions, and compact, making them a natural proxy for the boundaries of acceptable narratives presented to a general audience. By measuring how the distributional center of headline language shifts relative to a stable historical reference, this framework operationalizes window movement in a way that is reproducible, scalable, and applicable to future corpora.

Computational methods are applied to over 5 million U.S. news headlines spanning two decades. By treating the 2001–2015 period as a baseline and measuring how the 2016–2024 corpus drifts from it in TF-IDF space, the project quantifies how the boundaries of mainstream political discourse have shifted across five major news outlets, contributing to a growing foundation for understanding the limits of political acceptability.

**Research question:** Can the drift of the Overton Window be measured and characterized using natural language processing methods applied to large-scale news headline corpora, and is such a shift statistically significant and stable?

---

## Data

### Primary Dataset
**US Multi-Outlet News Headlines 2001–2024**  
(Chair for Data Science in the Economic and Social Sciences, University of Mannheim, 2026)  

Source: [`dess-mannheim/US_Multi_Outlet_News_Headlines2001_2024`](https://huggingface.co/datasets/dess-mannheim/US_Multi_Outlet_News_Headlines2001_2024)  
Access: **Restricted to research and academic use.** Users must not redistribute headline text. Request access via HuggingFace, then authenticate with `huggingface-hub` before running the main analysis notebook. Cite the associated dataset publication when using this data.

See data/README.md for loading instructions.

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
**Babel Briefings**  
(Leeb & Schölkopf, 2024)  

Source: [`felixludos/babel-briefings`](https://huggingface.co/datasets/felixludos/babel-briefings) on HuggingFace  
Multilingual news dataset used in early exploratory analysis (`01_baseline_exploration.ipynb`) before the primary dataset was identified. The US English subset (~92,443 headlines, August 2020 – November 2021) provided proof-of-concept validation of the centroid distance framework. 54 JSON files organized by country/language code.

Raw data files are not included in this repository. See data/README.md for loading instructions.

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
overton-window/
│
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
│
├── data/
│   └── README.md                       # data source documentation and access instructions/guidelines
│
├── notebooks/
│   ├── 01_baseline_exploration.ipynb   # early exploration using Babel Briefings dataset
│   ├── 02_main_analysis.ipynb          # final analysis utilizing US Multi-Outlet News Headlines 2001–2024 dataset
│   └── outputs/
│       └── figures/                    # notebook validation outputs and data files for reusability (when allowed)
│
├── outputs/
│   └── figures/                        # all report figures (fig1-fig7, 300 DPI)
│
└── src/
    └── overton_pipeline.py             # reusable utility functions and quick CLI processing
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
2. `02_main_analysis.ipynb` — full analysis pipeline; produces all figures **OR** `overton_pipeline.ipynb` for quick analysis/overview

---

## Dependencies

Core libraries: `datasets`, `huggingface-hub`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `scipy`, `vaderSentiment`, `tqdm`

See `requirements.txt` for pinned versions.

---

## Citation

If you reference this work, please cite:

```bibtex
@misc{heard2026overton,
  author      = {Heard, Paris},
  title       = {Peering Inside the {Overton} {Window}: A Computational Analysis of
                 Discourse Shift in {US} News Headlines (2001--2024)},
  year        = {2026},
  institution = {School of Information, University of Michigan},
  note        = {Course project, SI 630: Natural Language Processing},
  url         = {https://github.com/inevitableyellow/overton-window}
}
```

---

*School of Information · University of Michigan · Winter 2026*
