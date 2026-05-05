# Data

Raw data files are **not included** in this repository. This project uses two external datasets, documented below.

---

## Primary Dataset: US Multi-Outlet News Headlines 2001–2024

**Source:** [`dess-mannheim/US_Multi_Outlet_News_Headlines2001_2024`](https://huggingface.co/datasets/dess-mannheim/US_Multi_Outlet_News_Headlines2001_2024)  
**Host:** HuggingFace  
**Access:** Restricted — research use only. You must request and be granted access before downloading.

### How to access

1. Create a HuggingFace account at [huggingface.co](https://huggingface.co)
2. Navigate to the dataset page and submit an access request
3. Once approved, authenticate locally:
   ```python
   from huggingface_hub import login
   login()
   ```
4. The dataset will be streamed automatically when you run `02_main_analysis.ipynb`

### Schema

Each outlet split contains two fields:

| Field | Type | Description |
|---|---|---|
| `date` | string | Publication date (YYYY-MM-DD) |
| `headline` | string | Article headline text |

### Outlets and sizes (raw)

| Outlet | Split name | Headlines |
|---|---|---|
| New York Times | `nyt` | 2,156,490 |
| Los Angeles Times | `lat` | 1,674,433 |
| Fox News | `foxnews` | 1,583,499 |
| Wall Street Journal | `wsj` | 724,275 |
| MSNBC | `msnbc` | 97,184 |
| **Total** | | **6,235,881** |

After filtering (minimum 4 words per headline): **5,804,010** headlines retained (6.9% removed).

---

## Secondary Dataset: Babel Briefings

**Used in:** `01_baseline_exploration.ipynb` (early exploratory analysis only)  
**Format:** JSON — 54 files organized by country/language code  
**Source:** [Babel Briefings](https://babel-briefings.com)

### Schema

Key fields used from each article object:

| Field | Description |
|---|---|
| `title` | Headline text |
| `publishedAt` / `collectedAt` | Publication date |
| `language` | ISO language code |
| `source-name` | Publisher name |

This dataset was used for initial pipeline development and is **not** part of the main analysis. Please review Babel Briefings' terms of use before downloading.

---

## Corpus Split

The main analysis divides headlines into two periods:

| Period | Years | Role |
|---|---|---|
| Baseline | 2001–2015 | Reference centroid for cosine distance measurement |
| Comparison | 2016–2024 | Corpus measured for drift from baseline |
