# Personalised Federated Learning for Cross-Market Portfolio Allocation

**CS437/CS5317 Deep Learning — Spring 2026 | Group 30**
Muhammad Umar Malik (27100139) · Muhammad Moosa Kashif (27100451)

---

## Overview

This project studies federated learning for portfolio allocation across four international equity markets: S&P 500, FTSE 100, Nikkei 225, and KSE-100. Each market is a separate federated client holding 30 stocks. No raw data is shared between clients.

The central questions are: does FedAvg produce asymmetric per-client benefits when markets are structurally different, and can personalised FL correct any imbalance? We evaluate eight personalisation methods across every major paradigm and connect our findings to the cross-silo personalisation phase transition of Liu et al. (NeurIPS 2022).

**Key findings:**
- FedAvg improves over local-only training for all four markets, with the gain predicted by a pre-training idiosyncrasy score (Pearson r = +0.71)
- SCAFFOLD reduces seed variance 8× on S&P 500 and is bootstrap-significant over FedAvg on two markets
- Eight personalisation methods — spanning regularisation, meta-learning, mixture models, output interpolation, decoupled training, and similarity-weighted aggregation — all fail to significantly improve any client
- The optimal personalisation parameter collapses to fully-global or fully-local for every method and every market, consistent with the theoretical cross-silo phase transition

---

## Repository Structure

```
dl-project/
├── notebooks/
│   ├── expanded_dataset.ipynb   # Data pipeline, EDA, heterogeneity metrics
│   ├── baselines.ipynb          # Six baselines: EqWt, MeanVar, Local, Central, FedAvg, FedProx
│   ├── improvement1.ipynb       # FT-k, Split-FL, Ditto
│   └── improvement2.ipynb       # SCAFFOLD, Per-FedAvg, FedEM, APFL, pFedGraph, Transformer
├── research_paper/
│   └── 30_27100139_27100451_Report.pdf
├── soa_survey/
│   └── D1-SOA.pdf
└── README.md
```

---

## Dataset

Price data for all four markets is available on Kaggle:

**[https://www.kaggle.com/datasets/umarmalikk/updated-set](https://www.kaggle.com/datasets/umarmalikk/updated-set)**

- S&P 500, FTSE 100, Nikkei 225: 30 stocks each, downloaded via Yahoo Finance
- KSE-100: 30 stocks, manually sourced (Yahoo Finance coverage is unreliable for Pakistani equities)
- Period: January 2010 – April 2026
- Format: daily adjusted closing prices, one CSV per market

**Splits:** train 2010–2019 | validation 2020–2021 | test 2022–April 2026. All splits are strictly chronological with no overlap.

To use the data locally, download the dataset from Kaggle and update `DATA_DIR` in the config cell of each notebook to point to your local path. The Kaggle path used during training is already set as the default.

---

## Notebooks

Each notebook is self-contained and runs end-to-end. Run them in order.

| Notebook | What it does |
|---|---|
| `expanded_dataset.ipynb` | Loads prices, computes log returns, engineers 6 features per stock, runs EDA (cumulative returns, volatility, correlations), computes Hellinger Distance and idiosyncrasy scores |
| `baselines.ipynb` | Trains and evaluates equal-weight, mean-variance, local-only, centralised, FedAvg, and FedProx. Includes window ablation, FedProx mu ablation, survivorship bias check, and regime analysis |
| `improvement1.ipynb` | Post-FedAvg fine-tuning (FT-k), Split-FL, and Ditto. Includes per-client LR selection and k-ablation |
| `improvement2.ipynb` | SCAFFOLD, FT-k on SCAFFOLD, Ditto on SCAFFOLD, Per-FedAvg (FOMAML), FedEM, APFL, pFedGraph, and a capacity-matched Compact Transformer experiment |

All experiments use 10 random seeds (42–51). Significance is tested with a block bootstrap (5,000 resamples, block = 21 days) on pooled daily returns, with dual Jobson-Korkie + Wilcoxon as a secondary check.

---

## Setup

```bash
pip install torch numpy pandas matplotlib seaborn scipy yfinance
```

All notebooks were run on Kaggle (T4 GPU). The notebooks detect the available device automatically (CUDA / MPS / CPU). Expected runtimes on T4:

| Notebook | Approx. runtime |
|---|---|
| expanded_dataset | ~5 min |
| baselines | ~25 min |
| improvement1 | ~35 min |
| improvement2 | ~90 min |

---

## Architecture

All FL experiments use a 2-layer LSTM (hidden = 96, dropout = 0.1) with a linear softmax head over 30 assets — 184K parameters. Input: (batch, 100, 180) where 100 is the lookback window and 180 is 6 features × 30 stocks. Output: portfolio weights summing to 1 (long-only, softmax).

Training loss: negative Sharpe ratio with a turnover penalty (λ = 0.01). Transaction costs of 10 bps one-way applied at evaluation.

---

## Results Summary

| Method | SP500 | FTSE | KSE | Nikkei |
|---|---|---|---|---|
| Local-only | 0.893 | 0.464 | 0.824 | 0.762 |
| FedAvg | 0.950 | 0.510 | 0.938 | 0.866 |
| SCAFFOLD | 0.999 | 0.596 | 0.985 | 0.911 |
| Per-FedAvg k=0 | 1.015 | 0.609 | 0.992 | 0.920 |
| Best personalisation | ≤FedAvg | ≤FedAvg | ≤FedAvg | ≤FedAvg |

Sharpe ratio, mean across 10 seeds, test period 2022–April 2026, 10 bps TC one-way.

---

## Citation

If you use this code or data pipeline, please cite the research paper included in `research_paper/`.
