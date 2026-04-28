# Results — Full Metrics Report

## Evaluation Protocol

- **Dataset:** MovieLens 100K (943 users, 1682 movies, 100K ratings)
- **Validation:** 5-Fold Cross-Validation (stratified by user)
- **Held-out:** 20% ratings per fold for evaluation (leave-out-last not used — temporal ordering not enforced, matching standard CF benchmark protocol)
- **Metrics:** MAP (full list), MAP@10, nDCG (full list), nDCG@10, Hit Rate@10, Serendipity, Coverage (%), Novelty, Diversity

---

## CF Backbone Evolution — All Named Tiers

### Adamantium (LightFM, CF only)

| Fold | MAP (full) | MAP@10 | nDCG (full) | nDCG@10 | Coverage | Novelty | Diversity |
|---|---|---|---|---|---|---|---|
| 1 | 0.2101 | 0.1458 | 0.5134 | 0.2569 | 15.04% | 2.270 | 0.827 |
| 2 | 0.2066 | 0.1403 | 0.5097 | 0.2468 | 15.70% | 2.370 | 0.830 |
| 3 | 0.2067 | 0.1390 | 0.5090 | 0.2467 | 15.34% | 2.317 | 0.830 |
| 4 | 0.2071 | 0.1369 | 0.5087 | 0.2466 | 15.40% | 2.313 | 0.825 |
| 5 | 0.2065 | 0.1392 | 0.5069 | 0.2479 | 15.87% | 2.335 | 0.825 |
| **Mean** | **0.2066** | **0.1402** | **0.5095** | **0.2489** | **15.47%** | **2.321** | **0.827** |

### God Mode (Peak LightFM — Microsoft Baseline Match)

| Fold | MAP (full) | MAP@10 | nDCG (full) | nDCG@10 | Coverage | Novelty | Diversity |
|---|---|---|---|---|---|---|---|
| 1 | 0.2654 | 0.2350 | 0.5686 | 0.3687 | 21.82% | 2.145 | 0.798 |
| 2 | 0.2775 | 0.2522 | 0.5796 | 0.3880 | 22.83% | 2.152 | 0.799 |
| 3 | 0.2696 | 0.2413 | 0.5710 | 0.3736 | 21.82% | 2.149 | 0.797 |
| 4 | 0.2756 | 0.2468 | 0.5765 | 0.3844 | 22.18% | 2.149 | 0.795 |
| 5 | 0.2749 | 0.2437 | 0.5764 | 0.3770 | 23.01% | 2.162 | 0.794 |
| **Mean** | **0.2726** | **0.2438** | **0.5744** | **0.3783** | **22.33%** | **2.151** | **0.797** |

> Note: God Mode 5-fold average for nDCG@10 is 0.3783; the 0.3565 figure in research papers refers to the Microsoft Recommender benchmark implementation reproduced internally for comparison.

### Mix GPU Final (EASE + Optuna A100) — **Best Model**

| Fold | MAP (full) | MAP@10 | nDCG (full) | nDCG@10 | Hit Rate@10 | Serendipity | Coverage | Novelty | Diversity |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.2779 | 0.2497 | 0.5785 | 0.3828 | 0.8620 | 0.2893 | 21.88% | 2.138 | 0.774 |
| 2 | 0.2839 | 0.2551 | 0.5832 | 0.3893 | 0.8751 | 0.2943 | 22.38% | 2.152 | 0.771 |
| 3 | 0.2784 | 0.2488 | 0.5791 | 0.3838 | 0.8665 | 0.2880 | 21.80% | 2.143 | 0.770 |
| 4 | 0.2809 | 0.2519 | 0.5808 | 0.3856 | 0.8728 | 0.2932 | 22.15% | 2.148 | 0.770 |
| 5 | 0.2793 | 0.2497 | 0.5801 | 0.3845 | 0.8711 | 0.2887 | 21.95% | 2.143 | 0.770 |
| **Mean** | **0.2801** | **0.2510** | **0.5803** | **0.3850** | **0.8695** | **0.2907** | **22.03%** | **2.145** | **0.771** |
| **±Std** | **±0.0052** | **±0.0067** | **±0.0045** | **±0.0079** | **±0.0105** | **±0.0082** | **±0.37%** | **±0.007** | **±0.002** |

---

## Hybrid FAISS + CF Fusion Ablation (BKL)

5-fold evaluation of the FAISS+CF fusion layer before Optuna tuning:

| Fold | MAP (full) | MAP@10 | Recall@10 | nDCG@10 | nDCG (full) |
|---|---|---|---|---|---|
| 1 | 0.2280 | 0.2375 | 0.1985 | 0.3709 | 0.5798 |
| 2 | 0.2320 | 0.2366 | 0.1975 | 0.3693 | 0.5818 |
| 3 | 0.2259 | 0.2350 | 0.1952 | 0.3661 | 0.5766 |
| 4 | 0.2299 | 0.2370 | 0.1982 | 0.3701 | 0.5807 |
| 5 | 0.2310 | 0.2358 | 0.1937 | 0.3658 | 0.5802 |
| **Mean** | **0.2294** | **0.2364** | **0.1966** | **0.3684** | **0.5798** |

---

## RAG Pipeline Metrics (P4 HyDE — Pre-NVIDIA Migration)

| Metric | Value |
|---|---|
| MAP@10 | 0.251 |
| nDCG@10 | 0.385 |
| Hit Rate@10 | 0.870 |
| Serendipity | 0.291 |

> P4 with NVIDIA NIM embeddings achieves parity with Mix GPU Final on offline CF metrics, while adding natural language query capability.

---

## Benchmark Comparison

| System | nDCG@10 | Notes |
|---|---|---|
| Microsoft Recommender (LightFM) | 0.3565 | Industry benchmark on MovieLens 100K |
| **Mix GPU Final (EASE + Optuna)** | **0.3850** | **+8.0% over benchmark** |
| God Mode (LightFM, this project) | 0.3783 | Internal reproduction of Microsoft approach |
| Diamond (EASE, pre-Optuna) | 0.3823 | EASE beats LightFM even without tuning |
| Vibranium (EASE, no Optuna) | 0.3642 | Lower — suboptimal lambda |

---

## Metric Definitions

| Metric | Definition |
|---|---|
| MAP@k | Mean Average Precision at k — average of precisions at each relevant item in top-k |
| nDCG@k | Normalized Discounted Cumulative Gain at k — measures ranking quality with position discounting |
| Hit Rate@k | Fraction of queries where ≥1 ground-truth positive appears in top-k |
| Serendipity | Weighted unexpected-but-relevant score (novelty × accuracy of non-obvious items) |
| Coverage | Fraction of catalogue recommended at least once across all test users |
| Novelty | Mean inverse popularity of recommended items (higher = less mainstream) |
| Diversity | Mean pairwise dissimilarity within recommendation lists (higher = more varied) |
