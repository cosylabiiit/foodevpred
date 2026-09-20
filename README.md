# FoodEVPred

A sequence-based machine learning pipeline for classifying food-derived proteins as extracellular vesicle (EV) cargo.

- Predicts whether a food-derived protein is EV cargo, classifying it into one of three categories: **Non-EV**, **Milk EV**, or **Plant EV**.
- Sequences are represented using embeddings from the **ProtT5** protein language model.
- Feature selection: top 384 features via LightGBM-gain importance.
- Two-tier stacked ensemble — base models: **LightGBM, SVM (RBF), MLP, KNN (cosine)**; meta-learner: **ExtraTrees**.
- Pipeline covers dataset curation (UniProt fetch + CD-HIT redundancy filtering), stratified train/test split, PLM feature extraction, model/feature selection, and a leakage-audited stacking ensemble (nested CV overfitting audit included).


## Notebooks


| File | Stage |
|---|---|
| `data-curation-cdhit.ipynb` | UniProt fetch + CD-HIT redundancy filtering |
| `test-train-split.ipynb` | Stratified train/test split |
| `feature-embeddings.ipynb` | PLM (ProtT5) feature embeddings |
| `plm.ipynb` | Model/PLM benchmarking |
| `feature-opt.ipynb` | Feature optimization |
| `stack-base.ipynb` | Base models + stacked ensemble |
| `standalone-check.ipynb` | Final pipeline fit + reproducibility check |

## Data

- `features_prott5_10639.csv`, `features_prott5_2660.csv` — ProtT5 embeddings
- `split_index.csv` — train/test split index
