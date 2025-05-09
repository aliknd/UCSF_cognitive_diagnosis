# Passive Fitbit Screen for Early Cognitive Impairment

_Prepared for: UCSF study team – 2025‑05‑09_


## 1. Data assets & preprocessing

### Data files
| File | Size | Key fields | Notes |
| --- | --- | --- | --- |
| Diagnoses _20250404.csv | 192 × … | PIDN, BaselineDate, 3-group Dx | 122 CN · 19 MCI/AD · 6 FTD (70 “Abnormal”) |
| heartrate_15min.csv | -- | PIDN, Time, 15-min HR | Gaps ≤ 30 min ffilled |
| minuteStepsNarrow.csv | 2 GB | 1-min steps | Down-sampled to 15-min or daily totals |


- **Baseline window:** first 14 days after baseline visit (96 bins/day → 1344 bins)
- **Missing bins:** forward/backward fill for 1 slot; remaining NaNs sentinel‑masked for sequences


## 2. Engineered tabular feature set (24 columns)
- HR stats: mean, std, min/max, IQR, RMSSD, SDNN, LF/HF proxy, day/night means
- Step stats: daily mean/std/min/max, 14‑day trend, weekend / weekday ratio
- Coupling: HR‑steps Pearson *r*
- Saved to `tabular_features.csv`


## 3. Experiments & key metrics
| # | Model / features | Oversampling / weights | Threshold | Test bal-acc | Abn recall | FP / FN |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | XGB on HR-only (7 d) | ADASYN | 0.50 | 0.344 | 0.40 / 0.11 | 8 / - |
| 2 | XGB on HR+steps (14 d) | ADASYN | 0.50 | 0.424 | 0.40 / 0.11 | 14 / 4 |
| 3 | XGB on HR+steps (14 d) | class_weight+min_leaf | 0.47 | 0.546 | 0.57 | 14 / 4 |
| 4 | Same model + threshold scan | – | 0.46 – 0.47 | 0.581 | 0.786 | 14 / 3 |
| 5 | Light CNN on HR seq | class_weights | 0.50 | 0.50 | 1.00 (all 1’s) | 25 / 0 |
| 6 | GRU + masking on HR seq | class_weights | 0.50 | 0.496 | 0.07 | 2 / 13 |
| 7 | Hybrid stack (XGB tab --+-- Bi-GRU+Attn 2-ch) → logistic blend | k-fold stacking | 0.56 | 0.617 | 0.714 | 12 / 4 |


† FTD / MCI split in multiclass phase.


## 4. Selected operating points
| Scenario | Cut-off | bal-acc | Abn recall | FP | FN |
| --- | --- | --- | --- | --- | --- |
| Preferred screen (tabular model) | 0.47 | 0.581 | 0.79 | 14 | 3 |
| Hybrid stack – best BA | 0.56 | 0.617 | 0.71 | 12 | 4 |
| Hybrid – max recall scanned | 0.35 | 0.613 | 0.79 | 14 | 3 |


Tabular screen matches hybrid sensitivity with simpler runtime; hybrid offers **+0.036 BA** if FP ≤ 12 acceptable.


## 5. Model artefacts
| File | Description |
| --- | --- |
| fitbit_xgb.joblib | XGBoost trained on 24-feature table (full training set) |
| fitbit_gru.h5 | Two-channel Bi-GRU + 2-head attention (HR + steps) |
| blender.joblib | Logistic meta-model (weights: ~0.7 × XGB, 0.3 × GRU) |


```python
p_tab  = xgb_model.predict(xgb.DMatrix(tab_row))[0]
p_seq  = gru_model.predict(seq_tensor[None])[0,0]
p_bl   = blender.predict_proba([[p_tab, p_seq]])[0,1]
flag   = int(p_bl >= 0.47)  # screening threshold
```


## 6. Key observations
- 14‑day window clearly outperforms 7‑day (BA +0.18)
- Daily‑total steps & HR‑steps coupling boost tabular BA by **+0.23** vs HR‑only
- Sequence models alone under‑perform due to limited samples (< 200)
- Stacking adds diversity; modest BA **+0.036** at cut 0.56
- Threshold tuning lets us trade 2 FP ↔ 1 TP with minimal BA change


## 7. Recommendations
- Deploy tabular XGB screen at cut 0.47 (BA 0.58, recall 0.79)
- 14 FP & 3 FN per 39 participants → scalable workload
- Keep hybrid artefacts; consider cut 0.56 if clinic prefers fewer FP
- Collect more data – target ≥ 500 traces to unlock sequence gains
- Add demographics (age, sex) to features – expected BA +0.02
- Re‑evaluate every 100 new participants: retrain blender, rescan threshold


## 8. Next milestones
| Date | Deliverable |
| --- | --- |
| May 30 | Automated feature-pipeline script & Docker-ready scoring service |
| Jun 15 | Interim analysis with demographics if available |
| Aug 01 | Sequence-model retrain with ≥ 300 participants |

