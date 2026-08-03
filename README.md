# Ciona intestinalis behavioural analysis — larval locomotion and adult contraction dynamics

Pipeline for processing DeepLabCut tracking data from *Ciona intestinalis* larvae and adults exposed to emerging contaminants (cellulose nanofibers CNF/TOCNF and antidepressant venlafaxine VEN). Computes locomotion and morphometric features, screens for outliers, reduces feature dimensionality, and projects animals into a shared behavioural space via PCA.

## Modules

| File | Description |
|------|-------------|
| `loading.py` | Parse `*filtered.csv` DLC outputs (single- and multi-animal projects), apply likelihood threshold and IQR spatial outlier filters, remove large inter-frame jumps, compute centre of mass, resample, assign condition labels; reproducible sorted directory traversal; `build_exp_meta` constructs metadata from load log to prevent groupby-sort misalignment |
| `larvae_features.py` | Centre-of-mass kinematics (speed, acceleration, angular velocity/acceleration), Savitzky–Golay body curvature and quirkiness from skeleton keypoints, sliding-window tortuosity, SVD path entropy, net displacement; arena metrics (radius of gyration, ringness, anisotropy, winding, wall fraction); batch-anchored three-state behavioural classifier (quiescent / dwelling / roaming), bout detection, transition rate |
| `adult_features.py` | Six morphometric time series from siphon and mantle landmarks (polygon area via Shapely, siphon widths and areas via shoelace formula); median normalisation, outlier clipping, rolling-mean smoothing; seasonal decomposition trend extraction with leading-edge recovery and gap fill; z-score normalisation; contraction peak detection (scipy find_peaks) with full/half classification; quiescence interval detection; three-channel majority-vote unified state classifier; event-level features (rate, IPI, CV, duration); state time budget and duty cycle; amplitude and contraction/relaxation speed in µm |
| `feature_selection.py` | Log-transform skewed metrics; CV filter; greedy Spearman correlation filter (keep higher-variance metric for \|r\| > threshold); z-score standardisation; PCA fit on all animals (controls included to anchor the behavioural space); variance and loadings tables; `full_pipeline` convenience wrapper |

## Pipeline

```
DLC *filtered.csv
  → loading.py               →  df_x.csv, df_y.csv, load_log.csv, t_axis.csv

LARVAE
  → larvae_features.py       →  kinematics/   (speed, accel, omega, curvature,
                                               quirkiness, tortuosity, path_entropy,
                                               net_displacement, arena_metrics)
                                 states/       (state_timeseries, bout_summary,
                                               state_fractions, transition_rate)

ADULTS
  → adult_features.py        →  morphometrics/  (polygon_areas, siphon_widths,
                                                 siphon_areas, mantle_areas)
                                 processed/      (*_normalised, *_trend, *_zscored)
                                 events/         (contraction_events, quiescence_events,
                                                 unified_state_map)
                                 statistics/     (event_features, state_fractions,
                                                 amp_speed_per_individual)

BOTH SPECIES
  → feature_selection.py     →  pca/   (pca_scores.csv, pca_loadings.csv,
                                        pca_variance.csv, pca_centroids.csv,
                                        pca_dropped_metrics.csv)
```

## Key parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| FPS / resample rate — larvae | 30 fps / 1:5 → 6 Hz | `loading.py` |
| FPS / resample rate — adults | 20 fps / 1:5 → 4 Hz | `loading.py` |
| Pixel to mm — larvae | 11.56 µm/px → ÷ 1000 | `larvae_features.py` |
| Physical scale — adults | 245 µm / normalised unit | `adult_features.py` |
| Likelihood threshold | 0.99 | `loading.py` |
| IQR outlier factor | 2× IQR per bodypart | `loading.py` |
| Max inter-frame jump | 100 px | `loading.py` |
| Sliding window | 3 s | `larvae_features.py` |
| Savitzky–Golay (curvature) | window 15, order 2 | `larvae_features.py` |
| Quiescence threshold | 0.05 mm s⁻¹ | `larvae_features.py` |
| Roaming threshold | batch median of control active frames | `larvae_features.py` |
| Min bout duration | 5 frames (≥0.83 s) | `larvae_features.py` |
| Completeness filter | ≥80% features observed, no imputation | `feature_selection.py` |
| Multivariate outlier screen | top 5% sum-of-squared z-scores per group | `feature_selection.py` |
| CV filter | CV ≥ 0.10 | `feature_selection.py` |
| Spearman correlation threshold | \|r\| < 0.70 (keep higher-variance) | `feature_selection.py` |
| PCA fit condition | all animals including controls | `feature_selection.py` |
| Seasonal decomposition period | 50 frames | `adult_features.py` |
| Savitzky–Golay (detection) | window 20, order 1 | `adult_features.py` |
| Peak prominence | 0.7 z-score units | `adult_features.py` |
| Min peak distance / width | 5 frames | `adult_features.py` |
| Full contraction threshold | ≥60% of individual's max peak | `adult_features.py` |

## Requirements

```
pandas numpy scipy scikit-learn statsmodels
shapely tslearn
matplotlib seaborn
```

Edit the path variables at the top of each script before running. Larvae and adult pipelines share `loading.py` and `feature_selection.py` and can be run independently.
