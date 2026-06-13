# Network Intrusion Detection - CIC-IDS-2017

Lab project for **Data Mining and Learning Algorithms**, Spring 2025-2026,
University of Patras (Dept. of Computer Engineering & Informatics).

End-to-end ML pipeline on the CIC-IDS-2017 dataset:

- **Q1 - EDA, cleaning, feature selection**
- **Q2 - Classification** - Logistic Regression, Decision Tree, Random Forest, each with split- (Scenario A) and 5-fold CV-based (Scenario B) grid search. Best model: **Decision Tree** (F1-macro 0.87).
- **Q3 - Clustering** - K-Means, Hierarchical, DBSCAN with PCA visualization. Dominant structure: a binary *normal vs volumetric-DoS* split (silhouette-best k=2).

Both Q2 and Q3 run at a configurable sample size (`TARGET_ROWS`) and sampling strategy (`BALANCED_SAMPLING`). Q2 uses **balanced** sampling (a classifier must learn class *boundaries*); Q3 uses **proportional** sampling (clustering should see the true density). The presented runs use 500,000 rows.

Optional hardware acceleration (both with graceful CPU fallback):

- **Q2**: Intel `scikit-learn-intelex` (oneDAL, CPU) - bit-identical to stock scikit-learn, just faster on the tree models.
- **Q3**: NVIDIA `RAPIDS cuML` (GPU, via WSL2) for PCA / K-Means / DBSCAN - e.g. the DBSCAN sweep drops from ~58 min (CPU) to ~2 min (GPU) at 150k.

Key optimization findings and scaling limits are summarized in the [Optimization notes](#optimization-notes) section below.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
python -m ipykernel install --user --name nids --display-name "Python (nids)"
```

Requires Python 3.10+ and ~16 GB RAM (the full load uses ~4-6 GB after memory optimization).

**Optional accelerators** (the notebooks auto-detect them and fall back to stock scikit-learn if absent):

```bash
# Q2 - Intel CPU acceleration (x86_64 only; pip, same venv)
pip install scikit-learn-intelex

# Q3 - NVIDIA GPU via RAPIDS cuML: Linux/WSL2 only, separate env.
#   python3 -m venv ~/rapids-env && source ~/rapids-env/bin/activate
#   pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12
#   python -m ipykernel install --user --name rapids-gpu --display-name "Python (RAPIDS GPU)"
# See the "GPU setup (WSL2 + RAPIDS)" subsection below for full steps.
```

## Getting the data

Download the CIC-IDS-2017 MachineLearningCSV.zip from <https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/> (or the official UNB mirror) and extract the 8 CSV files into `./data/`.

After extraction, `./data/` should contain:

```text
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## Running the notebooks

Open each notebook in order - they depend on artifacts saved by the previous one (Q1 produces `data/cic_ids_2017_clean.parquet`, which Q2/Q3 load directly):

1. `notebooks/q1_eda.ipynb`            - EDA, cleaning, feature selection
2. `notebooks/q2_classification.ipynb` - supervised models (LR / DT / RF)
3. `notebooks/q3_clustering.ipynb`     - unsupervised models (K-Means / Hierarchical / DBSCAN)

In the Q2/Q3 config cell you can adjust `TARGET_ROWS` (`150_000` / `500_000` / `None` for the full dataset) and `BALANCED_SAMPLING` (Q2 defaults to `True`, Q3 to `False`). Flip `BALANCED_SAMPLING` to reproduce the comparison runs. For GPU-accelerated Q3, run it under the `rapids-gpu` kernel inside WSL2.

To execute a notebook headlessly:

```bash
jupyter nbconvert --to notebook --execute notebooks/q1_eda.ipynb --output q1_eda.ipynb
```

## Layout

```text
data/                raw CSVs (gitignored) + cleaned Parquet snapshot
notebooks/           Q1/Q2/Q3 analysis notebooks
utils/
  __init__.py        re-exports the public helper API
  helpers.py         loading, cleaning, caching, sampling, plotting helpers
scripts/             auxiliary build/automation scripts
outputs/
  figures/           saved plots (PNG)
  results/           saved metrics tables (CSV)
report/
  doc.tex            full LaTeX report (compile with XeLaTeX)
  exam_notes.md      oral-exam prep notes (Greek + English terms)
exercise.pdf         official assignment brief
prompt.md            project specification / teaching contract
requirements.txt     pinned library list
```

## Reproducibility

`utils.RANDOM_STATE = 42` is used for every split, model seed, and stochastic operation (sampling, shuffling, K-Means init, RF bootstrap, GridSearchCV folds, PCA solver). Re-running any notebook top-to-bottom on the same backend yields bit-identical results. Note that GPU (cuML) and CPU (scikit-learn) can differ slightly on the same seed - clustering metrics may shift at the 3rd–4th decimal; for the **graded** classification metrics Q2 deliberately stays on scikit-learn (+intelex), which is bit-identical to stock scikit-learn.

## Optimization notes

### Sampling strategy (`subsample_to_target`)

`utils.subsample_to_target(y, target_rows, target_frac, rare_floor, balanced)` controls dataset size and class mix. Classes with `<= rare_floor` rows (default 1000) are always kept in full; the remaining budget is split among the large classes either **evenly** (`balanced=True`) or **proportionally** to their true frequency (`balanced=False`).

- **Q2 = balanced.** A classifier's training set should represent the *boundaries* between classes, not the real prior. Proportional sampling here starves the medium-rare classes (e.g. Bot 1948 -> 113, Web Attack Brute Force 1470 -> 85) and even inverts frequencies (the rare-floor keeps XSS=652 whole while cutting Brute Force to 85). Balanced sampling lifted LR F1-macro from 0.50 to 0.67 and RF from 0.61 to 0.74; DT was unchanged (a deep tree catches rare classes regardless).
- **Q3 = proportional.** Clustering should see the true density. The honest result is silhouette-best **k=2** (normal vs volumetric DoS). This does *not* mean only 2 attack types exist - it means the 15 labels are not cleanly separable as 15 geometric balls in PCA space under an 83%-BENIGN sample. Balanced sampling would push best-k toward ~15; both are reported.

### Hardware acceleration

- **Q2 - Intel `scikit-learn-intelex` (CPU).** Patches scikit-learn via `from sklearnex import patch_sklearn; patch_sklearn()` (in a try/except). Verified **bit-identical** to stock scikit-learn, so it is safe for the graded metrics. We tried cuML (GPU) and `SGDClassifier` mini-batching for LR; both were rejected - their solver/regularization differs from scikit-learn's `lbfgs`, shifting F1-macro by ~0.15 (not reproducible).
- **Q3 - NVIDIA `RAPIDS cuML` (GPU).** Used for PCA, K-Means, and DBSCAN. PCA is numerically identical to scikit-learn; DBSCAN gives equivalent cluster counts. cuML K-Means must use `init='k-means++'` (the default `scalable-k-means++` produces a degenerate split - one ~24-point cluster with a fake ~0.95 silhouette that hijacks best-k selection), and the notebook keeps a runtime guard that refits on CPU if a result still looks degenerate. Hierarchical clustering stays on CPU (no cuML equivalent). **Lesson:** a drop-in accelerator can change *results*, not just speed - always A/B against stock scikit-learn before trusting it.

### GPU setup (WSL2 + RAPIDS)

cuML has no Windows build; it requires Linux or WSL2 with an NVIDIA driver >= 525 (CUDA 12). No conda or `nvcc` needed - the wheels ship precompiled kernels.

```bash
# inside WSL2 (Ubuntu), with the NVIDIA driver visible (`nvidia-smi` works):
sudo apt install -y python3.12-venv python3-pip
python3 -m venv ~/rapids-env && source ~/rapids-env/bin/activate
pip install --upgrade pip
pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12
pip install jupyter ipykernel pandas pyarrow scikit-learn matplotlib seaborn scipy
python -m ipykernel install --user --name rapids-gpu --display-name "Python (RAPIDS GPU)"
```

To use the kernel from VS Code on Windows: in WSL2 run `jupyter lab --no-browser --port=8888`, copy the printed `http://localhost:8888/...?token=...` URL, then in VS Code pick **"Existing Jupyter Server..."**, paste the URL, and select the **Python (RAPIDS GPU)** kernel (WSL2 forwards `localhost` to Windows). The first Q3 cell prints `DBSCAN backend: cuML (GPU)` / `PCA/KMeans backend: cuML (GPU)` when it is active, or `scikit-learn (CPU)` on the fallback path.

### Scaling limits

- **DBSCAN is O(n²)** (cuML uses a brute-force neighbor graph). The full 15-config sweep is ~2 min at 150k on GPU but ~23 min at 500k - the GPU gives a large constant-factor speedup, not a lower complexity class. It is not practical much beyond 500k.
- **Hierarchical is O(n²) in memory** (full distance matrix: ~0.07 GB at 3k rows, ~20 GB at 50k, ~80 GB at 100k), so it always runs on a small stratified subsample (~2.5k rows). There is no GPU workaround.
- **Silhouette is O(n²)**, so it is always estimated on a 20k-row sample. **Davies-Bouldin is O(n)** and runs on the full data.
