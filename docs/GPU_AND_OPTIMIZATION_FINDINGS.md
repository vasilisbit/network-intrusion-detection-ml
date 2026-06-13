# GPU Acceleration & Optimization — Full Findings Log

Complete record of the performance/scaling work on this project (branch `gpu-dev`), for the
report write-up and oral defense. Covers: scikit-learn acceleration choices, the sampling
redesign, the WSL2 + RAPIDS cuML GPU setup, every measured number, and the algorithm-scaling
lessons (degeneracy, O(n²) walls).

Hardware: Intel i7-10700 (8c/16t), NVIDIA RTX 2060 (6 GB VRAM), 17 GB RAM, Windows + WSL2.
Software: Python 3.12, scikit-learn 1.8.0, NumPy 2.4. RANDOM_STATE=42 throughout.

---

## 1. CPU acceleration: scikit-learn-intelex (Q2 only)

- `scikit-learn-intelex` (Intel oneDAL) patches sklearn estimators to run on Intel CPU kernels.
- Installed `scikit-learn-intelex==2026.0.0` (needs scikit-learn>=1.0; works with our 1.8.0).
- Used in **Q2 only**, via `patch_sklearn()` in a try/except at the top of the imports cell.
- Measured (Q2 data): RandomForest ~4x, KMeans ~25x, DBSCAN ~2.8x faster; results identical.
- **REJECTED for Q3** — see the degeneracy section below.
- The NVIDIA GPU cannot use intelex: oneDAL is SYCL/Intel-GPU only, not CUDA.

## 2. Sampling redesign — `utils/helpers.py: subsample_to_target(...)`

The original (main) used a per-class cap `SUBSAMPLE_CAP = min(class_size, cap)`. We replaced it
with a target-size API:
- `target_rows` (e.g. 1_000_000) or `target_frac`, with `rare_floor=1000`.
- Default **floored proportional**: classes ≤ rare_floor kept in full; remaining budget split
  among big classes IN PROPORTION to their true size; overflow redistributed; largest-remainder
  rounding hits the target exactly.
- `balanced=True` option: splits the big-class budget EVENLY instead of proportionally.

WHY floored-proportional and not pure proportional: CIC-IDS-2017 is ~83% BENIGN with three
attack classes at 11/21/36 rows. Pure proportional to 150k would give Heartbleed→1 row,
SQL-Injection→1 row — unsplittable for train/test and fatal for StratifiedKFold(5). The
rare-class floor fixes that.

Sampling decisions:
- **Q2**: `balanced=False` (proportional). The realistic 83% BENIGN prior IS the point — it
  makes LR collapse on rare classes, which is the answer to the assignment's imbalance question.
- **Q3**: `balanced=False` (proportional), per user — "represents the whole dataset better."
  Consequence: best-k is honestly 2 (normal-vs-anomalous). The `balanced` toggle stays available
  for the report's comparison.

## 3. THE BIG LESSON: drop-in accelerators can change RESULTS, not just speed

Two separate accelerators produced **degenerate clusterings** on this data. Both were caught by
A/B testing against stock sklearn BEFORE adoption. This is the headline methodological point.

### 3a. intelex KMeans (Q3) — REJECTED
- A/B on identical 134k balanced data, stock vs intelex (only the patch differs):
  - stock k=2: clusters [102k, 32k], silhouette 0.374 (honest)
  - intelex k=2: clusters **[134163, 24]**, silhouette **0.955** (degenerate — isolates 24 outliers)
- That fake 0.955 hijacks the best-k `argmax`, forcing a meaningless k=2.
- FIX: Q3 does NOT call `patch_sklearn()`. KMeans/PCA/Hierarchical run on stock sklearn (CPU)
  or cuML (GPU, see below). Only DBSCAN was the original GPU target.

### 3b. cuML KMeans default init — FIXED with init='k-means++' + auto-guard
- A/B on 150k proportional, cuML vs sklearn (same PCA space):
  - cuML default `init='scalable-k-means++'`: k=2 → **[149998, 2]**, fake sil 0.9848. DEGENERATE.
  - cuML `init='k-means++'`: k=2 → [133627, 16373] sil 0.5750 vs sklearn [133134, 16866] sil
    0.5733. MATCHES. k=3 near-identical; k=15 sil 0.4610 vs 0.4605.
- The default init oversamples outliers into tiny clusters. `k-means++` fixes it.
- We force `init='k-means++'` AND keep a runtime **auto-guard** `fit_kmeans_guarded()`: if any
  cuML cluster holds < 0.1% of points, it refits that k on stock sklearn CPU and prints `[guard]`.
- NOTE (observed at 500k): the guard fires on MANY k values (k=6..15), so at large N the cuML
  KMeans speedup partly evaporates — most k end up refit on CPU anyway. The guard is doing its
  job (protecting correctness), but cuML KMeans is unreliable on this data at scale.

### 3c. cuML PCA — IDENTICAL, adopted freely
- 19 components for 85% var, byte-identical explained_variance_ratio to sklearn. No issues.

### 3d. cuML DBSCAN — EQUIVALENT, adopted (but see O(n²) wall)
- Matches sklearn cluster counts (e.g. ε=0.3/ms=5: 736 clusters/14,340 noise both at 150k).
- This was the GPU win that worked cleanly. But it does NOT escape O(n²) — see section 6.

---

## 4. WSL2 + RAPIDS cuML SETUP — full replication steps

cuML has **no Windows build** — it requires Linux or WSL2. Steps that worked:

### 4a. One-time WSL2 + driver check (in PowerShell / WSL)
```
wsl --status                      # confirm WSL2 + a distro (we had Ubuntu-24.04)
wsl -d Ubuntu-24.04 -- nvidia-smi # confirm WSL2 sees the GPU (driver passthrough)
```
- NVIDIA driver 591.86 (CUDA 13-capable) satisfies RAPIDS CUDA-12 wheels (needs ≥525).
- RTX 2060 = compute capability 7.5 ≥ RAPIDS minimum 7.0. No `nvcc` needed (precompiled kernels).

### 4b. Create the Linux env + install cuML (inside WSL2)
```bash
sudo apt update && sudo apt install -y python3.12-venv python3-pip
python3 -m venv ~/rapids-env
source ~/rapids-env/bin/activate
pip install --upgrade pip
# sanity: driver reachable from python
python -c "import ctypes; ctypes.CDLL('/usr/lib/wsl/lib/libcuda.so.1'); print('libcuda OK')"

# cuML (CUDA-12 wheels) from NVIDIA's index — ~3-4 GB download
pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12      # -> cuml 26.04.0
# smoke test BEFORE installing the rest
python -c "import cuml; from cuml.cluster import DBSCAN; print('cuml', cuml.__version__)"
python -c "import cupy; cupy.zeros(10); print('cupy GPU OK')"

# notebook stack (cuML already pulled sklearn 1.8.0 / numpy 2.4.6 / pandas — matches Windows)
pip install jupyter ipykernel pandas pyarrow scikit-learn matplotlib seaborn scipy

# register the kernel
python -m ipykernel install --user --name rapids-gpu --display-name "Python (RAPIDS GPU)"
```

### 4c. Use the GPU kernel from VS Code on Windows (THE GOTCHA)
VS Code runs on Windows, so the WSL kernel does NOT appear in the picker directly. Working method:
```bash
# in WSL2, env active, project dir:
jupyter lab --no-browser --port=8888
# copy the printed  http://localhost:8888/lab?token=....  URL
```
Then in VS Code: open the notebook → kernel picker → **"Existing Jupyter Server..."** → paste the
URL → select **"Python (RAPIDS GPU)"**. Works because WSL2 forwards localhost to Windows.
(Alternative: `code .` from inside WSL = a Remote-WSL window, but then the Windows .venv isn't visible.)

Confirmation it's live: the imports cell prints
`DBSCAN backend: cuML (GPU)` and `PCA/KMeans backend: cuML (GPU)`.

### 4d. Notebook portability
All GPU use is behind factories with sklearn fallback (`make_dbscan_labels`, `make_pca`,
`make_kmeans`). On plain Windows (no cuml) the notebook still runs on CPU — verified.

---

## 5. Metric computation at scale — silhouette vs Davies-Bouldin

- **Silhouette is O(n²)** (needs all pairwise distances). At 0.5-2.5M rows the full computation is
  infeasible on ANY hardware (the n×n matrix alone is TB-scale). So `evaluate_clustering` SAMPLES
  it (default 20k rows). This is a statistical estimate (±~0.01), NOT a hardware shortcut — GPU
  cannot "do it at full" either. A bigger sample (20k vs 10k) is just a tighter estimate.
- **Davies-Bouldin is O(n·k·d)** (linear). It runs on the FULL data, no cap, even at millions.
- So: silhouette sampled, DB full. This split is intrinsic to the math.

---

## 6. SCALING WALLS — what scales to the full dataset and what does NOT

Measured behavior per algorithm as N grows:

| Step              | Complexity   | 150k    | 500k        | 1M          | Scales to full? |
|-------------------|--------------|---------|-------------|-------------|-----------------|
| PCA (cuML)        | ~O(n·d²)     | 0.2s    | fast        | 0.2s        | YES (trivial)   |
| KMeans k-search   | O(n·k·iters) | ~2s GPU | 124s*       | ~2s GPU     | mostly (CPU-guard caveat) |
| DBSCAN (cuML)     | **O(n²)**    | 120s    | **1409s**   | ~5min/config| **NO**          |
| Hierarchical      | **O(n²) MEM**| (subsample only) | — | —      | **NEVER** (subsample mandatory) |
| Silhouette        | O(n²)        | sampled 20k | sampled | sampled  | sampled by design |
| Davies-Bouldin    | O(n)         | full    | full        | full        | YES             |

*500k KMeans 124s is inflated by the auto-guard refitting most k on CPU (cuML degenerate).

### 6a. Why DBSCAN's first config took ~5 min at 1M (user's question)
cuML DBSCAN uses **`algorithm='brute'`** internally (no kd-tree/ball-tree). It builds the neighbor
graph by brute-force pairwise distances = **O(n²)**. Evidence: ε=0.3/ms=5 took 4.4s @150k,
**52.7s @500k** (3.3× rows → ~12× time = quadratic), ~5min @1M. The GPU gives a huge CONSTANT-factor
speedup (it made 150k feel instant) but cannot change the complexity class. Small ε is slowest
(most border-point bookkeeping). This is the SAME O(n²) density wall that froze the CPU run —
cuML moved it from "impossible at 150k" to "minutes at 1M," not gone.
- Practical guidance: DBSCAN should stay on a smaller sample (≤150-200k) even when KMeans/PCA use
  the full set, OR accept minutes-per-config. It does not belong at 1M+.

### 6b. Hierarchical clustering & HIER_SAMPLE_SIZE (user's question)
- `HIER_SAMPLE_SIZE` WAS in main (=3000), unchanged by us. It is NOT new.
- WHY it exists: Agglomerative needs the full n×n distance matrix = **O(n²) MEMORY**:
  n=3,000 → 0.07 GB; n=10,000 → 0.8 GB; n=20,000 → 3.2 GB; n=50,000 → 20 GB; n=100,000 → 80 GB.
  So a subsample is MANDATORY — physically impossible otherwise. cuML has no Agglomerative either.
- "Can we have it all?" NO — hierarchical is the one algorithm that cannot scale on any hardware.
- WHY the silhouette/graph changes with sample size: silhouette is computed on whatever subsample
  is fed in. The stratified subsample artificially BALANCES classes (200 each), so clusters look
  cleanly separated → high silhouette (0.92 @500k-run sample, 0.886 @150k-run sample). The full
  data's real overlap is not represented. So a BIGGER HIER_SAMPLE_SIZE gives LOWER, MORE HONEST
  silhouette, not "better" results. Bigger ≈ more realistic, not better. Safe ceiling ~10k
  (0.8 GB, readable-ish dendrogram); past ~20k both memory and dendrogram readability break down.

---

## 7. Q3 RESULTS by sample size (all stock-sklearn KMeans / cuML DBSCAN, proportional)

| Quantity                  | 150k run        | 500k run        |
|---------------------------|-----------------|-----------------|
| PCA comps for 85%         | 19 (cum 0.8662) | 19 (cum 0.8627) |
| KMeans best k (silhouette)| 2 (sil 0.5733)  | 2 (sil 0.5653)  |
| KMeans k=15 ref           | sil 0.4363      | sil 0.4300, DB 0.803 |
| Hierarchical best         | average 0.886*  | complete/avg 0.9201* |
| DBSCAN best config        | ε0.3/ms20 sil .422 | ε1.5/ms5 sil .105 |
| DBSCAN noise (best)       | 23,755 (16%)    | 2,649 (0.5%)    |
| DBSCAN full sweep time    | 120s            | 1409s (23 min)  |
| KMeans k=2 heatmap        | c0 89% BENIGN / c1 49% DoS Hulk+12% DDoS | same shape |

*Hierarchical high silhouette is partly the balanced-subsample artifact (see 6b); cross-check
against the cluster-to-label heatmap, which shows degenerate/outlier-isolation splits at k=2.

KEY FINDING (stable across sizes): silhouette-best k = 2 = "normal traffic vs volumetric DoS/DDoS".
Scaling up does NOT change this — the BENIGN blob dominates more, if anything. The cluster-to-label
HEATMAP (not the scatter) is the real "do clusters match attack categories?" evidence → answer:
partially (clean DoS/DDoS/PortScan pockets; BENIGN is heterogeneous; rare attacks fall into noise).

EXTENDED-K TEST (k up to 50, proportional 500k, cuML): k=2 STILL WINS (sil 0.5655). Silhouette at
high k climbs back toward k=2 (k=50: 0.5500) ONLY because KMeans shards into tiny outlier clusters —
"smallest cluster %" collapses from 11.3% (k=2) to ~0% (k>=12). That is silhouette being GAMED by
near-zero-internal-distance shards, not real structure (same family as the degenerate-split trap).
Lesson for report: silhouette alone is gameable both ways (one huge degenerate cluster OR many tiny
ones); always cross-check cluster sizes + the label heatmap.

REPORT CAVEAT (k=2 interpretation — also in q3 notebook in Greek): k=2 winning does NOT mean there are
only 2 attack types. It means the 15 labels are NOT cleanly separable as 15 geometric balls in this
PCA space under proportional (83% BENIGN) sampling. The strongest geometric signal is binary (normal
vs flood/volumetric); the finer 15-way structure exists but is tangled/overlapping — an honest,
defensible finding.

cuML LR on BALANCED 500k (empirical, single fit C=10): STILL diverges from sklearn — sklearn f1_macro
0.6783 vs cuML pn=True 0.6933 (+0.015), pn=False 0.3580 (broken). Confirms the mismatch is
solver/regularization (penalty_normalized changes effective C), NOT sampling. Q2 stays sklearn+intelex,
now proven on balanced data. (Note: sklearn LR hit the 2000-iter cap at 500k — LR is hard to fully
converge at this scale for any backend; reported lbfgs numbers are best-effort, true of all backends.)

---

## 7b. Q2 RESULTS — balanced sampling, 500k (MAIN run, A==B identical test metrics)
Train/val/test = 300k/100k/100k. Balanced: BENIGN/DDoS/DoS Hulk ~75k each, PortScan ~54k, rares full.
| Model | best params | f1_weighted | f1_macro |
|---|---|---|---|
| Decision Tree | entropy, max_depth None(A)/20(B), mss=2 | 0.9973 / 0.9971 | **0.8705 / 0.8691** (WINNER) |
| Random Forest | n_est=200, max_depth=None, mss=2 | 0.9927 | 0.7403 |
| Logistic Regression | C=10, lbfgs, max_iter=500 | 0.9548 | 0.6703 |
Grid times (500k): LR-A 454s, LR-B(CV) 1718s, DT-A 144s, DT-B 267s, RF-A 362s, RF-B 2174s.
Shallow DT(depth=3) test F1=0.1768 (viz only). DT importance: Dest Port 0.464 dominant.
RF importance flat: Dest Port 0.055, top-10 ~0.04-0.05 (feature randomness spreads signal).

### balanced vs proportional COMPARISON (the critique payoff) — both at ~150-500k:
| Model | f1_macro PROPORTIONAL | f1_macro BALANCED (main) |
|---|---|---|
| Logistic Regression | 0.4986 | 0.6703  (+0.17) |
| Random Forest | 0.6138 | 0.7403  (+0.13) |
| Decision Tree | 0.8803 | 0.8705  (~same) |
=> balanced helps LR/RF a lot, DT barely (deep tree catches rare classes regardless). Confirms the
rule: supervised training set must represent class BOUNDARIES, not the real prior. Proportional run
kept in the report as the "why we balance" evidence. (Q3 is the mirror image: proportional = main.)

## 8. Other settings touched
- LR (Q2): solver grid = ['lbfgs'] only (saga ranked 7th-9th and never won; newton-cholesky was
  ~10x slower and OOM-killed the kernel with n_jobs=-1). LR-B GridSearchCV n_jobs=4 (not -1) to
  bound memory. RF-B/DT-B keep n_jobs=-1 (fine).
- PCA scatter plots: `plot_sample_size=50_000` RENDERING cap (a 2-D scatter saturates visually;
  metrics use full data; tunable knob).
- Notebook cosmetics: emojis removed, em-dash → hyphen, intelex/GPU explanation markdown added.
