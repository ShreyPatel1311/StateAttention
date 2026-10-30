# StateAttention

**A 783k-parameter battery SOH trajectory forecaster in which the recurrent state queries the input.**

Given a cell's first 100 cycles, StateAttention predicts its state of health (SOH) at every future cycle up to end of life. It works on Li-ion, Na-ion and Zn-ion cells. Under [BatteryMFormer](https://github.com/Ruifeng-Tan/BatteryMFormer)'s own evaluation protocol it beats a 12.9M-parameter reproduction of BatteryMFormer on Zn-ion and ties it on Li-ion, with about 6% of the parameters ([how much to believe](#how-much-to-believe)).

> The model is called **DecayFormer** in our paper, *Small State, Long Horizon: A Hybrid State Space and Transformer Architecture for Forecasting the Trajectories of Battery Degradation* (Shrey Patel and Bhargav Srinivas Rao, University of Maryland, College Park).

| MAPE % (lower is better) | **StateAttention** (783k) | BatteryMFormer, reproduced (12.9M) | BatteryMFormer, published |
| --- | --- | --- | --- |
| Zn-ion | **4.391 ± 0.638** | 5.102 ± 0.413 | 4.970 ± 0.654 |
| Li-ion | 2.563 ± 0.082 | 2.604 ± 0.096 | **2.248 ± 0.037** |
| Na-ion | **0.921 ± 0.338** | 1.045 ± 0.350 | 1.002 ± 0.446 |
| CALB | 1.474 ± 1.297 | 1.725 ± 0.941 | 1.789 ± 0.282 |

Mean ± std across folds, 3 seeds per fold. See [Results](#results) for what each row does and does not show.

## Contents

- [Quick start](#quick-start)
- [The task](#the-task)
- [Architecture](#architecture)
- [Results](#results)
- [Protocol](#protocol)
- [Reproducing the results](#reproducing-the-results)
- [Repository layout](#repository-layout)
- [Notice on third-party code and data](#notice-on-third-party-code-and-data)
- [Citation](#citation)

---

## Quick start

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.2 and a CUDA GPU. The reported results were produced on NVIDIA A100 GPUs on the University of Maryland Zaratan cluster.

```bash
git clone https://github.com/ShreyPatel1311/StateAttention.git
cd StateAttention
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add [dev] for pytest/ruff, [gpu] for fused CUDA kernels

# Download BatteryLife and build the Zn-ion caches (~600 MB after processing)
python scripts/prepare_data.py --root data/batterylife --domains zn-ion

# Train and evaluate one fold on a single GPU (no SLURM needed)
python scripts/run_fold.py --domain zn-ion --model hx \
    --group <fold name from the split file> \
    --data-root data/batterylife --out results \
    --head-type rateshape --soc-view --epochs 25 --patience 10
```

Each run writes a result JSON to `results/`. To run the full sweep on a cluster, see [Reproducing the results](#reproducing-the-results).

Run the tests with `pytest`.

---

## The task

Given the first **S ≤ 100 cycles** of a cell's voltage, current and capacity curves, predict its SOH at every cycle after the input window, up to end of life (EOL):

```
ŷ[S+1 : t_eol] = f(G[1:S])
```

- **SOH** = `Cap_i / (nominal_capacity × DoD)`.
- **EOL** is the first cycle where SOH crosses the domain threshold: 0.80 for Li-ion and Zn-ion, 0.90 for Na-ion and CALB.
- **Metric:** MAPE on raw SOH over the **forecast window only**, i.e. the cycles after the input, up to EOL.

---

## Architecture

```
INPUT   x : [B, 100 cycles, 300 SOC-aligned points, 4 ch]    V, I, Q, SOC
  │
  ├─────────────────────────────────┐
  ▼                                 ▼
CyclePatchEmbed                  SOCView
one token per CYCLE              one token per SOC INTERVAL
  │                              (pools over cycles, not within them)
  ▼  z : [B, 100, d]                │
┌──────────────────────────────┐    │
│  FusedBlock  × 3             │    │
│                              │    │
│   ① DiagonalSSM              │    │   linear ⇒ parallel prefix scan
│      h_t = a_t·h_{t-1} + b_t │    │
│                              │    │
│   ② StateAttention           │    │   ◄── the contribution
│      q  = W_q · h_{t-1}      │    │   the state asks the question
│      kv = z_{1:t}            │    │   the input answers it
│      h ← h + Attn(·)         │    │   the answer becomes the state
│                              │    │
│   ③ y = C·h + D·z            │    │
│   ④ y ← y + FFN(LN(y))       │    │
└──────────────────────────────┘    │
  │                                 │
  ▼  mean ‖ log-var over cycles     │
  │                                 │
  ▼  concat ◄───────────────────────┘
Rate/shape head
  shape = 1 − cumsum(softplus(knots))     monotone over u ∈ [0, 1]
  life  = exp(life_head(feats))           one scalar, in cycles
  SOH(t) = shape(t / life)                resampled with grid_sample
```

### Why the state queries the input

An SSM compresses history into the state `h_t`. In standard attention, *tokens* query tokens. Here the **state** queries the raw input cycles. We compared four variants:

| Variant | Query | Keys / values | Result lands in |
| --- | --- | --- | --- |
| `none` | — | — | — (pure SSM) |
| `h` | h_t | states | state |
| **`hx`** (used throughout) | **h_{t-1}** | **input z** | **state** |
| `y` | h_t | input | readout only |

The query is `h_{t-1}`, the state *before* the current cycle is absorbed. Querying with `h_t` would let the question already contain its own answer. The recurrence stays linear in `h_{t-1}`, so the parallel prefix scan still applies: ⌈log₂ 100⌉ = 7 sequential steps instead of 100.

### Why rate and shape are separate

If *how fast* a cell degrades is coupled to *what shape* the decay has, then a test fold whose cells die three times sooner than anything in training must relearn the shape as well as the rate. Splitting them leaves only one scalar, the lifetime, to extrapolate.

The split is worth nothing on its own: alone, it was the **worst** of nine variants tried. Combined with the SOC view, it is the best by a wide margin:

| Zn-ion, hardest fold | MAPE | Change vs. baseline |
| --- | --- | --- |
| baseline | 7.398 | — |
| `+socview` | 7.526 | +0.128 |
| `+rateshape` | 7.793 | +0.396 |
| **`+socview+rateshape`** | **6.973** | **−0.425** |
| *sum of the two effects, if additive* | *7.890* | |

Estimating a lifetime needs SOC-localised degradation signatures. Without them, the split just adds an unconstrained scalar to guess.

---

## Results

Protocol: BatteryMFormer's published setup, with aging-condition-exclusive folds, every fold trained independently, and mean ± **std across folds**. Each fold is run with 3 seeds; seeds are averaged within a fold before the std is taken. Training runs for up to 25 epochs with early stopping.

| Domain | **StateAttention** (783k) | BatteryMFormer, reproduced (12.9M) | BatteryMFormer, published | Folds used |
| --- | --- | --- | --- | --- |
| Zn-ion | **4.391 ± 0.638** | 5.102 ± 0.413 | 4.970 ± 0.654 | 3 / 3 |
| Li-ion | 2.563 ± 0.082 | 2.604 ± 0.096 | **2.248 ± 0.037** | 3 / 3 |
| Na-ion | **0.921 ± 0.338** | 1.045 ± 0.350 | 1.002 ± 0.446 | 10 / 12 |
| CALB | 1.474 ± 1.297 | 1.725 ± 0.941 | 1.789 ± 0.282 | 4 / 4 |

### How much to believe

- **Zn-ion is the result.** StateAttention is 0.711 lower than the reproduced baseline and wins all three folds, with a fold std comparable to the baseline's.
- **Li-ion is a tie.** The gap is 0.042 against fold stds of about 0.09. Li-ion is also the best-measured domain (the smallest fold std anywhere), so this is a well-measured tie, not merely an uncertain one. Note that the reproduced baseline is 0.356 above its published number here, the only domain where the reproduction misses. The published BatteryMFormer remains better on Li-ion.
- **Na-ion is within noise.** The gap is 0.124 against a std of ±0.338, and StateAttention wins 7 of 10 folds.
- **CALB should not be claimed.** The margin comes from three flat-temperature folds. The one fold whose cells actually degrade (`cond396`, holding out 0 °C) goes to the baseline, 3.708 vs. 3.350. The fold std exceeds the mean. And BatteryLife's public CALB release stops at **99–100 cycles per cell** against a labelled EOL past 1,000, so the target is censored whatever the split.
- **Stability caveat.** Seed spread is larger than the baseline's on every domain (e.g. 0.317 vs. 0.108 on Na-ion, 0.936 vs. 0.847 on Zn-ion). The means favour StateAttention; the baseline is the steadier model.

### Baseline reproduction

| Domain | Reproduced | Published | Gap |
| --- | --- | --- | --- |
| Na-ion | 1.045 | 1.002 | 0.043 |
| CALB | 1.725 | 1.789 | 0.065 |
| Zn-ion | 5.102 | 4.970 | 0.132 |
| Li-ion | 2.604 | 2.248 | 0.356 |

Three of four domains land inside the published std, using a single configuration at 25 epochs against the original 30 Optuna trials at 300 epochs. This is what makes the reproduced column a fair comparison.

---

## Protocol

### Folds

An aging condition is the combination of positive electrode, negative electrode, electrolyte, package, nominal capacity, manufacturer, formation/charge/discharge protocol and temperature. Folds are split so that **test cells always come from conditions unseen in training**. Train, validation and test splits come from the official BatteryLife split file.

| Domain | Conditions | Protocol | Folds |
| --- | --- | --- | --- |
| Li-ion | 466 | 3 random condition-level splits, 6:2:2 train:val:test | 3 |
| Zn-ion | 95 | 3 random condition-level splits | 3 |
| Na-ion | 12 | leave-one-aging-condition-out | 12 (10 used) |
| CALB | 4 | leave-one-aging-condition-out | 4 |

- For CALB, the aging condition **is** the temperature. This is why `cond396` (holding out 0 °C, the only CALB cells that fade) behaves nothing like the other three folds.
- Two Na-ion folds are refused automatically. None of their test cells reach the EOL threshold, so the target is a flat line and MAPE would measure flat-line reproduction rather than forecasting. `build_splits` raises an error rather than silently producing a number.

### Training

| Setting | Value |
| --- | --- |
| Input | first 100 cycles, 300 SOC-aligned points, 4 channels |
| Augmentation | one sample per prefix length 1…100 (57 cells → ~5,700 samples) |
| Supervision | each sample scored only on cycles *after* its own prefix |
| Target | `(SOH − τ) / (1 − τ)`, predicted in that space and converted back for the metric |
| Loss | masked MSE over valid cycles |
| Optimiser | AdamW, constant LR 5e-4, weight decay 0 |
| Batch size | 128 |
| Epochs | up to 25, early stopping on validation MAPE (patience 10) |
| Model | d_model 96, 3 blocks, 4 heads, dropout 0.1 |
| Seeds | 2024, 2025, 2026 |

The prefix augmentation matters more than any architectural change we measured. Without it, a 12.9M-parameter model trains on 57 examples and scores 7.579 on Zn-ion instead of 5.102.

### Variance

Fold difficulty dominates run-to-run noise by about **19×**:

| Source of variation | Magnitude (MAPE) |
| --- | --- |
| Fold difficulty (same model, hard vs. easy fold) | 1.6 – 2.1 |
| Model-to-model gap within a fold | 0.09 – 0.19 |
| Seed (same model, same fold) | **0.02 – 0.15** |

So compare models **within** a fold, not by their overall mean ± std. `scripts/collect.py` reports both; the paired per-fold view is the one that separates models.

---

## Reproducing the results

The full sweep runs as a SLURM array.

```bash
git clone https://github.com/ShreyPatel1311/StateAttention.git && cd StateAttention
export WORK=$PWD

bash scripts/setup.sh                        # venv + BatteryMFormer's fold definitions
python scripts/prepare_data.py --root data/batterylife --domains zn-ion
python scripts/make_joblist.py --domains zn-ion --variants socview+rateshape \
       --seeds 2024,2025,2026 --out joblist.txt
sbatch --array=1-$(wc -l < joblist.txt) scripts/submit.sbatch
python scripts/collect.py --results results
```

- Set `--account` and `--qos` in `scripts/submit.sbatch` for your allocation.
- To add the BatteryMFormer column, run `bash scripts/fetch_baseline.sh`, then pass `--models hx,BatteryMFormer` to `make_joblist.py`.
- Each run writes its own JSON before returning, so an interrupted sweep keeps whatever finished.

**Data.** `prepare_data.py` downloads BatteryLife from [Zenodo record 19688272](https://zenodo.org/records/19688272) and builds three caches (~600 MB in total).

- The raw archives are about 31 GB and can be deleted afterwards with `--clean-raw`.
- Li-ion alone spans 13 source datasets and about 28 GB.

---

## Repository layout

```
src/stateattention/
  models/fused.py        FusedBlock, DiagonalSSM, StateAttention   ← the contribution
  models/traj.py         TrajForecaster: free / monotone / rate-shape heads
  models/socview.py      per-SOC-interval tokens
  models/embeds.py       intra-cycle encoders (flatten, patch, conv, delta, …)
  models/scan.py         parallel prefix scan
  data/trajectory.py     SOH extraction, artefact smoothing, EOL estimation
  data/trajdata.py       fold construction, censoring and coverage guards
  data/soc_curves.py     SOC-aligned 4-channel curve preparation
  bmf_eval.py            one training run, backend-agnostic
scripts/                 data prep, job list, SLURM array, result collection
tests/                   unit tests (pytest)
```

---

## Notice on third-party code and data

- This project does **not** redistribute any part of BatteryMFormer. Their repository declared no licence at the time of writing. Their model code, aging-condition embeddings and fold definitions are therefore **fetched at setup time** (`scripts/setup.sh`, `scripts/fetch_baseline.sh`) into gitignored directories. Check their terms before using their code.
- `src/stateattention/data/soc_curves.py` implements the SOC-aligned curve preparation described in Appendix B of the BatteryMFormer paper, following their reference implementation. It is included because the comparison is meaningless without identical inputs.
- BatteryLife data is distributed by its own authors under their terms; see the Zenodo record.

---

## Citation

If you use this code, please cite our work:

```bibtex
@misc{patel2026smallstate,
  title  = {Small State, Long Horizon: A Hybrid State Space and Transformer
            Architecture for Forecasting the Trajectories of Battery Degradation},
  author = {Patel, Shrey and Rao, Bhargav Srinivas},
  year   = {2026},
  note   = {University of Maryland, College Park},
  url    = {https://github.com/ShreyPatel1311/StateAttention}
}
```

This work builds on:

```bibtex
@inproceedings{tan2026batterymformer,
  title     = {BatteryMFormer: Multi-level Learning for Battery Degradation
               Trajectory Forecasting},
  author    = {Tan, Ruifeng and Dong, Jintao and Hong, Weixiang and Li, Jia
               and Huang, Jiaqiang and Zhang, Tong-Yi},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining},
  year      = {2026},
  doi       = {10.1145/3770855.3818948}
}
@inproceedings{tan2025batterylife,
  title     = {BatteryLife: A Comprehensive Dataset and Benchmark for Battery
               Life Prediction},
  author    = {Tan, Ruifeng and others},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining},
  year      = {2025}
}
```

## License

MIT. See [LICENSE](LICENSE).
