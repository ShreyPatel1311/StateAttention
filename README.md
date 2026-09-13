# StateAttention

A small SOH trajectory forecaster for lithium-, sodium- and zinc-ion cells, built
around one idea: **let the recurrent state query the input.**

At **783k parameters** it matches or beats
[BatteryMFormer](https://github.com/Ruifeng-Tan/BatteryMFormer) (12.9M) on all
four BatteryLife domains under BatteryMFormer's own evaluation protocol.

---

## The task

Given the first **S ≤ 100 cycles** of a cell's voltage / current / capacity
curves, predict its **state of health at every future cycle to end of life**:

```
ŷ[S+1 : t_eol] = f(G[1:S])
```

SOH is `Cap_i / (nominal_capacity × DoD)`, and end of life is the first cycle
where SOH crosses a domain threshold (0.80 for Li-ion and Zn-ion, 0.90 for
Na-ion and CALB). The metric is MAPE on raw SOH over the **forecast window
only** — cycles after the input, to EOL.

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
│   ① DiagonalSSM              │    │
│      h_t = a_t·h_{t-1} + b_t │    │   linear ⇒ parallel prefix scan
│                              │    │
│   ② StateAttention  ◄── the contribution                    │
│      q  = W_q · h_{t-1}      │    │   the state asks the question
│      kv = z_{1:t}            │    │   the input answers it
│      h ← h + Attn(·)         │    │   the answer becomes the state
│                              │    │
│   ③ y = C·h + D·z            │    │
│   ④ y ← y + FFN(LN(y))       │    │
└──────────────────────────────┘    │
  │                                 │
  ▼  mean ‖ log-var over cycles ────┘
  │
  ▼  concat
Rate/shape head
  shape = 1 − cumsum(softplus(knots))     monotone, over u ∈ [0,1]
  life  = exp(life_head(feats))           scalar, in cycles
  SOH(t) = shape(t / life)                resampled with grid_sample
```

### Why the state queries the input

An SSM compresses history into `h_t`. Attention normally lets *tokens* query
tokens. Here the **state** queries the raw input cycles:

| arm | query | keys/values | result lands in |
| --- | --- | --- | --- |
| `none` | — | — | — (pure SSM) |
| `h` | h_t | states | state |
| **`hx`** | **h_{t-1}** | **input z** | **state** |
| `y` | h_t | input | readout only |

`hx` is the arm used throughout. The query is `h_{t-1}`, the state *before* this
cycle is absorbed — querying with `h_t` would let the question already contain
its own answer. The recurrence stays linear in `h_{t-1}`, so the parallel prefix
scan still applies: ⌈log₂ 100⌉ = 7 steps instead of 100.

### Why rate and shape are separate

Coupling *how fast* a cell dies to *what shape* the decay has means a test fold
whose cells die three times sooner than anything in training must relearn the
shape as well as the rate. Splitting them leaves only a scalar to extrapolate.

The split is worth nothing on its own — alone it was the **worst** of nine
variants tried. Combined with the SOC view it is the best, by a wide margin:

| Zn-ion, hard fold | MAPE | vs baseline |
| --- | --- | --- |
| baseline | 7.398 | — |
| `+socview` | 7.526 | +0.128 |
| `+rateshape` | 7.793 | +0.396 |
| **`+socview+rateshape`** | **6.973** | **−0.425** |
| *additive prediction* | *7.890* | |

Estimating a lifetime needs SOC-localised degradation signatures; without them
the split just adds an unconstrained scalar to guess.

---

## Results

BatteryMFormer's published protocol: aging-condition-exclusive folds, every fold
trained independently, mean ± **std across folds**. 25 epochs, 3 seeds per fold,
seeds averaged within a fold before the std is taken.

| target | **StateAttention** (783k) | BatteryMFormer reproduced (12.9M) | BatteryMFormer published |
| --- | --- | --- | --- |
| **Zn-ion** | **4.391 ± 0.638** | 5.102 ± 0.413 | 4.970 ± 0.654 |
| **CALB** | **1.474 ± 1.297** | 1.725 ± 0.941 | 1.789 ± 0.282 |
| **Na-ion** | **0.921 ± 0.338** | 1.045 ± 0.350 | 1.002 ± 0.446 |
| **Li-ion** | **2.563 ± 0.082** | 2.604 ± 0.096 | 2.248 ± 0.037 |

Folds: Li-ion 3/3, Zn-ion 3/3, Na-ion 10/10, CALB 4/4.

### How much of this to believe

**Zn-ion is the result.** −0.711, winning all three folds, fold std comparable
to the baseline's.

**Li-ion is a tie** (−0.042 against fold stds ≈ 0.09). It is also the
best-measured domain here — the smallest fold std anywhere — so the tie is
well-measured rather than merely uncertain. Note the reproduced baseline is
0.356 above its published number on this domain, the only one where the
reproduction misses.

**Na-ion sits inside noise** (−0.124 against ±0.338, 7 of 10 folds).

**CALB should not be claimed.** The margin comes from three flat-temperature
folds; the one fold whose cells actually degrade (`cond396`, holding out 0 °C)
goes to the baseline, 3.708 vs 3.350. The fold std exceeds the mean. And
BatteryLife's public CALB release stops at **99–100 cycles per cell** against a
labelled EOL past 1000, so the target is a censored stub whatever the split.

**Stability caveat.** Seed spread is larger than the baseline's on every domain
(0.317 vs 0.108 on Na-ion, 0.936 vs 0.847 on Zn-ion). The means favour this
model; the baseline is the steadier of the two.

### Baseline reproduction

| target | reproduced | published | gap |
| --- | --- | --- | --- |
| Na-ion | 1.045 | 1.002 | 0.043 |
| CALB | 1.725 | 1.789 | 0.065 |
| Zn-ion | 5.102 | 4.970 | 0.132 |
| Li-ion | 2.604 | 2.248 | 0.356 |

Three of four land inside the published std, on a single configuration at 25
epochs against their 30 Optuna trials at 300 — which is what makes the
comparison column trustworthy.

---

## Protocol

### Folds

Aging condition = (positive electrode, negative electrode, electrolyte, package,
nominal capacity, manufacturer, formation / charge / discharge protocol,
temperature). BatteryMFormer splits so that **test cells come from conditions
unseen in training**:

| domain | conditions | protocol | folds |
| --- | --- | --- | --- |
| Li-ion | 466 | 3 random splits at a 6:2:2 condition ratio | 3 |
| Zn-ion | 95 | 3 random splits | 3 |
| Na-ion | 12 | leave-one-aging-condition-out | 12 |
| CALB | 4 | leave-one-aging-condition-out | 4 |

For CALB the aging condition **is** the temperature, which is why `cond396`
(holding out 0 °C, the only CALB cells that fade) behaves nothing like the other
three.

Two Na-ion folds are refused automatically: 100 % of their test cells never reach
the EOL threshold, so the target is a flat stub and a MAPE on it measures
flat-line reproduction rather than forecasting. `build_splits` raises rather than
silently producing a number.

### Training

| | |
| --- | --- |
| input | first 100 cycles, 300 SOC-aligned points, 4 channels |
| augmentation | one sample per prefix length 1…100 — 57 cells become ~5 700 samples |
| supervision | each sample scored only on cycles *after* its own prefix |
| target | `(SOH − τ) / (1 − τ)`, predicted in that space, converted back for the metric |
| loss | masked MSE over valid cycles |
| optimiser | AdamW, constant LR 5e-4, weight decay 0 |
| batch | 128 |
| epochs | 25, early stopping on validation MAPE, patience 10 |
| model | d_model 96, 3 blocks, 4 heads, dropout 0.1 |

The prefix augmentation matters more than any architectural change measured
here: without it a 12.9M-parameter model trains on 57 examples and scores 7.579
on Zn-ion instead of 5.102.

### Variance

Fold difficulty dominates run-to-run noise by **19×**:

| effect | magnitude |
| --- | --- |
| fold difficulty (same model, hard vs easy fold) | 1.6 – 2.1 |
| model-to-model gap within a fold | 0.09 – 0.19 |
| seed (same model, same fold) | **0.02 – 0.15** |

So compare models **within** a fold, not by their means ± std. `scripts/collect.py`
reports both, and the paired per-fold view is the one that separates models.

---

## Reproducing

```bash
git clone https://github.com/<you>/StateAttention && cd StateAttention
export WORK=$PWD

bash scripts/setup.sh                       # venv + BatteryMFormer's fold definitions
python scripts/prepare_data.py --root data/batterylife --domains zn-ion
python scripts/make_joblist.py --domains zn-ion --variants socview+rateshape \
       --seeds 2024,2025,2026 --out joblist.txt
sbatch --array=1-$(wc -l < joblist.txt) scripts/submit.sbatch
python scripts/collect.py --results results
```

Set `--account` / `--qos` in `scripts/submit.sbatch` for your allocation.

To add the baseline column, `bash scripts/fetch_baseline.sh` then pass
`--models hx,BatteryMFormer`.

**Data.** `prepare_data.py` pulls BatteryLife from
[Zenodo record 19688272](https://zenodo.org/records/19688272) and builds three
caches (~600 MB total). The raw archives are ~31 GB and can be deleted after
with `--clean-raw`. Li-ion alone spans 13 source datasets and ~28 GB.

Each run writes its own JSON before returning, so an interrupted sweep keeps
whatever finished.

---

## Layout

```
src/stateattention/
  models/fused.py        FusedBlock, DiagonalSSM, StateAttention  ← the contribution
  models/traj.py         TrajForecaster: free / monotone / rate-shape heads
  models/socview.py      per-SOC-interval tokens
  models/embeds.py       intra-cycle encoders (flatten, patch, conv, delta, …)
  models/scan.py         parallel prefix scan
  data/trajectory.py     SOH extraction, artefact smoothing, EOL estimation
  data/trajdata.py       fold construction, censoring and coverage guards
  data/soc_curves.py     SOC-aligned 4-channel curve preparation
  bmf_eval.py            one training run, backend-agnostic
scripts/                 data prep, job list, SLURM array, collection
```

---

## NOTICE

This project does **not** redistribute any part of BatteryMFormer. Their
repository declared no licence at the time of writing, so their model code,
aging-condition embeddings and fold definitions are **fetched at setup time**
(`scripts/setup.sh`, `scripts/fetch_baseline.sh`) into gitignored directories.
Check their terms before using their code.

`src/stateattention/data/soc_curves.py` implements the SOC-aligned curve
preparation described in Appendix B of their paper, following their reference
implementation. It is included because the comparison is meaningless without
identical inputs.

BatteryLife data is distributed by its own authors under their terms; see the
Zenodo record.

## Citing the work this builds on

```bibtex
@inproceedings{tan2026batterymformer,
  title     = {BatteryMFormer: Multi-level Learning for Battery Degradation
               Trajectory Forecasting},
  author    = {Tan, Ruifeng and others},
  booktitle = {KDD},
  year      = {2026}
}
@inproceedings{tan2025batterylife,
  title     = {BatteryLife: A Comprehensive Dataset and Benchmark for Battery
               Life Prediction},
  author    = {Tan, Ruifeng and others},
  booktitle = {KDD},
  year      = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE).
