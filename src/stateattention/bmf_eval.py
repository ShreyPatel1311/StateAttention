"""One training run: build the splits, train a model, score it on the test fold.

Everything environment-specific -- where the caches live, where the baseline's
code sits, where results are written -- arrives as an argument, so the same
function drives a laptop, a single GPU box and a SLURM array without forking.

Each run writes its own result JSON before returning, so an interrupted sweep
keeps whatever finished.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BMF_DEFAULTS = dict(
    input_mode="current_voltage", eol_threshold=0.8, truncate_start_cycle=100,
    kernel_size=10, cnn_channels=16, num_routed_experts=8, num_shared_experts=2,
    num_query=6, seq_len=100, num_slots=128, top_k=2, temperature=1.0,
    pred_len=5000, d_model=128, d_llm=1024, n_heads=8, e_layers=3, e_layers2=3,
    d_layers=1, d_ff=256, d_ffs=256, k_dim=4, mem_dim=128, dropout=0.1,
    activation="gelu", lradj="constant", warmup_epochs=5, weight_decay=0.0,
    early_cycle_threshold=100, charge_discharge_length=300, num_segments=50,
    factor=1, task_name="soh_forecast", moving_avg=25, label_len=48,
    patch_len=16, stride=8, embedding_dropout=0.1, enc_in=3, use_norm=1,
    head_dropout=0.0, revin=0, affine=0, subtract_last=0, deformable=1,
    horizon=5000, context=100, patch_ks=300, patch_sd=300,
    lambda_recovery=100.0, lambda_mem=10.0, lambda_life_loss=0.0,
)

ZN_HP = {
    "bmf:ZNcoin_split_seed2021": dict(lr=1e-4, dropout=0.25, d_model=128, d_ff=128,
                                      d_ffs=256, k_dim=512, mem_dim=128, e_layers=2,
                                      d_layers=6, num_query=4, num_slots=64, kernel=10),
    "bmf:ZNcoin_split_seed42":   dict(lr=5e-5, dropout=0.5, d_model=64, d_ff=128,
                                      d_ffs=64, k_dim=512, mem_dim=128, e_layers=4,
                                      d_layers=6, num_query=50, num_slots=64, kernel=10),
    "bmf:ZNcoin_split_seed2024": dict(lr=3e-5, dropout=0.05, d_model=64, d_ff=32,
                                      d_ffs=64, k_dim=512, mem_dim=128, e_layers=4,
                                      d_layers=8, num_query=20, num_slots=64, kernel=10),
}
EOL_THRESHOLD = {"zn-ion": 0.8, "li-ion": 0.8, "na-ion": 0.9, "calb": 0.9}
# Used for domains where we have not transcribed their per-fold table; these are
# their published constants plus the argparse defaults for the tuned fields.
DEFAULT_HP = dict(lr=1e-4, dropout=0.1, d_model=128, d_ff=256, d_ffs=256,
                  k_dim=512, mem_dim=128, e_layers=2, d_layers=6,
                  num_query=8, num_slots=64, kernel=10)
# "none" (pure SSM control) dropped at the user's request -- it served as the
# no-attention baseline and is no longer of interest.
OURS = {"hx": ("fused", "hx"), "stateattention": ("stateattention", "none")}
# Trivial controls. They go through the same dataset, mask and metric as the
# trained models, so their score says how much of a MAPE is actually earned.
TRIVIAL = ("constant", "trainmean")

DEFAULT_HP = dict(lr=1e-4, dropout=0.1, d_model=128, d_ff=256, d_ffs=256,
                  k_dim=512, mem_dim=128, e_layers=2, d_layers=6,
                  num_query=8, num_slots=64, kernel=10)
# "none" (pure SSM control) dropped at the user's request -- it served as the
# no-attention baseline and is no longer of interest.
OURS = {"hx": ("fused", "hx"), "stateattention": ("stateattention", "none")}
# Trivial controls. They go through the same dataset, mask and metric as the
# trained models, so their score says how much of a MAPE is actually earned.
TRIVIAL = ("constant", "trainmean")

EOL_THRESHOLD = {"zn-ion": 0.8, "li-ion": 0.8, "na-ion": 0.9, "calb": 0.9}
# Used for domains where we have not transcribed their per-fold table; these are
# their published constants plus the argparse defaults for the tuned fields.
DEFAULT_HP = dict(lr=1e-4, dropout=0.1, d_model=128, d_ff=256, d_ffs=256,
                  k_dim=512, mem_dim=128, e_layers=2, d_layers=6,
                  num_query=8, num_slots=64, kernel=10)
# "none" (pure SSM control) dropped at the user's request -- it served as the
# no-attention baseline and is no longer of interest.
OURS = {"hx": ("fused", "hx"), "stateattention": ("stateattention", "none")}
# Trivial controls. They go through the same dataset, mask and metric as the
# trained models, so their score says how much of a MAPE is actually earned.
TRIVIAL = ("constant", "trainmean")

OURS = {"hx": ("fused", "hx"), "stateattention": ("stateattention", "none")}
# Trivial controls. They go through the same dataset, mask and metric as the
# trained models, so their score says how much of a MAPE is actually earned.
TRIVIAL = ("constant", "trainmean")

TRIVIAL = ("constant", "trainmean")


class _Cfg:
    """Their config object: full defaults, then explicit overrides."""

    def __init__(self, **kw):
        self.__dict__.update(BMF_DEFAULTS)
        self.__dict__.update(kw)


def _result_tag(label, group, seed, cfgd):
    """Filename that is unique per run.

    The first version keyed only on (label, group, aug), so every seed and every
    ablation variant overwrote the same file -- 30 runs left 2 JSONs on the
    volume. Console tables were unaffected (they aggregate returned dicts), but
    the persisted record was lossy, which also meant an interrupted grid lost far
    more than it appeared to.
    """
    bits = [label, group.split(":")[-1], f"s{seed}",
            "aug" if cfgd.get("prefix_aug") else "noaug"]
    if cfgd.get("prefix_stride", 1) != 1:
        bits.append(f"ps{cfgd['prefix_stride']}")
    if cfgd.get("embed_type", "flatten") != "flatten":
        bits.append(f"e-{cfgd['embed_type']}")
    if cfgd.get("ssm_core", "diagonal") != "diagonal":
        bits.append(f"c-{cfgd['ssm_core']}")
    if cfgd.get("soc_view"):
        bits.append("socview")
    if cfgd.get("head_type", "free") != "free":
        bits.append(f"h-{cfgd['head_type']}")
    return "__".join(bits)


def run_one(domain, label, group, seed, cfgd, data_root, bmf_root,
            emb_path, bmf_repo, results_dir=None, on_commit=None):
    from stateattention.data.trajdata import DOMAIN_TAU, build_splits

    t0 = time.time()
    root = Path(data_root)
    tau = DOMAIN_TAU.get(domain, 0.80)
    thr = EOL_THRESHOLD[domain]
    t_max = cfgd["t_max"]
    S = 100  # input window; their early_cycle_threshold

    try:
        (tr, va, te), _, pool_n = build_splits(
            root, group=group, t_max=t_max, tau=tau,
            max_censoring=cfgd.get("max_censoring", 0.5))
    except ValueError as e:
        # A fold can be refused for censoring (its test cells never reach EOL,
        # so the target is a flat stub) or for coverage. Skipping it must not
        # take down the other folds, so report it instead of raising.
        print(f"[{label}/{group.split('_')[-1]}] SKIPPED: {e}", flush=True)
        return {"domain": domain, "label": label, "group": group, "seed": seed,
                "skipped": str(e).split(".")[0], "mape_pct": float("nan"),
                "mae_e2": float("nan")}

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    is_theirs = label == "BatteryMFormer"
    is_trivial = label in TRIVIAL

    # ---- evaluation mask: their convention, cycles after the input window ----
    def fwd_mask(m):
        out = m.copy()
        out[..., :S] = 0.0
        return out

    aug = bool(cfgd.get("prefix_aug", False))
    their_inputs = is_theirs or bool(cfgd.get("their_inputs", False))

    if their_inputs:
        bmf_dir = Path(bmf_root)
        emb = pickle.load(open(emb_path, "rb"))

        stride = max(1, int(cfgd.get("prefix_stride", 1)))

        def _prefixes(n_valid):
            """Their loader emits one sample per prefix length, 1..100.

            `prefix_stride` subsamples that ladder. Li-ion has ~600 training
            cells, so the full ladder is ~60k samples per epoch against Zn-ion's
            5.7k; stride 10 keeps the two domains at a comparable number of
            optimisation steps. A documented deviation from their loader.
            """
            hi = min(n_valid, S)
            if not aug:
                return [hi]
            ks = list(range(1, hi + 1, stride))
            if ks[-1] != hi:
                ks.append(hi)          # always include the full 100-cycle window
            return ks

        class TheirDS(Dataset):
            """Their inputs, their prefix augmentation, their normalisation.

            One sample per (cell, prefix length). The trajectory mask starts at
            the prefix length, so a sample is supervised only on the cycles it
            has not seen -- which is what makes the short prefixes informative
            rather than redundant.
            """

            def __init__(self, ds):
                self.rows = []
                for it in ds.items:
                    f = bmf_dir / f"{it['stem']}.npz"
                    if not f.exists() or f"{it['stem']}.pkl" not in emb:
                        continue
                    n_valid = int(np.load(f)["n_cycles"])
                    for k in _prefixes(n_valid):
                        self.rows.append((it, k))

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, i):
                it, k = self.rows[i]
                z = np.load(bmf_dir / f"{it['stem']}.npz")
                cur = np.asarray(z["curves"], dtype=np.float32)   # [100, 4, 300]
                curves = np.zeros((S, 4, 300), dtype=np.float32)
                curves[: min(len(cur), S)] = cur[:S]
                curves[k:] = 0.0                                   # unseen cycles
                amask = np.zeros(S, dtype=np.float32)
                amask[:k] = 1.0

                y = ((it["y"] - thr) / (1.0 - thr)).astype(np.float32)
                m = it["m"].copy()
                m[:k] = 0.0                                        # forecast only
                soh_in = np.zeros((S, 1), dtype=np.float32)
                soh_in[:k, 0] = y[:k]
                ce = np.zeros((S, 1), dtype=np.float32); ce[:k, 0] = z["ce"][:k]
                ee = np.zeros((S, 1), dtype=np.float32); ee[:k, 0] = z["ee"][:k]
                return {
                    "curves": curves[:, :3, :], "soc": curves[:, 3, :],
                    "amask": amask, "y": y, "ymask": m.astype(np.float32),
                    "yraw": it["y"].astype(np.float32),
                    "x4": curves.transpose(0, 2, 1),               # [S, 300, 4] for ours
                    "soh_in": soh_in, "ce": ce, "ee": ee,
                    "emb": emb[f"{it['stem']}.pkl"].reshape(1, -1).astype(np.float32),
                    "life": np.float32(it["eol"]),
                }

    if is_theirs:

        DS = TheirDS

        def collate(b):
            return {k: torch.from_numpy(np.stack([x[k] for x in b])) for k in b[0]}

        hp = ZN_HP.get(group, DEFAULT_HP)
        # their code uses absolute imports (layers.*, utils.*), so their
        # repo root has to be on sys.path as-is rather than as a package
        sys.path.insert(0, str(bmf_repo))
        from models.BatteryMFormer import Model
        cfg = _Cfg(seq_len=S, pred_len=t_max, early_cycle_threshold=S,
                   d_model=hp["d_model"], d_ff=hp["d_ff"], d_ffs=hp["d_ffs"],
                   e_layers=hp["e_layers"], d_layers=hp["d_layers"],
                   dropout=hp["dropout"], k_dim=hp["k_dim"], mem_dim=hp["mem_dim"],
                   num_query=hp["num_query"], num_slots=hp["num_slots"],
                   kernel_size=hp["kernel"], stride=hp["kernel"])
        model = Model(cfg).to(dev)
        lr = hp["lr"]
    elif is_trivial:
        DS = TheirDS

        def collate(b):
            return {k: torch.from_numpy(np.stack([x[k] for x in b])) for k in b[0]}
    else:
        from stateattention.config import ModelConfig
        from stateattention.models.traj import TrajForecaster

        if their_inputs:
            DS = TheirDS          # same inputs, augmentation and normalisation
            in_ch = 4             # V, I, Q, SOC
        else:
            class DS(Dataset):
                def __init__(self, ds):
                    self.ds = ds

                def __len__(self):
                    return len(self.ds)

                def __getitem__(self, i):
                    d = self.ds[i]
                    return {"x": d["x"], "mask": d["mask"], "y": d["y"],
                            "ymask": fwd_mask(d["ymask"]), "yraw": d["y"]}
            in_ch = 3

        def collate(b):
            return {k: torch.from_numpy(np.stack([x[k] for x in b])) for k in b[0]}

        bb, arm = OURS[label]
        d_m = cfgd["d_model"]
        mc = ModelConfig(name="fused", d_model=d_m, d_ff=2 * d_m, n_layers=3,
                         n_heads=4, attn_on=arm, dropout=0.1)
        # intra-cycle encoder is an ablation axis; "flatten" is the default and
        # what every earlier result used.
        mc.embed_type = cfgd.get("embed_type", "flatten")
        mc.embed_patch = cfgd.get("embed_patch", 25)
        mc.ssm_core = cfgd.get("ssm_core", "diagonal")
        mc.soc_view = bool(cfgd.get("soc_view", False))
        mc.soc_patch = cfgd.get("soc_patch", 25)
        mc.head_type = cfgd.get("head_type", "free")
        model = TrajForecaster(mc, S, 300, in_ch, t_max, cfgd["stride"], bb).to(dev)
        lr = cfgd["lr"]

    dsets = {s: DS(d) for s, d in (("train", tr), ("val", va), ("test", te))}
    ld = {s: DataLoader(d, batch_size=cfgd["batch_size"], shuffle=(s == "train"),
                        collate_fn=collate, num_workers=2,
                        drop_last=(s == "train" and len(d) > cfgd["batch_size"]))
          for s, d in dsets.items()}

    if is_trivial:
        # No model and no training: score a predictor that ignores the input,
        # through the same dataset, mask and metric as everything else.
        def gather(loader):
            Y, M = [], []
            for b in loader:
                Y.append(b["yraw"].numpy())
                M.append(b["ymask"].numpy())
            return np.concatenate(Y), np.concatenate(M)

        ytr, mtr = gather(ld["train"])
        yte, mte = gather(ld["test"])
        if label == "constant":
            pred = np.full_like(yte, 1.0 - np.log1p(np.exp(-4.0)))
        else:  # trainmean: the mean training trajectory, per cycle
            pred = np.broadcast_to(
                ((ytr * mtr).sum(0) / np.maximum(mtr.sum(0), 1.0))[None, :], yte.shape)
        keep = (mte > 0) & (np.abs(yte) > 1e-6)
        pc = np.clip(pred[keep], 0.0, 1.5)
        t = {"mape_pct": float(np.mean(np.abs(pc - yte[keep]) / np.abs(yte[keep])) * 100),
             "mae_e2": float(np.mean(np.abs(pc - yte[keep])) * 100)}
        print(f"[{label}/{group.split('_')[-1]}] TRIVIAL test MAPE {t['mape_pct']:.3f}",
              flush=True)
        res = {"domain": domain, "label": label, "group": group, "seed": seed,
           "embed": cfgd.get("embed_type", "flatten"),
           "ssm_core": cfgd.get("ssm_core", "diagonal"),
           "soc_view": bool(cfgd.get("soc_view", False)),
           "head_type": cfgd.get("head_type", "free"),
               "params": 0, "best_epoch": -1, "epochs_run": 0,
               "epochs_cap": cfgd["epochs"], "val_mape": float("nan"),
               "n_train": len(dsets["train"]), "n_test": len(dsets["test"]),
               "pool": pool_n, "seconds": round(time.time() - t0, 1),
               "protocol": {k: cfgd[k] for k in ("prefix_aug", "their_inputs",
                                                 "epochs", "patience", "batch_size")},
               **t}
        import json
        if results_dir is None:
            return res
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        tg = _result_tag(label, group, seed, cfgd)
        (out / f"{tg}.json").write_text(json.dumps(res, indent=2))
        if on_commit:
            on_commit()
        return res

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def masked_mse(pred, tgt, m):
        return (((pred * m) - (tgt * m)) ** 2).sum() / m.sum().clamp(min=1.0)

    def step(batch, train: bool):
        b = {k: v.to(dev) for k, v in batch.items()}
        if is_theirs:
            ret = model(
                cycle_curve_data=b["curves"], curve_attn_mask=b["amask"],
                soh_trajectory=b["y"], trajectory_mask=b["ymask"],
                aging_condition_embedding=b["emb"], soc_input=b["soc"],
                soh_input=b["soh_in"],
                cycle_level_features=torch.cat([b["ce"], b["ee"]], dim=-1),
                life_labels=b["life"])
            # their forward returns the auxiliary losses only in training mode
            if isinstance(ret, tuple):
                out, rec, _life, align, _ = ret
                loss = (masked_mse(out, b["y"], b["ymask"])
                        + 100.0 * rec + 10.0 * align)   # their lambdas
            else:
                out = ret
                loss = masked_mse(out, b["y"], b["ymask"])
            pred_raw = out * (1.0 - thr) + thr          # back to original SOH
        elif their_inputs:
            # our model, their inputs/targets: predict in their normalised
            # space and convert back only for the metric
            out = model(b["x4"], b["amask"])
            loss = masked_mse(out, b["y"], b["ymask"])
            pred_raw = out * (1.0 - thr) + thr
        else:
            out = model(b["x"], b["mask"])
            loss = masked_mse(out, b["yraw"], b["ymask"])
            pred_raw = out
        return loss, pred_raw, b

    def evaluate(loader):
        model.eval()
        P, Y, M = [], [], []
        with torch.no_grad():
            for batch in loader:
                _, pred, b = step(batch, False)
                P.append(pred.float().cpu().numpy())
                Y.append(b["yraw"].cpu().numpy())
                M.append(b["ymask"].cpu().numpy())
        p, y, m = (np.concatenate(a) for a in (P, Y, M))
        keep = (m > 0) & (np.abs(y) > 1e-6)
        if not keep.any():
            return {"mape_pct": float("nan"), "mae_e2": float("nan")}
        pc = np.clip(p[keep], 0.0, 1.5)
        return {"mape_pct": float(np.mean(np.abs(pc - y[keep]) / np.abs(y[keep])) * 100),
                "mae_e2": float(np.mean(np.abs(pc - y[keep])) * 100)}

    tag = f"{label}/{group.split('_')[-1]}"
    print(f"[{tag}] start: {len(dsets['train'])} train / {len(dsets['val'])} val / "
          f"{len(dsets['test'])} test samples, {n_params:,} params, "
          f"{len(ld['train'])} batches/epoch", flush=True)

    best, best_state, best_ep, bad = 1e18, None, -1, 0
    best_test = float("nan")
    for ep in range(cfgd["epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for batch in ld["train"]:
            loss, _, _ = step(batch, True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        v = evaluate(ld["val"])["mape_pct"]
        improved = np.isfinite(v) and v < best - 1e-6
        if improved:
            best, best_ep, bad = v, ep, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            # Test score of the checkpoint that would be selected, so live
            # monitoring shows the number that will actually be reported.
            # Selection stays on validation -- this is never read back.
            best_test = evaluate(ld["test"])["mape_pct"]
        else:
            bad += 1
        # a silent hour-long run is unmonitorable, so report every epoch
        note = "  *" if improved else f"  (patience {bad}/{cfgd['patience']})"
        print(f"[{tag}] ep {ep+1}/{cfgd['epochs']}  loss {tot / max(nb, 1):.5f}  "
              f"val {v:.3f}  best val {best:.3f}@{best_ep + 1}  "
              f"TEST {best_test:.3f}{note}", flush=True)
        if bad >= cfgd["patience"]:
            print(f"[{tag}] early stop at epoch {ep+1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    t = evaluate(ld["test"])
    res = {"domain": domain, "label": label, "group": group, "seed": seed,
           "params": n_params, "best_epoch": best_ep, "epochs_run": ep + 1,
           "epochs_cap": cfgd["epochs"], "val_mape": best,
           "n_train": len(dsets["train"]), "n_test": len(dsets["test"]),
           "pool": pool_n, "seconds": round(time.time() - t0, 1),
           "protocol": {k: cfgd[k] for k in ("prefix_aug", "their_inputs",
                                             "epochs", "patience", "batch_size")},
           "embed_type": cfgd.get("embed_type", "flatten"),
           **t}

    # Persist to the Volume before returning. starmap results live only in the
    # local driver, so a dropped session -- or a killed run -- threw away every
    # finished container. A run that completes is now recoverable regardless.
    import json

    if results_dir is None:
        return res
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    tagname = _result_tag(label, group, seed, cfgd)
    (out / f"{tagname}.json").write_text(json.dumps(res, indent=2))
    if on_commit:
        on_commit()
    print(f"[{tag}] DONE  test MAPE {t['mape_pct']:.3f}  -> results/{domain}/{tagname}.json",
          flush=True)
    return res


