"""Training loop.

Targets are trained in log space and exponentiated before metrics, so the MSE
objective is aligned with MAPE (see models/classical.py for the same argument).
Model selection is on validation MAPE, never on test.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .metrics import compute_metrics
from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loss(name: str):
    return {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.HuberLoss(delta=0.1),
        "log_mse": nn.MSELoss(),
    }[name.lower()]


def collate(batch):
    return {
        "x": torch.from_numpy(np.stack([b["x"] for b in batch])),
        "mask": torch.from_numpy(np.stack([b["mask"] for b in batch])),
        "y": torch.tensor([float(b["y"]) for b in batch], dtype=torch.float32),
        "n_cycles": torch.tensor([int(b["n_cycles"]) for b in batch]),
    }


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred) in *cycle* units, not log units."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        out = model(x, m)
        ps.append(torch.exp(out.float()).cpu().numpy())
        ys.append(batch["y"].numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one(
    cfg: Config,
    train_ds,
    val_ds,
    test_ds,
    seed: int,
    out_dir: Path,
    verbose: bool = True,
) -> dict:
    """Train a single seed, return test metrics at the best-val checkpoint."""
    set_seed(seed)
    device = pick_device(cfg.train.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    train_ld = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_ld = DataLoader(val_ds, shuffle=False, **common)
    test_ld = DataLoader(test_ds, shuffle=False, **common)

    model = build_model(cfg.model, cfg.data.max_cycles, cfg.data.curve_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    crit = make_loss(cfg.train.loss)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Start the head at the mean training label so early stopping is not
    # measuring how long the bias takes to climb off zero.
    model.init_head_bias(float(np.log(np.clip(np.asarray(train_ds.labels), 1.0, None)).mean()))

    best_mape, best_epoch, bad = float("inf"), -1, 0
    ckpt = out_dir / f"best_seed{seed}.pt"
    history = []

    for epoch in range(cfg.train.epochs):
        model.train()
        t0, total, nb = time.time(), 0.0, 0
        for batch in train_ld:
            x = batch["x"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            y = torch.log(batch["y"].clamp(min=1.0)).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x, m), y)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            nb += 1

        yv, pv = predict(model, val_ld, device)
        vm = compute_metrics(yv, pv)
        history.append({"epoch": epoch, "train_loss": total / max(nb, 1), **vm})

        if verbose:
            print(
                f"  epoch {epoch:3d} | loss {total / max(nb, 1):.4f} "
                f"| val MAPE {vm['mape']:.4f} | val 15%-Acc {vm['acc@15%']:.3f} "
                f"| {time.time() - t0:.1f}s"
            )

        if vm["mape"] < best_mape - 1e-5:
            best_mape, best_epoch, bad = vm["mape"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": vm}, ckpt)
        else:
            bad += 1
            if bad >= cfg.train.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best {best_epoch})")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    yt, pt = predict(model, test_ld, device)
    test_m = compute_metrics(yt, pt)

    result = {
        "seed": seed,
        "model": cfg.model.name,
        "params": n_params,
        "best_epoch": best_epoch,
        "val_mape": best_mape,
        "test": test_m,
    }
    (out_dir / f"result_seed{seed}.json").write_text(
        json.dumps({**result, "history": history}, indent=2)
    )
    np.savez(out_dir / f"preds_seed{seed}.npz", y=yt, pred=pt)
    return result
