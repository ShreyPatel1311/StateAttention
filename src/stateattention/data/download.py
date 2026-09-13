"""Fetch the two benchmarks.

BatteryLife comes straight off the Hub. NASA PCoE does not: the official NASA
download has been broken for a while, so we pull from the Kaggle mirror of the
same .mat files and fall back to printed instructions if the Kaggle CLI is not
configured.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

BL_REPO = "Hongwxx/BatteryLife_processed"
BL_FARASIS_REPO = "Battery-Life/BatteryLife_Farasis"

# The Hugging Face copy is `gated: auto` -- it 403s until the account holder
# clicks through the terms on the dataset page, which cannot be automated. The
# Zenodo copy of the same processed release is ungated and needs no token, so it
# is the default source here. Per-subset zips, ~22 GB for the paper's subsets.
ZENODO_RECORD = "19688272"
ZENODO_FILE_URL = "https://zenodo.org/api/records/{rec}/files/{name}/content"

# The 13 public Li-ion sources the paper groups into the "Li-ion" domain, plus
# the three single-source domains. SDU and Stanford_2 landed after the paper, so
# they are excluded by default -- including them makes numbers non-comparable.
DOMAIN_SUBSETS: dict[str, tuple[str, ...]] = {
    "li-ion": (
        "CALCE", "HNEI", "HUST", "ISU_ILCC", "MATR", "MICH", "MICH_EXP",
        "RWTH", "SNL", "Stanford", "Tongji", "UL_PUR", "XJTU",
    ),
    "zn-ion": ("ZN-coin",),
    "na-ion": ("NA-ion",),
    "calb": ("CALB",),
}

# Post-paper additions. Opt in explicitly if you want to report on them.
EXTRA_SUBSETS = ("SDU", "Stanford_2")

NASA_KAGGLE_SLUG = "ckskaggle/li-ion-battery-dataset-from-nasa-pcoe"


def download_batterylife(
    root: Path, domains: tuple[str, ...] = ("li-ion",), include_extras: bool = False
) -> Path:
    """Download only the subsets the requested domains need.

    The full processed release is tens of GB; pulling one domain keeps a laptop
    checkout small and the disk footprint down.
    """
    root = Path(root) / "batterylife"
    root.mkdir(parents=True, exist_ok=True)

    subsets: list[str] = []
    for d in domains:
        if d not in DOMAIN_SUBSETS:
            raise ValueError(f"unknown domain {d!r}; pick from {sorted(DOMAIN_SUBSETS)}")
        subsets.extend(DOMAIN_SUBSETS[d])
    if include_extras:
        subsets.extend(EXTRA_SUBSETS)

    # "Life labels" holds the cycle-life target for every cell -- always needed.
    patterns = ["Life labels/*", "README.md"] + [f"{s}/*" for s in subsets]

    print(f"[batterylife] downloading {len(subsets)} subsets -> {root}")
    snapshot_download(
        repo_id=BL_REPO,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=patterns,
        max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "8")),
    )
    return root


def download_subset(root: Path, subset: str) -> Path:
    """Fetch one BatteryLife subset (plus the shared label files).

    Used by the streaming cache job: download a subset, reduce it to npz, delete
    the raw files, move on. Peak disk is then the largest single subset (~30 GB
    for ISU_ILCC) rather than the whole 64 GB release.
    """
    root = Path(root) / "batterylife"
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=BL_REPO,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=["Life labels/*", f"{subset}/*"],
        max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "8")),
    )
    return root / subset


def _zenodo_fetch_extract(root: Path, member: str) -> None:
    """Stream one Zenodo zip into `root`, extract it, and drop the archive.

    Extraction is normalised: some archives wrap their contents in a top-level
    directory and some do not, so we unpack to a staging dir and move the
    payload into place either way.
    """
    import shutil as _shutil
    import tempfile
    import urllib.request
    from urllib.parse import quote

    from tqdm import tqdm

    stem = member.removesuffix(".zip")
    target = root / stem
    if target.exists() and any(target.iterdir()):
        print(f"[zenodo] {stem}: already present")
        return

    url = ZENODO_FILE_URL.format(rec=ZENODO_RECORD, name=quote(member))
    with tempfile.TemporaryDirectory(dir=str(root)) as tmp:
        tmp = Path(tmp)
        zpath = tmp / member
        # stdlib rather than requests: one less dependency to get out of sync
        # between this repo and a long-lived venv on a rented box.
        req = urllib.request.Request(url, headers={"User-Agent": "mambaformer/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - fixed https host
            total = int(r.headers.get("content-length") or 0)
            with (
                open(zpath, "wb") as fh,
                tqdm(
                    total=total, unit="B", unit_scale=True, desc=f"  {stem}",
                    file=sys.stdout, mininterval=2.0, dynamic_ncols=True,
                ) as bar,
            ):
                while chunk := r.read(1 << 22):
                    fh.write(chunk)
                    bar.update(len(chunk))

        stage = tmp / "stage"
        with zipfile.ZipFile(zpath) as z:
            z.extractall(stage)
        zpath.unlink()

        inner = stage / stem
        src = inner if inner.is_dir() else stage
        target.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            dest = target / item.name
            if dest.exists():
                continue
            _shutil.move(str(item), str(dest))

    n = len(list(target.rglob("*")))
    print(f"[zenodo] {stem}: {n} files")


def download_subset_zenodo(root: Path, subset: str) -> Path:
    """Fetch one subset (and the shared life labels) from the ungated Zenodo copy."""
    root = Path(root) / "batterylife"
    root.mkdir(parents=True, exist_ok=True)
    _zenodo_fetch_extract(root, "Life labels.zip")
    _zenodo_fetch_extract(root, f"{subset}.zip")
    return root / subset


def download_nasa(root: Path) -> Path:
    """Fetch the NASA PCoE Li-ion aging .mat files via the Kaggle mirror."""
    root = Path(root) / "nasa"
    root.mkdir(parents=True, exist_ok=True)

    if list(root.rglob("*.mat")):
        print(f"[nasa] already present at {root}")
        return root

    if shutil.which("kaggle") is None:
        _print_nasa_manual(root)
        return root

    print(f"[nasa] downloading {NASA_KAGGLE_SLUG} -> {root}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", NASA_KAGGLE_SLUG, "-p", str(root)],
            check=True,
        )
    except subprocess.CalledProcessError:
        _print_nasa_manual(root)
        return root

    for zf in root.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(root)
        zf.unlink()

    n = len(list(root.rglob("*.mat")))
    print(f"[nasa] extracted {n} .mat files")
    return root


def _print_nasa_manual(root: Path) -> None:
    print(
        f"""
[nasa] automatic download unavailable. The Kaggle CLI is either missing or
       unauthenticated. Either:

  1. pip install kaggle, put your token at ~/.kaggle/kaggle.json, re-run; or
  2. download by hand and unzip the .mat files into {root}:
       - https://www.kaggle.com/datasets/{NASA_KAGGLE_SLUG}
       - https://ieee-dataport.org/documents/nasa-lithium-ion-battery-dataset

   NASA's own PCoE page currently has direct download disabled, which is why we
   mirror. The .mat contents are identical.
"""
    )


def push_to_hub(local_dir: Path, repo_id: str, repo_type: str = "dataset", path_in_repo: str = "") -> None:
    """Mirror a processed/cached directory to your own Hub repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
    )
    print(f"[hub] pushed {local_dir} -> {repo_id}/{path_in_repo}")
