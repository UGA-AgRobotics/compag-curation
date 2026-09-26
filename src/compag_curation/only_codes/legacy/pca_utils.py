# [ONLY-CODES-COMPAT] Vendored verbatim from Only_codes/Clean/sam2_pipeline/pca_utils.py.
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import json, os

def _load_embeddings_from_npy(path: str) -> np.ndarray:
    p = Path(path)
    if p.is_dir():
        mats = []
        for f in sorted(p.glob("*.npy")):
            try:
                z = np.load(str(f)).astype(np.float32)
                if z.ndim == 1:
                    z = z[None, :]
                mats.append(z)
            except Exception:
                pass
        if not mats:
            raise RuntimeError(f"No .npy embeddings in {p}")
        return np.vstack(mats)
    Z = np.load(str(p)).astype(np.float32)
    if Z.ndim == 1:
        Z = Z[None, :]
    return Z

def _load_embeddings_from_npz(path: str) -> np.ndarray:
    p = Path(path)
    if p.is_dir():
        mats = []
        for f in sorted(p.glob("*.npz")):
            try:
                d = np.load(str(f))
                for key in ("Z","embeddings","arr_0"):
                    if key in d:
                        z = d[key].astype(np.float32)
                        break
                else:
                    continue
                if z.ndim == 1:
                    z = z[None, :]
                mats.append(z)
            except Exception:
                pass
        if not mats:
            raise RuntimeError(f"No usable arrays in {p}")
        return np.vstack(mats)
    d = np.load(str(p))
    for key in ("Z","embeddings","arr_0"):
        if key in d:
            Z = d[key].astype(np.float32)
            break
    else:
        raise RuntimeError("NPZ missing 'Z' or 'embeddings'")
    if Z.ndim == 1:
        Z = Z[None, :]
    return Z

def _load_embeddings_from_csv(path: str) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(str(path))
    cols = list(df.columns)
    mu_cols = [c for c in cols if c.lower().startswith("mu_") or c.lower().startswith("z_")]
    Z_list: List[np.ndarray] = []
    if mu_cols:
        mu_cols_sorted = sorted(mu_cols, key=lambda c: int(''.join(filter(str.isdigit, c)) or 0))
        Z_list.append(df[mu_cols_sorted].values.astype(np.float32))
    if 'z' in df.columns:
        def _parse_json_arr(x):
            try:
                return np.array(json.loads(x), dtype=np.float32)
            except Exception:
                return None
        zs = [ _parse_json_arr(x) for x in df['z'].astype(str).tolist() ]
        zs = [z for z in zs if z is not None]
        if zs:
            Z_list.append(np.vstack(zs))
    if not Z_list:
        raise RuntimeError("CSV must have mu_0.. or z JSON column")
    return np.vstack(Z_list)

def _stack_from_dirs(dirs_csv: str) -> np.ndarray:
    from pathlib import Path
    mats = []
    for d in [Path(s.strip()) for s in dirs_csv.split(',') if s.strip()]:
        if d.exists() and d.is_dir():
            for f in sorted(list(d.glob('*.npy')) + list(d.glob('*.npz'))):
                try:
                    if f.suffix.lower() == '.npy':
                        mats.append(np.load(str(f)).astype(np.float32))
                    else:
                        dd = np.load(str(f))
                        key = 'Z' if 'Z' in dd else ('embeddings' if 'embeddings' in dd else 'arr_0')
                        mats.append(dd[key].astype(np.float32))
                except Exception:
                    pass
    if not mats:
        raise RuntimeError("No .npy/.npz files found in provided dirs")
    mats = [m[None, :] if m.ndim == 1 else m for m in mats]
    return np.vstack(mats)

def _compute_pca(Z: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2:
        raise RuntimeError("Embeddings must be 2D (N,D)")
    N, D = Z.shape
    if k > min(N, D):
        k = min(N, D)
    mean = Z.mean(axis=0)
    Zc = Z - mean
    # Economy SVD
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    comps = Vt[:k].astype(np.float32)
    return comps, mean.astype(np.float32)

def run_make_pca(args) -> int:
    sources: List[np.ndarray] = []
    if getattr(args, 'pca_from_npy', ''):
        sources.append(_load_embeddings_from_npy(args.pca_from_npy))
    if getattr(args, 'pca_from_npz', ''):
        sources.append(_load_embeddings_from_npz(args.pca_from_npz))
    if getattr(args, 'pca_from_csv', ''):
        sources.append(_load_embeddings_from_csv(args.pca_from_csv))
    if getattr(args, 'pca_stack_dirs', ''):
        sources.append(_stack_from_dirs(args.pca_stack_dirs))
    if not sources:
        raise SystemExit("[ERROR] No PCA source provided. Use one of --pca-from-npy/npz/csv or --pca-stack-dirs.")
    Z = np.vstack(sources)
    print(f"[INFO] PCA builder: stacked embeddings {Z.shape}")
    comps, mean = _compute_pca(Z, int(getattr(args, 'pca_dims', 16)))
    outp = str(getattr(args, 'make_embed_pca'))
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, components=comps, mean=mean)
    print(f"[DONE] PCA saved to: {outp} (k={comps.shape[0]}, d={comps.shape[1]})")
    return 0

def resolve_pca_path(args, out_dir: Path) -> str:
    explicit = str(getattr(args, 'embed_pca', '')).strip()
    if explicit:
        return explicit
    k = int(getattr(args, 'embed_pca_preset', 0) or 0)
    if k in (16, 32):
        candidates = [
            out_dir / f"embed_pca_{k}.npz",
            Path('models') / f"embed_pca_{k}.npz",
            Path.cwd() / f"embed_pca_{k}.npz",
        ]
        for c in candidates:
            if c.exists():
                print(f"[INFO] Using PCA preset {k}: {c}")
                return str(c)
        print(f"[WARN] PCA preset {k} not found in: {', '.join(str(x) for x in candidates)}")
    return ""
