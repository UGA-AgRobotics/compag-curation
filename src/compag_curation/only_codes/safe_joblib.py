"""Restricted reader for legacy ``foldsafe_pack.joblib`` data packs.

The notebook stores ``{"used_img_ids": [...], "mu": ndarray, "embed_thresh":
float, "pca_components": ndarray, "pca_mean": ndarray}`` with ``joblib.dump``.
Plain ``joblib.load`` would execute arbitrary pickle opcodes.  This reader uses
joblib's own ``NumpyUnpickler`` (so array bytes are decoded identically) but
refuses every global except the NumPy array reconstruction machinery.  It is
used for data packs only; trained classifiers go through the explicit-trust
conversion in :mod:`model_bundle`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_ALLOWED_GLOBALS = {
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
}


class RestrictedPickleError(RuntimeError):
    pass


def load_data_pack(path: Path | str) -> Any:
    import joblib.numpy_pickle as jnp

    class _Restricted(jnp.NumpyUnpickler):
        def find_class(self, module, name):  # noqa: D401
            if (module, name) in _ALLOWED_GLOBALS or (
                module in ("numpy.dtypes",) and name.endswith("DType")
            ):
                return super().find_class(module, name)
            raise RestrictedPickleError(f"refusing global {module}.{name} in a data pack")

    p = Path(path)
    with p.open("rb") as fobj:
        unpickler = _Restricted(str(p), fobj, ensure_native_byte_order="auto", mmap_mode=None)
        return unpickler.load()
