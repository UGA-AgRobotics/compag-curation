"""Stable persisted-estimator implementation for reviewed XGBoost pipelines.

This module intentionally imports ``imbalanced-learn`` because ``SafeSMOTE``
must be a real ``SMOTE`` subclass for estimator cloning and deserialization.
Planning and package facades do not import this module; it is reached only by
an explicit estimator import or a separately authorized scientific workflow.
"""

from __future__ import annotations

from typing import Any

from imblearn.over_sampling import SMOTE


class SafeSMOTE(SMOTE):
    """Skip unsafe or unnecessary SMOTE operations.

    The control flow and state mutation preserve the reviewed historical
    implementation: labels are normalized to an array, the current class
    ratio can short-circuit resampling, ``self.k_neighbors`` is reduced in
    place, and a ``ValueError`` from the parent sampler falls back to the
    unchanged features and normalized labels.
    """

    def fit_resample(self, features: Any, labels: Any) -> tuple[Any, Any]:
        import numpy as np

        normalized_labels = np.asarray(labels)
        values, counts = np.unique(normalized_labels, return_counts=True)
        if len(values) < 2:
            return features, normalized_labels
        minority = int(counts.min())
        majority = int(counts.max())
        if minority < 2 or majority < 2:
            return features, normalized_labels
        current_ratio = minority / float(majority)
        strategy = self.sampling_strategy
        if isinstance(strategy, float):
            if current_ratio >= float(strategy) - 1e-12:
                return features, normalized_labels
        try:
            neighbors = int(self.k_neighbors)
            self.k_neighbors = max(1, min(neighbors, minority - 1))
        except Exception:
            _neighbors_remain_unchanged = self.k_neighbors
        try:
            return super().fit_resample(features, normalized_labels)
        except ValueError:
            return features, normalized_labels


def build_smote_safe(
    labels: Any,
    sampling_strategy: float = 0.5,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> SafeSMOTE | None:
    """Construct the reviewed sampler only when both classes can resample."""

    import numpy as np

    normalized_labels = np.asarray(labels)
    values, counts = np.unique(normalized_labels, return_counts=True)
    if len(values) < 2 or int(counts.min()) < 2:
        return None
    return SafeSMOTE(
        sampling_strategy=float(sampling_strategy),
        k_neighbors=int(k_neighbors),
        random_state=int(random_state),
    )


__all__ = ["SafeSMOTE", "build_smote_safe"]
