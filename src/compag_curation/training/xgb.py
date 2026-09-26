"""Source-backed XGBoost training pipeline with explicit state.

Nine operational training cells repeated and incrementally refined the same
pipeline.  Their common algorithms live here as module-level functions.  No
scientific dependency is imported until an authorized handler calls
``train_xgb``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from compag_curation.contracts import ContractError, HandlerServices


LEAK_PATTERNS = (
    r"^class(_?id)?$",
    r"^category(_?id)?$",
    r"^cat(_?id)?$",
    r"^cid$",
    r"^name$",
    r"^target$",
    r"^y$",
    r".*(_|^)class(_|$).*",
    r".*(_|^)category(_|$).*",
    r".*(_|^)catname(_|$).*",
    r".*(_|^)classname(_|$).*",
)
MUST_KEEP_FEATURES = (
    "pred_iou",
    "stability",
    "embed_sim",
    "g_quality",
    "g_light",
    "g_color",
    "g_shape",
    "g_embed",
    "g_robust",
    "g_maha",
    "g_border",
)
REVIEW_TAG_WEIGHTS = {
    "accept": 1.0,
    "flip": 1.0,
    "sus_accept": 0.4,
    "sus_flip": 0.4,
    "skip": 0.0,
}
GROUP_POS_FRAC = 0.50


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    apply_before_search: bool = True
    exclude_discrete_max_nunique: int = 10
    mixup_enabled: bool = True
    mixup_alpha: float = 0.2
    mixup_multiplier: float = 0.5
    dropout_enabled: bool = True
    dropout_probability: float = 0.05
    dropout_multiplier: float = 1.0
    dropout_strategy: str = "median_by_class"
    jitter_enabled: bool = True
    jitter_multiplier: float = 0.5
    jitter_sigma: float = 0.01
    jitter_per_feature: bool = True
    jitter_clip_quantiles: tuple[float, float] = (0.001, 0.999)
    shuffle_rows: bool = True

    def __post_init__(self) -> None:
        if self.exclude_discrete_max_nunique != 10:
            raise ContractError("augmentation discrete-feature cutoff changed")
        if self.dropout_strategy != "median_by_class":
            raise ContractError("augmentation dropout strategy changed")
        low, high = self.jitter_clip_quantiles
        if not 0.0 <= low < high <= 1.0:
            raise ContractError("augmentation clipping quantiles are invalid")


@dataclass(frozen=True)
class XGBTrainingRequest:
    feature_table: Path
    split_manifest: Path
    model_output: Path
    variant: str
    random_state: int = 42
    decision_threshold: float = 0.5
    feature_order_sha256: str | None = None
    test_size: float = 0.2
    n_splits_cv: int = 5
    positive_label: int = 1
    top_features_to_report: int = 30
    border_pixels: int = 4
    tile_size: int = 512
    tiny_minimum_width_height: int = 32
    tiny_maximum_area: int = 1500
    border_weight: float = 0.5
    uncertainty_margin: float = 0.05
    uncertainty_multiplier: float = 0.7
    drop_tiny_in_train: bool = False
    drop_border_from_test: bool = True
    early_stopping_rounds: int = 50
    early_stop_validation_fraction: float = 0.3
    large_estimator_count: int = 2000
    use_review_weighting: bool = True
    correct_review_weight: float = 0.7
    wrong_review_weight: float = 1.0
    group_positive_fraction: float = 0.5
    smote_sampling_strategy: float = 0.5
    smote_neighbors: int = 5
    hyperparameter_iterations: int = 30
    force_hyperparameter_search: bool = False
    device: str = "cpu"
    thread_count: int = 1
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def __post_init__(self) -> None:
        if self.variant not in {"coco-xgb", "xgb", "xgb-main"}:
            raise ContractError("XGBoost training variant is invalid")
        if self.random_state != 42:
            raise ContractError("sealed training seed must remain 42")
        if self.decision_threshold != 0.5:
            raise ContractError("sealed decision threshold must remain 0.5")
        if self.device not in {"cpu", "cuda"}:
            raise ContractError("training device must be explicit cpu or cuda")
        if isinstance(self.thread_count, bool) or not isinstance(self.thread_count, int) or self.thread_count < 1:
            raise ContractError("training thread count must be a positive integer")
        if not 0.0 < self.test_size < 1.0:
            raise ContractError("test_size must be between zero and one")
        if not 0.0 < self.smote_sampling_strategy <= 1.0:
            raise ContractError("SMOTE sampling strategy is invalid")
        if self.feature_order_sha256 is not None and (
            len(self.feature_order_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.feature_order_sha256)
        ):
            raise ContractError("feature-order digest must be lowercase SHA-256")


@dataclass(frozen=True)
class XGBTrainingResult:
    output_root: Path
    model_output: Path
    threshold_output: Path
    metrics_output: Path
    feature_importance_output: Path
    tile_predictions_output: Path
    image_predictions_output: Path
    config_output: Path
    dropped_columns_output: Path
    selected_feature_count: int
    selected_threshold: float
    split_mode: str


def original_image_group(value: Any) -> str:
    """Collapse tiled names to their original-image group."""

    name = Path(str(value)).name
    stem = Path(name).stem
    patterns = (
        r"^(.*?)(?:_y\d+x\d+)$",
        r"^(.*?)(?:[_-]tile[_-]?\d+)$",
        r"^(.*?)(?:[_-][rc]\d+[_-][rc]\d+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return stem


# SOURCE_CELL: NB-LIVE-0006-C0000
# SOURCE_CELL: NB-LIVE-0006-C0001
# SOURCE_CELL: NB-LIVE-0006-C0002
# SOURCE_STATEMENT_MAP: compute-group-labels -> compute_group_labels/group-fallback
def compute_group_labels(
    groups: Any,
    labels: Any,
    fraction: float = GROUP_POS_FRAC,
) -> tuple[Any, Any]:
    """Return any-positive group labels, with the source fraction fallback."""

    import pandas as pd

    grouped = (
        pd.DataFrame({"g": groups, "y": labels})
        .groupby("g")["y"]
        .agg(["max", "mean"])
    )
    labels_any = (grouped["max"] > 0).astype(int)
    if labels_any.nunique() == 1:
        labels_fraction = (grouped["mean"] >= fraction).astype(int)
        return grouped.index.to_numpy(), labels_fraction.to_numpy()
    return grouped.index.to_numpy(), labels_any.to_numpy()


# SOURCE_CELL: NB-LIVE-0001-C0005
# SOURCE_CELL: NB-LIVE-0007-C0000
# SOURCE_CELL: NB-LIVE-0007-C0006
# SOURCE_STATEMENT_MAP: compute-group-labels-any -> compute_group_labels_any
def compute_group_labels_any(
    groups: Any,
    labels: Any,
    fraction: float = GROUP_POS_FRAC,
) -> tuple[Any, Any]:
    """Return the source any-positive label variant; fraction is API parity."""

    import pandas as pd

    del fraction
    grouped = (
        pd.DataFrame({"g": groups, "y": labels})
        .groupby("g")["y"]
        .agg(["max", "mean"])
    )
    labels_any = (grouped["max"] > 0).astype(int)
    if labels_any.nunique() == 1:
        return grouped.index.to_numpy(), labels_any.to_numpy()
    return grouped.index.to_numpy(), labels_any.to_numpy()


# SOURCE_CELL: NB-LIVE-0001-C0005
# SOURCE_STATEMENT_MAP: normalize-features-csv -> normalize_features_csv_add_review_tag
def normalize_features_csv_add_review_tag(
    csv_in: Path,
    csv_out: Path | None = None,
) -> Path:
    """Normalize mixed old/new feature rows without loading the CSV in memory."""

    import csv

    if csv_out is None:
        csv_out = csv_in.with_suffix(".normalized.csv")
    with csv_in.open("r", newline="", encoding="utf-8", errors="replace") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if not header:
            raise ContractError("feature CSV is empty or missing its header")
        header = [item.strip() for item in header]
        has_review_tag = "review_tag" in header
        if has_review_tag:
            tag_index = header.index("review_tag")
            output_header = header
            expected_count = len(output_header)
        else:
            if "reviewed" in header:
                tag_index = header.index("reviewed") + 1
            elif "class_id" in header:
                tag_index = header.index("class_id")
            else:
                tag_index = len(header)
            output_header = header[:tag_index] + ["review_tag"] + header[tag_index:]
            expected_count = len(output_header)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(output_header)
            input_count = len(header)
            for row in reader:
                if not row:
                    continue
                if not has_review_tag:
                    if len(row) == input_count:
                        row = row[:tag_index] + [""] + row[tag_index:]
                    elif len(row) == input_count + 1:
                        pass
                    elif len(row) < input_count:
                        row = row + [""] * (input_count - len(row))
                        row = row[:tag_index] + [""] + row[tag_index:]
                    else:
                        row = row[:input_count] + [",".join(row[input_count:])]
                        row = row[:tag_index] + [""] + row[tag_index:]
                else:
                    if len(row) < expected_count:
                        row = row + [""] * (expected_count - len(row))
                    elif len(row) > expected_count:
                        row = row[: expected_count - 1] + [
                            ",".join(row[expected_count - 1 :])
                        ]
                if len(row) < expected_count:
                    row = row + [""] * (expected_count - len(row))
                elif len(row) > expected_count:
                    row = row[: expected_count - 1] + [
                        ",".join(row[expected_count - 1 :])
                    ]
                writer.writerow(row)
    return csv_out


# SOURCE_CELL: NB-LIVE-0001-C0005
# SOURCE_STATEMENT_MAP: read-features-csv-robust -> read_features_csv_robust
def read_features_csv_robust(csv_path: Path) -> tuple[Any, Path]:
    """Read feature data and repair the historical mixed review-tag schema."""

    import pandas as pd
    from pandas.errors import ParserError

    try:
        frame = pd.read_csv(csv_path, low_memory=False, keep_default_na=False)
        used = csv_path
    except ParserError:
        used = normalize_features_csv_add_review_tag(csv_path)
        frame = pd.read_csv(used, low_memory=False, keep_default_na=False)
    if "review_tag" not in frame.columns:
        frame["review_tag"] = ""
    frame["review_tag"] = frame["review_tag"].astype(str).str.strip()
    frame.loc[
        frame["review_tag"].str.lower().isin({"nan", "none"}), "review_tag"
    ] = ""
    if "reviewed" in frame.columns:
        frame["reviewed"] = (
            pd.to_numeric(frame["reviewed"], errors="coerce").fillna(0).astype(int)
        )
    else:
        frame["reviewed"] = 0
    if "label" not in frame.columns:
        raise ContractError("feature CSV is missing required label column")
    frame["label"] = (
        pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype(int)
    )
    if "scale" in frame.columns:
        frame["scale"] = pd.to_numeric(frame["scale"], errors="coerce")
    return frame, used


# SOURCE_CELL: NB-LIVE-0001-C0005
# SOURCE_CELL: NB-LIVE-0006-C0001
# SOURCE_CELL: NB-LIVE-0006-C0002
# SOURCE_CELL: NB-LIVE-0007-C0000
# SOURCE_CELL: NB-LIVE-0007-C0006
# SOURCE_STATEMENT_MAP: fit-maybe-callbacks -> fit_with_maybe_callbacks
def fit_with_maybe_callbacks(model: Any, features: Any, labels: Any, **kwargs: Any) -> Any:
    import inspect

    signature = inspect.signature(model.fit)
    if "callbacks" not in signature.parameters and "callbacks" in kwargs:
        kwargs.pop("callbacks", None)
    return model.fit(features, labels, **kwargs)


# SOURCE_CELL: NB-LIVE-0001-C0005
# SOURCE_STATEMENT_MAP: ensure-numeric-series -> _ensure_numeric_series
def _ensure_numeric_series(series: Any, default: float = 0.0) -> Any:
    import pandas as pd

    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)


# SOURCE_CELL: NB-LIVE-0001-C0005
# SOURCE_STATEMENT_MAP: border-tiny-mask -> border_and_tiny_masks_from_df_auto
def border_and_tiny_masks_from_df_auto(
    frame: Any,
    name_column: str,
    tile_size_default: int = 512,
    border_pixels: int = 4,
    tiny_minimum_width_height: int = 32,
    tiny_maximum_area: int = 1500,
) -> tuple[Any, Any]:
    import pandas as pd

    bbox_x = _ensure_numeric_series(frame.get("bbox_x", 0))
    bbox_y = _ensure_numeric_series(frame.get("bbox_y", 0))
    bbox_width = _ensure_numeric_series(frame.get("bbox_w", 0))
    bbox_height = _ensure_numeric_series(frame.get("bbox_h", 0))
    if {"tile_w", "tile_h"}.issubset(frame.columns):
        tile_width = _ensure_numeric_series(frame["tile_w"]).clip(lower=1.0)
        tile_height = _ensure_numeric_series(frame["tile_h"]).clip(lower=1.0)
    else:
        per_tile = (
            pd.DataFrame(
                {
                    name_column: frame[name_column].astype(str),
                    "mx_w": bbox_x + bbox_width,
                    "mx_h": bbox_y + bbox_height,
                }
            )
            .groupby(name_column, sort=False)
            .agg({"mx_w": "max", "mx_h": "max"})
        )
        tile_width = (
            frame[name_column]
            .astype(str)
            .map(per_tile["mx_w"])
            .fillna(float(tile_size_default))
            .clip(lower=1.0)
        )
        tile_height = (
            frame[name_column]
            .astype(str)
            .map(per_tile["mx_h"])
            .fillna(float(tile_size_default))
            .clip(lower=1.0)
        )
    border_fraction = float(border_pixels) / float(tile_size_default)
    tiny_minimum_fraction = float(tiny_minimum_width_height) / float(tile_size_default)
    tiny_area_fraction = float(tiny_maximum_area) / float(
        tile_size_default * tile_size_default
    )
    border = (
        (bbox_x <= border_fraction * tile_width)
        | (bbox_y <= border_fraction * tile_height)
        | (bbox_x + bbox_width >= (1.0 - border_fraction) * tile_width)
        | (bbox_y + bbox_height >= (1.0 - border_fraction) * tile_height)
    ).to_numpy()
    tiny = (
        (bbox_width < tiny_minimum_fraction * tile_width)
        | (bbox_height < tiny_minimum_fraction * tile_height)
        | (bbox_width * bbox_height < tiny_area_fraction * (tile_width * tile_height))
    ).to_numpy()
    return border, tiny


# SOURCE_CELL: NB-LIVE-0001-C0005
# SOURCE_CELL: NB-LIVE-0006-C0000
# SOURCE_CELL: NB-LIVE-0006-C0001
# SOURCE_CELL: NB-LIVE-0006-C0002
# SOURCE_CELL: NB-LIVE-0007-C0000
# SOURCE_CELL: NB-LIVE-0007-C0006
# SOURCE_STATEMENT_MAP: pick-name-column -> pick_name_column
def pick_name_column(frame: Any) -> str:
    for column in ("image", "file_name", "filename", "file"):
        if column in frame.columns:
            return column
    raise ContractError("feature CSV has no supported image-name column")


def _read_split_groups(path: Path) -> tuple[set[str], set[str], set[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("training split manifest cannot be read") from exc
    if not isinstance(value, Mapping):
        raise ContractError("training split manifest must be an object")

    aliases = {
        "train": ("train", "train_groups", "orig_train_ids"),
        "validation": ("validation", "val", "validation_groups", "orig_val_ids"),
        "test": ("test", "test_groups", "orig_test_ids"),
    }
    allowed = {
        name for names in aliases.values() for name in names
    } | {"random_state", "freeze_existing"}
    if set(value) - allowed:
        raise ContractError("training split manifest fields are invalid")

    def normalized(role: str) -> set[str]:
        names = aliases[role]
        present = [name for name in names if name in value]
        if len(present) > 1:
            raise ContractError(f"training split manifest has ambiguous {role} fields")
        raw: Any = value[present[0]] if present else []
        if not isinstance(raw, list):
            raise ContractError("training split members must be lists")
        normalized_values: list[str] = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, (int, str)):
                raise ContractError("training split members must be integer or string identifiers")
            rendered = str(item).strip()
            if not rendered:
                raise ContractError("training split identifiers must be nonempty")
            normalized_values.append(rendered)
        if len(set(normalized_values)) != len(normalized_values):
            raise ContractError("training split identifiers must be unique")
        return set(normalized_values)

    train = normalized("train")
    validation = normalized("validation")
    test = normalized("test")
    if train & validation or train & test or validation & test:
        raise ContractError("training split groups overlap")
    if not train or not (validation or test):
        raise ContractError("training split manifest requires nonempty train and held-out groups")
    return train, validation, test


def _resolve_manifest_groups(
    known_groups: set[str],
    train: set[str],
    validation: set[str],
    test: set[str],
) -> tuple[set[str], set[str]]:
    held_out = validation | test
    unassigned = known_groups - train - held_out
    if unassigned:
        raise ContractError("training feature groups are absent from the sealed split manifest")
    explicit_train = known_groups & train
    explicit_held_out = known_groups & held_out
    if not explicit_train or not explicit_held_out:
        raise ContractError("sealed split manifest produces an empty observed train or held-out set")
    return explicit_train, explicit_held_out


def pick_groups_tile_balanced(
    groups: Any,
    *,
    test_fraction: float,
    random_state: int,
    tolerance: float = 0.01,
    maximum_tries: int = 2000,
) -> tuple[set[Any], set[Any], int, int]:
    """Select whole groups while matching the requested tile count."""

    import numpy as np

    values, counts = np.unique(groups, return_counts=True)
    target = int(round(len(groups) * test_fraction))
    generator = np.random.default_rng(random_state)
    best: tuple[int, set[Any]] | None = None
    for _ in range(maximum_tries):
        order = generator.permutation(len(values))
        selected: set[Any] = set()
        total = 0
        for index in order:
            candidate = total + int(counts[index])
            if abs(candidate - target) <= abs(total - target) or not selected:
                selected.add(values[index])
                total = candidate
            if abs(total - target) <= max(1, int(tolerance * len(groups))):
                break
        difference = abs(total - target)
        if best is None or difference < best[0]:
            best = difference, selected
        if difference == 0:
            break
    assert best is not None
    test_groups = best[1]
    train_groups = set(values.tolist()) - test_groups
    observed = sum(int(counts[index]) for index, group in enumerate(values) if group in test_groups)
    return train_groups, test_groups, target, observed


def leakage_scan_vectorized(
    frame: Any,
    labels: Any,
    *,
    excluded_names: set[str],
    correlation_threshold: float = 0.999,
) -> tuple[list[str], list[str]]:
    """Remove literal label aliases, constant columns, and near-label copies."""

    import numpy as np
    import pandas as pd

    compiled = tuple(re.compile(pattern, re.IGNORECASE) for pattern in LEAK_PATTERNS)
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    keep: list[str] = []
    dropped: list[str] = []
    normalized_labels = np.asarray(labels, dtype=float)
    for column in numeric:
        if column in excluded_names or any(pattern.fullmatch(column) for pattern in compiled):
            dropped.append(column)
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.nunique(dropna=False) <= 1:
            dropped.append(column)
            continue
        valid = series.notna().to_numpy() & np.isfinite(normalized_labels)
        if valid.sum() > 2:
            values = series.to_numpy(dtype=float)[valid]
            target = normalized_labels[valid]
            correlation = np.corrcoef(values, target)[0, 1]
            if math.isfinite(float(correlation)) and abs(float(correlation)) >= correlation_threshold:
                dropped.append(column)
                continue
            if np.array_equal(values, target) or np.array_equal(values, 1.0 - target):
                dropped.append(column)
                continue
        keep.append(column)
    return keep, sorted(set(dropped))


def _promote_source_features(frame: Any, labels: Any, columns: Sequence[str]) -> list[str]:
    import numpy as np
    import pandas as pd

    promoted: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().sum() < 2 or series.nunique(dropna=True) <= 1:
            continue
        values = series.fillna(series.median()).to_numpy(dtype=float)
        if np.array_equal(values, labels) or np.array_equal(values, 1 - labels):
            continue
        promoted.append(column)
    return promoted


def build_review_weights(
    frame: Any,
    labels: Any,
    train_indices: Any,
    request: XGBTrainingRequest,
) -> Any:
    import numpy as np

    weights = np.ones(len(train_indices), dtype=np.float32)
    if not request.use_review_weighting:
        return weights
    training = frame.iloc[train_indices]
    if "review_weight" in training:
        provided = training["review_weight"].fillna(1.0).to_numpy(dtype=float)
        weights *= np.clip(provided, 0.0, None).astype(np.float32)
    elif "review_tag" in training:
        tags = training["review_tag"].fillna("").astype(str).str.casefold()
        weights *= tags.map(REVIEW_TAG_WEIGHTS).fillna(1.0).to_numpy(dtype=np.float32)
    prediction_column = next(
        (name for name in ("model_pred", "final_pred", "xgb_pred") if name in training),
        None,
    )
    if prediction_column:
        predictions = training[prediction_column].fillna(-1).to_numpy(dtype=int)
        truth = np.asarray(labels)[train_indices]
        weights *= np.where(
            predictions == truth,
            request.correct_review_weight,
            request.wrong_review_weight,
        ).astype(np.float32)
    if {"bbox_x", "bbox_y", "bbox_w", "bbox_h"} <= set(training.columns):
        name_column = pick_name_column(training)
        border, tiny = border_and_tiny_masks_from_df_auto(
            training,
            name_column,
            tile_size_default=request.tile_size,
            border_pixels=request.border_pixels,
            tiny_minimum_width_height=request.tiny_minimum_width_height,
            tiny_maximum_area=request.tiny_maximum_area,
        )
        weights *= np.where(border, request.border_weight, 1.0).astype(np.float32)
        if request.drop_tiny_in_train:
            weights *= np.where(tiny, 0.0, 1.0).astype(np.float32)
    probability_column = next(
        (name for name in ("prob", "xgb_prob", "proba") if name in training),
        None,
    )
    if probability_column:
        probabilities = training[probability_column].to_numpy(dtype=float)
        uncertain = np.abs(probabilities - request.decision_threshold) <= request.uncertainty_margin
        weights *= np.where(uncertain, request.uncertainty_multiplier, 1.0).astype(np.float32)
    return weights


def _continuous_indices(frame: Any, maximum_unique: int) -> list[int]:
    return [
        index
        for index, column in enumerate(frame.columns)
        if frame[column].nunique(dropna=False) > maximum_unique
    ]


def _mixup_same_class(
    features: Any,
    labels: Any,
    groups: Any,
    weights: Any,
    *,
    indices: Sequence[int],
    alpha: float,
    multiplier: float,
    generator: Any,
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    count = int(round(len(features) * multiplier))
    if count <= 0 or not indices:
        return features[:0], labels[:0], groups[:0], weights[:0]
    rows: list[Any] = []
    output_labels: list[int] = []
    output_groups: list[Any] = []
    output_weights: list[float] = []
    for _ in range(count):
        label = generator.choice(np.unique(labels))
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < 2:
            continue
        first, second = generator.choice(candidates, size=2, replace=False)
        lam = float(generator.beta(alpha, alpha))
        row = np.asarray(features[first], dtype=float).copy()
        row[list(indices)] = (
            lam * np.asarray(features[first])[list(indices)]
            + (1.0 - lam) * np.asarray(features[second])[list(indices)]
        )
        rows.append(row)
        output_labels.append(int(label))
        output_groups.append(groups[first])
        output_weights.append(float(lam * weights[first] + (1.0 - lam) * weights[second]))
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(output_labels, dtype=int),
        np.asarray(output_groups),
        np.asarray(output_weights, dtype=np.float32),
    )


def _dropout_augmentation(
    features: Any,
    labels: Any,
    groups: Any,
    weights: Any,
    *,
    indices: Sequence[int],
    probability: float,
    multiplier: float,
    generator: Any,
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    count = int(round(len(features) * multiplier))
    if count <= 0 or not indices:
        return features[:0], labels[:0], groups[:0], weights[:0]
    selected = generator.choice(len(features), size=count, replace=True)
    output = np.asarray(features[selected], dtype=np.float32).copy()
    for label in np.unique(labels):
        class_rows = np.asarray(features)[labels == label]
        medians = np.nanmedian(class_rows[:, list(indices)], axis=0)
        positions = np.flatnonzero(labels[selected] == label)
        if not len(positions):
            continue
        mask = generator.random((len(positions), len(indices))) < probability
        block = output[np.ix_(positions, list(indices))]
        block[mask] = np.broadcast_to(medians, block.shape)[mask]
        output[np.ix_(positions, list(indices))] = block
    return output, labels[selected], groups[selected], weights[selected]


def _jitter_augmentation(
    features: Any,
    labels: Any,
    groups: Any,
    weights: Any,
    *,
    indices: Sequence[int],
    multiplier: float,
    sigma: float,
    quantiles: tuple[float, float],
    generator: Any,
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    count = int(round(len(features) * multiplier))
    if count <= 0 or not indices:
        return features[:0], labels[:0], groups[:0], weights[:0]
    selected = generator.choice(len(features), size=count, replace=True)
    output = np.asarray(features[selected], dtype=np.float32).copy()
    source = np.asarray(features, dtype=float)[:, list(indices)]
    scale = np.nanstd(source, axis=0) * sigma
    noise = generator.normal(0.0, scale, size=(count, len(indices)))
    block = output[:, list(indices)] + noise
    low = np.nanquantile(source, quantiles[0], axis=0)
    high = np.nanquantile(source, quantiles[1], axis=0)
    output[:, list(indices)] = np.clip(block, low, high)
    return output, labels[selected], groups[selected], weights[selected]


def apply_augmentations(
    frame: Any,
    labels: Any,
    groups: Any,
    weights: Any,
    config: AugmentationConfig,
    random_state: int,
) -> tuple[Any, Any, Any, Any]:
    """Apply reviewed same-class mixup, dropout, jitter, then shuffle."""

    import numpy as np

    features = frame.to_numpy(dtype=np.float32)
    normalized_labels = np.asarray(labels, dtype=int)
    normalized_groups = np.asarray(groups)
    normalized_weights = np.asarray(weights, dtype=np.float32)
    if not config.enabled:
        return features, normalized_labels, normalized_groups, normalized_weights
    indices = _continuous_indices(frame, config.exclude_discrete_max_nunique)
    generator = np.random.default_rng(random_state)
    additions: list[tuple[Any, Any, Any, Any]] = []
    if config.mixup_enabled:
        additions.append(
            _mixup_same_class(
                features,
                normalized_labels,
                normalized_groups,
                normalized_weights,
                indices=indices,
                alpha=config.mixup_alpha,
                multiplier=config.mixup_multiplier,
                generator=generator,
            )
        )
    if config.dropout_enabled:
        additions.append(
            _dropout_augmentation(
                features,
                normalized_labels,
                normalized_groups,
                normalized_weights,
                indices=indices,
                probability=config.dropout_probability,
                multiplier=config.dropout_multiplier,
                generator=generator,
            )
        )
    if config.jitter_enabled:
        additions.append(
            _jitter_augmentation(
                features,
                normalized_labels,
                normalized_groups,
                normalized_weights,
                indices=indices,
                multiplier=config.jitter_multiplier,
                sigma=config.jitter_sigma,
                quantiles=config.jitter_clip_quantiles,
                generator=generator,
            )
        )
    for augmented_features, augmented_labels, augmented_groups, augmented_weights in additions:
        if len(augmented_features):
            features = np.vstack((features, augmented_features))
            normalized_labels = np.concatenate((normalized_labels, augmented_labels))
            normalized_groups = np.concatenate((normalized_groups, augmented_groups))
            normalized_weights = np.concatenate((normalized_weights, augmented_weights))
    if config.shuffle_rows:
        order = generator.permutation(len(features))
        features = features[order]
        normalized_labels = normalized_labels[order]
        normalized_groups = normalized_groups[order]
        normalized_weights = normalized_weights[order]
    return features, normalized_labels, normalized_groups, normalized_weights


# SOURCE_CELL: NB-LIVE-0001-C0006
# SOURCE_STATEMENT_MAP: ensure-numeric-frame -> ensure_numeric_df
def ensure_numeric_df(frame: Any, columns: Sequence[str]) -> Any:
    import pandas as pd

    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


# SOURCE_CELL: NB-LIVE-0001-C0006
# SOURCE_STATEMENT_MAP: build-group-cv-pairs -> build_cv_pairs
def build_cv_pairs(
    groups: Any,
    labels: Any,
    target_folds: int,
) -> list[tuple[Any, Any]]:
    import numpy as np
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return []
    fold_count = min(int(target_folds), len(unique_groups))
    if fold_count < 2:
        return []
    splitter = GroupKFold(n_splits=fold_count)
    pairs: list[tuple[Any, Any]] = []
    for train_indices, validation_indices in splitter.split(
        np.zeros(len(labels)), labels, groups
    ):
        if (
            len(np.unique(labels[train_indices])) < 2
            or len(np.unique(labels[validation_indices])) < 2
        ):
            continue
        pairs.append((train_indices, validation_indices))
    return pairs


# SOURCE_CELL: NB-LIVE-0001-C0006
# SOURCE_CELL: NB-LIVE-0007-C0000
# SOURCE_CELL: NB-LIVE-0007-C0006
# SOURCE_STATEMENT_MAP: weights-after-smote -> make_weights_after_smote
def make_weights_after_smote(
    original_weights: Any,
    original_labels: Any,
    resampled_labels: Any,
) -> Any:
    import numpy as np
    import pandas as pd

    original_labels = np.asarray(original_labels)
    resampled_labels = np.asarray(resampled_labels)
    source_weights = np.asarray(original_weights, dtype=np.float32)
    original_count = len(original_labels)
    resampled_count = len(resampled_labels)
    weights = np.empty(resampled_count, dtype=np.float32)
    weights[:original_count] = source_weights[:original_count]
    if resampled_count > original_count:
        synthetic_labels = resampled_labels[original_count:]
        if synthetic_labels.size > 0:
            synthetic_class = int(pd.Series(synthetic_labels).mode().iloc[0])
            mean_weight = (
                float(source_weights[original_labels == synthetic_class].mean())
                if np.any(original_labels == synthetic_class)
                else float(source_weights.mean())
            )
            weights[original_count:] = mean_weight
        else:
            weights[original_count:] = float(source_weights.mean())
    return weights


# SOURCE_CELL: NB-LIVE-0001-C0006
# SOURCE_STATEMENT_MAP: threshold-pr-f1 -> pick_threshold_by_pr_f1
def pick_threshold_by_pr_f1(
    labels: Any,
    probabilities: Any,
) -> tuple[float, dict[str, float], Any]:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return (
            0.5,
            {
                "best_f1": float("nan"),
                "best_precision": float(precision[-1]),
                "best_recall": float(recall[-1]),
            },
            pd.DataFrame(
                {
                    "precision": precision,
                    "recall": recall,
                    "threshold": np.r_[np.nan],
                }
            ),
        )
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_index = int(np.nanargmax(f1))
    threshold = float(thresholds[max(best_index - 1, 0)])
    curve = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.r_[np.nan, thresholds],
        }
    )
    statistics = {
        "best_f1": float(f1[best_index]),
        "best_precision": float(precision[best_index]),
        "best_recall": float(recall[best_index]),
    }
    return threshold, statistics, curve


def pick_threshold_by_pr(labels: Any, probabilities: Any) -> float:
    """Select the PR threshold with maximum F1; fall back to sealed 0.5."""

    threshold, _, _ = pick_threshold_by_pr_f1(labels, probabilities)
    return threshold


# SOURCE_CELL: NB-LIVE-0001-C0006
# SOURCE_CELL: NB-LIVE-0006-C0000
# SOURCE_CELL: NB-LIVE-0006-C0001
# SOURCE_CELL: NB-LIVE-0006-C0002
# SOURCE_CELL: NB-LIVE-0007-C0001
# SOURCE_CELL: NB-LIVE-0007-C0007
# SOURCE_STATEMENT_MAP: predict-probability-batches -> predict_proba_batched
def predict_proba_batched(
    model: Any,
    features: Any,
    batch: int = 50000,
    description: str = "Predict proba",
) -> Any:
    import numpy as np

    del description
    count = len(features)
    output = np.empty(count, dtype=np.float32)
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        batch_features = (
            features.iloc[start:stop]
            if hasattr(features, "iloc")
            else features[start:stop]
        )
        output[start:stop] = model.predict_proba(batch_features)[:, 1]
    return output


def _feature_order_digest(columns: Sequence[str]) -> str:
    payload = "\n".join(columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metrics(labels: Any, probabilities: Any, threshold: float) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        roc_auc_score,
    )

    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) > 1
            else None
        ),
        "pr_auc": (
            float(average_precision_score(labels, probabilities))
            if len(np.unique(labels)) > 1
            else None
        ),
        "confusion_matrix": matrix.astype(int).tolist(),
        "threshold": float(threshold),
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2), encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


# SOURCE_CELL: NB-LIVE-0001-C0005
# SOURCE_CELL: NB-LIVE-0001-C0006
# SOURCE_CELL: NB-LIVE-0006-C0000
# SOURCE_CELL: NB-LIVE-0006-C0001
# SOURCE_CELL: NB-LIVE-0006-C0002
# SOURCE_CELL: NB-LIVE-0007-C0000
# SOURCE_CELL: NB-LIVE-0007-C0001
# SOURCE_CELL: NB-LIVE-0007-C0006
# SOURCE_CELL: NB-LIVE-0007-C0007
# SOURCE_STATEMENT_MAP: training-cell-closure -> train_xgb/full-source-backed-pipeline
def train_xgb(
    request: XGBTrainingRequest,
    services: HandlerServices,
) -> XGBTrainingResult:
    """Train, select a threshold, evaluate, and persist a stable pipeline."""

    import joblib
    import numpy as np
    import pandas as pd
    from imblearn.pipeline import Pipeline as ImbalancedPipeline
    from packaging import version
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedKFold
    from xgboost import XGBClassifier
    import xgboost

    from .safe_smote import build_smote_safe

    frame, _ = read_features_csv_robust(request.feature_table)
    name_column = pick_name_column(frame)
    labels = frame["label"].astype(int).to_numpy()
    groups = frame[name_column].astype(str).map(original_image_group).to_numpy()
    excluded = {
        name_column,
        "image",
        "file_name",
        "filename",
        "file",
        "id",
        "ann_id",
        "label",
        "group",
        "area_px",
        "perim_sqrt",
        "reviewed",
        "review_tag",
        "review_weight",
    }
    numeric_columns, dropped_columns = leakage_scan_vectorized(
        frame,
        labels,
        excluded_names=excluded,
    )
    promoted = _promote_source_features(
        frame,
        labels,
        (*MUST_KEEP_FEATURES, *(name for name in frame if name.startswith("embed_pca_"))),
    )
    feature_columns = sorted(set(numeric_columns) | set(promoted))
    if not feature_columns:
        raise ContractError("no nonleaking numeric training features remain")
    observed_order_hash = _feature_order_digest(feature_columns)
    if request.feature_order_sha256 and observed_order_hash != request.feature_order_sha256:
        raise ContractError("training feature order SHA-256 does not match")
    features = ensure_numeric_df(frame[feature_columns], feature_columns).astype(np.float32)

    manifest_train, manifest_validation, manifest_test = _read_split_groups(
        request.split_manifest
    )
    known_groups = set(groups.tolist())
    train_groups, test_groups = _resolve_manifest_groups(
        known_groups,
        manifest_train,
        manifest_validation,
        manifest_test,
    )
    split_mode = "Manifest(GroupPure)"
    train_indices = np.flatnonzero(np.isin(groups, tuple(train_groups)))
    test_indices = np.flatnonzero(np.isin(groups, tuple(test_groups)))
    if not len(train_indices) or not len(test_indices):
        raise ContractError("group-pure train/test split is empty")
    if set(groups[train_indices]) & set(groups[test_indices]):
        raise ContractError("group leakage detected after split")

    review_weights = build_review_weights(frame, labels, train_indices, request)
    train_frame = features.iloc[train_indices]
    train_labels = labels[train_indices]
    train_groups_array = groups[train_indices]
    augmented_features, augmented_labels, augmented_groups, augmented_weights = apply_augmentations(
        train_frame,
        train_labels,
        train_groups_array,
        review_weights,
        request.augmentation,
        request.random_state,
    )

    use_cuda = request.device == "cuda"
    xgb_parameters: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "random_state": request.random_state,
        "n_jobs": 1 if use_cuda else request.thread_count,
        "tree_method": "hist",
    }
    if use_cuda:
        if version.parse(xgboost.__version__) >= version.parse("2.0.0"):
            xgb_parameters["device"] = "cuda"
        else:
            xgb_parameters["tree_method"] = "gpu_hist"

    sampler = build_smote_safe(
        augmented_labels,
        sampling_strategy=request.smote_sampling_strategy,
        k_neighbors=request.smote_neighbors,
        random_state=request.random_state,
    )
    base_model = XGBClassifier(**xgb_parameters)
    pipeline = ImbalancedPipeline(
        (
            ("imp", SimpleImputer(strategy="median")),
            ("smote", sampler if sampler is not None else "passthrough"),
            ("clf", base_model),
        )
    )
    best_parameters: dict[str, Any] = {}
    cv_score: float | None = None
    if len(np.unique(augmented_labels)) < 2:
        pipeline = ImbalancedPipeline(
            (
                ("imp", SimpleImputer(strategy="median")),
                ("smote", "passthrough"),
                ("clf", DummyClassifier(strategy="most_frequent")),
            )
        )
        fit_with_maybe_callbacks(pipeline, augmented_features, augmented_labels)
    else:
        distributions = {
            "clf__max_depth": [3, 4, 5, 6, 7, 8, 9],
            "clf__min_child_weight": [0.5, 1.0, 2.0, 5.0, 10.0],
            "clf__subsample": [0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "clf__reg_lambda": [0.1, 1.0, 10.0],
            "clf__reg_alpha": [0.0, 0.01, 0.1, 1.0],
        }
        cv_pairs = build_cv_pairs(
            augmented_groups,
            augmented_labels,
            request.n_splits_cv,
        )
        if cv_pairs and (request.force_hyperparameter_search or request.variant != "xgb"):
            search = RandomizedSearchCV(
                pipeline,
                distributions,
                n_iter=request.hyperparameter_iterations,
                scoring="average_precision",
                cv=cv_pairs,
                random_state=request.random_state,
                n_jobs=1,
                refit=True,
            )
            fit_parameters: dict[str, Any] = {}
            if sampler is None:
                fit_parameters["clf__sample_weight"] = augmented_weights
            fit_with_maybe_callbacks(
                search,
                augmented_features,
                augmented_labels,
                groups=augmented_groups,
                **fit_parameters,
            )
            pipeline = search.best_estimator_
            best_parameters = dict(search.best_params_)
            cv_score = float(search.best_score_)
        else:
            fit_parameters = {}
            if sampler is None:
                fit_parameters["clf__sample_weight"] = augmented_weights
            fit_with_maybe_callbacks(
                pipeline,
                augmented_features,
                augmented_labels,
                **fit_parameters,
            )

    test_features = features.iloc[test_indices].to_numpy(dtype=np.float32)
    test_labels = labels[test_indices]
    probabilities = predict_proba_batched(pipeline, test_features)
    selected_threshold = (
        pick_threshold_by_pr(test_labels, probabilities)
        if len(np.unique(test_labels)) > 1
        else request.decision_threshold
    )
    tile_metrics = _metrics(test_labels, probabilities, selected_threshold)
    predictions = pd.DataFrame(
        {
            "image": frame.iloc[test_indices][name_column].astype(str).to_numpy(),
            "group": groups[test_indices],
            "label": test_labels,
            "probability": probabilities,
            "prediction": (probabilities >= selected_threshold).astype(int),
        }
    )
    image_predictions = (
        predictions.groupby("group", as_index=False)
        .agg(label=("label", "max"), probability=("probability", "max"))
    )
    image_metrics = _metrics(
        image_predictions["label"].to_numpy(),
        image_predictions["probability"].to_numpy(),
        selected_threshold,
    )

    output_root = request.model_output
    output_root.mkdir(parents=True, exist_ok=False)
    model_output = output_root / "model.joblib"
    threshold_output = output_root / "threshold.json"
    metrics_output = output_root / "metrics.json"
    importance_output = output_root / "feature_importances.csv"
    tile_output = output_root / "tile_preds.csv"
    image_output = output_root / "image_preds.csv"
    config_output = output_root / "config.json"
    dropped_output = output_root / "dropped_cols.json"
    classifier = pipeline.named_steps.get("clf")
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        importance_frame = pd.DataFrame(
            {"feature": feature_columns, "importance": np.zeros(len(feature_columns))}
        )
    else:
        importance_frame = pd.DataFrame(
            {"feature": feature_columns, "importance": np.asarray(importances, dtype=float)}
        ).sort_values("importance", ascending=False)

    metadata = {
        "feature_table": str(request.feature_table),
        "variant": request.variant,
        "random_state": request.random_state,
        "decision_threshold": request.decision_threshold,
        "selected_threshold": selected_threshold,
        "feature_order_sha256": observed_order_hash,
        "xgboost_version": xgboost.__version__,
        "device": request.device,
        "used_cv": cv_score is not None,
        "cv_best_pr_auc": cv_score,
        "best_parameters": best_parameters,
        "split_mode": split_mode,
        "stable_sampler": "compag_curation.training.safe_smote.SafeSMOTE",
    }
    temporary_model = output_root / ".model.joblib.tmp"
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": feature_columns,
            "threshold": float(selected_threshold),
            "meta": metadata,
        },
        temporary_model,
    )
    temporary_model.replace(model_output)
    _atomic_json(threshold_output, {"threshold": selected_threshold})
    _atomic_json(
        metrics_output,
        {"tile": tile_metrics, "image": image_metrics, "cv_best_pr_auc": cv_score},
    )
    _atomic_json(config_output, {**asdict(request), "feature_table": str(request.feature_table), "split_manifest": str(request.split_manifest), "model_output": str(output_root)})
    _atomic_json(dropped_output, dropped_columns)
    importance_frame.to_csv(importance_output, index=False)
    predictions.to_csv(tile_output, index=False)
    image_predictions.to_csv(image_output, index=False)
    services.report(
        f"XGBoost model written: features={len(feature_columns)} "
        f"threshold={selected_threshold:.6f}"
    )
    return XGBTrainingResult(
        output_root,
        model_output,
        threshold_output,
        metrics_output,
        importance_output,
        tile_output,
        image_output,
        config_output,
        dropped_output,
        len(feature_columns),
        float(selected_threshold),
        split_mode,
    )


__all__ = [
    "AugmentationConfig",
    "GROUP_POS_FRAC",
    "XGBTrainingRequest",
    "XGBTrainingResult",
    "apply_augmentations",
    "border_and_tiny_masks_from_df_auto",
    "build_cv_pairs",
    "build_review_weights",
    "compute_group_labels",
    "compute_group_labels_any",
    "ensure_numeric_df",
    "fit_with_maybe_callbacks",
    "leakage_scan_vectorized",
    "make_weights_after_smote",
    "normalize_features_csv_add_review_tag",
    "original_image_group",
    "pick_name_column",
    "pick_groups_tile_balanced",
    "pick_threshold_by_pr",
    "pick_threshold_by_pr_f1",
    "predict_proba_batched",
    "read_features_csv_robust",
    "train_xgb",
]
