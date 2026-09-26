"""Authorized source-backed NB7 hybrid XGBoost training algorithms.

The reviewed source specification is the implementation authority. Scientific libraries are
resolved only from the authorized service boundary.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from hashlib import sha256
from os import fstat
from pathlib import Path
from stat import S_ISREG
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from ..contracts import ContractError, ExecutionPolicy, require_authorized
from .source_paths import prepare_source_output_directory

ALLOWED_THREAD_ENVIRONMENT_KEYS = frozenset({'MKL_NUM_THREADS', 'OMP_NUM_THREADS'})
SOURCE_ALGORITHM_ORDER = ('NB-LIVE-0007-C0000', 'NB-LIVE-0007-C0001', 'NB-LIVE-0007-C0006', 'NB-LIVE-0007-C0007')
SOURCE_BODY_SHA256: Mapping[str, str] = {
    'NB-LIVE-0007-C0000': '696adc214b8b0ede703ec6923ad8ba80c82b691f4d60219b5769e913c602030f',
    'NB-LIVE-0007-C0001': '732da35241ac3e1c84592f1b7d12d4133652907af9c24382c48d9a9b3de5a5d8',
    'NB-LIVE-0007-C0006': 'b021b9ed6071b68a73a32b8ba8de226fd8374898663013a1212e07a1bd7a9cc0',
    'NB-LIVE-0007-C0007': 'b0d1accbd6f05d3e329562f3ef0d295c6c3ed16292016fea81bed9a51d819834',
}
SOURCE_OWNER_COUNT: Mapping[str, int] = {
    'NB-LIVE-0007-C0000': 332,
    'NB-LIVE-0007-C0001': 300,
    'NB-LIVE-0007-C0006': 441,
    'NB-LIVE-0007-C0007': 322,
}
SOURCE_TRANSFORMATION_CLASSES: Mapping[str, tuple[str, ...]] = {
    'NB-LIVE-0007-C0000': ('AMBIENT_CPU_COUNT_TO_EXPLICIT_SERVICE', 'AMBIENT_ENVIRONMENT_WRITE_TO_INJECTED_SERVICE', 'AUTO_DISCOVERY_TO_EXPLICIT_VERIFIED_ASSETS', 'CAPABILITY_PROBE_TO_EXPLICIT_DEVICE_POLICY', 'PRINT_DISPLAY_TO_INJECTED_REPORTER', 'SCIENTIFIC_IMPORT_TO_INJECTED_DEPENDENCY', 'SOURCE_TOKEN_TO_TYPED_ROOT', 'TOKENIZED_TRAINING_PATH_TO_VERIFIED_ARTIFACT', 'TYPE_ANNOTATION_NORMALIZATION'),
    'NB-LIVE-0007-C0001': ('CAPABILITY_PROBE_TO_EXPLICIT_DEVICE_POLICY', 'EXPLICIT_TYPED_CONTINUATION_STATE', 'PRINT_DISPLAY_TO_INJECTED_REPORTER', 'SCIENTIFIC_IMPORT_TO_INJECTED_DEPENDENCY', 'TYPE_ANNOTATION_NORMALIZATION'),
    'NB-LIVE-0007-C0006': ('AMBIENT_CPU_COUNT_TO_EXPLICIT_SERVICE', 'AMBIENT_ENVIRONMENT_WRITE_TO_INJECTED_SERVICE', 'AUTO_DISCOVERY_TO_EXPLICIT_VERIFIED_ASSETS', 'CAPABILITY_PROBE_TO_EXPLICIT_DEVICE_POLICY', 'LEGACY_RESIDUAL_TO_EXPLICIT_DERIVED_STATE', 'PRINT_DISPLAY_TO_INJECTED_REPORTER', 'SCIENTIFIC_IMPORT_TO_INJECTED_DEPENDENCY', 'SOURCE_TOKEN_TO_TYPED_ROOT', 'TOKENIZED_TRAINING_PATH_TO_VERIFIED_ARTIFACT', 'TYPE_ANNOTATION_NORMALIZATION'),
    'NB-LIVE-0007-C0007': ('CAPABILITY_PROBE_TO_EXPLICIT_DEVICE_POLICY', 'EXPLICIT_TYPED_CONTINUATION_STATE', 'PRINT_DISPLAY_TO_INJECTED_REPORTER', 'SCIENTIFIC_IMPORT_TO_INJECTED_DEPENDENCY', 'TYPE_ANNOTATION_NORMALIZATION'),
}

class SourceAlgorithmDependencies(Protocol):
    DummyClassifier: Any
    GroupKFold: Any
    GroupShuffleSplit: Any
    ImbPipeline: Any
    Pipeline: Any
    RandomizedSearchCV: Any
    SMOTE: Any
    SimpleImputer: Any
    StratifiedKFold: Any
    StratifiedShuffleSplit: Any
    TrainingCallback: Any
    XGBClassifier: Any
    average_precision_score: Any
    confusion_matrix: Any
    cupy: Any
    joblib: Any
    loguniform: Any
    np: Any
    pd: Any
    plt: Any
    precision_recall_curve: Any
    randint: Any
    roc_auc_score: Any
    roc_curve: Any
    tqdm: Any
    uniform: Any
    version: Any
    xgb: Any

class ArrayValue(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, key: object) -> object: ...

class TableValue(ArrayValue, Protocol):
    @property
    def columns(self) -> object: ...

class EstimatorValue(Protocol):
    def fit(self, *args: object, **kwargs: object) -> object: ...
    def predict_proba(self, values: object) -> object: ...

@dataclass(frozen=True)
class SourceInputArtifact:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ContractError("source input artifact path must be absolute")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ContractError("source input artifact SHA-256 must be lowercase hexadecimal")


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
        raw: object = value[present[0]] if present else []
        if not isinstance(raw, list):
            raise ContractError("training split members must be lists")
        members: list[str] = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, (int, str)):
                raise ContractError("training split members must be integer or string identifiers")
            rendered = str(item).strip()
            if not rendered:
                raise ContractError("training split identifiers must be nonempty")
            members.append(rendered)
        if len(set(members)) != len(members):
            raise ContractError("training split identifiers must be unique")
        return set(members)

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
    if known_groups - train - held_out:
        raise ContractError("training feature groups are absent from the sealed split manifest")
    explicit_train = known_groups & train
    explicit_held_out = known_groups & held_out
    if not explicit_train or not explicit_held_out:
        raise ContractError("sealed split manifest produces an empty observed train or held-out set")
    return explicit_train, explicit_held_out

@dataclass(frozen=True)
class XgbDevicePolicy:
    mode: str = 'cpu'

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {'cpu', 'cuda'}:
            raise ContractError("XGBoost device mode must be cpu or cuda")

@dataclass(frozen=True)
class SourceAlgorithmServices:
    dependencies: SourceAlgorithmDependencies
    report: Callable[..., None]
    approved_input_artifacts: tuple[SourceInputArtifact, ...]
    cpu_count: int
    thread_environment_setter: Callable[[str, str], None]
    device_policy: XgbDevicePolicy = XgbDevicePolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.approved_input_artifacts, tuple) or any(
            not isinstance(artifact, SourceInputArtifact)
            for artifact in self.approved_input_artifacts
        ):
            raise ContractError("approved source input artifacts must be a typed tuple")
        if not callable(self.report) or not callable(self.thread_environment_setter):
            raise ContractError("source algorithm callbacks must be callable")
        if not isinstance(self.device_policy, XgbDevicePolicy):
            raise ContractError("source algorithm device policy must be typed")
        paths = tuple(artifact.path for artifact in self.approved_input_artifacts)
        if len(paths) != len(set(paths)):
            raise ContractError("approved source input artifact paths must be unique")
        if isinstance(self.cpu_count, bool) or not isinstance(self.cpu_count, int) or self.cpu_count < 1:
            raise ContractError("source algorithm CPU count must be a positive integer")
        object.__setattr__(self, "approved_input_artifacts", tuple(self.approved_input_artifacts))

    def preflight_input_artifact(
        self,
        artifact: SourceInputArtifact,
        policy: ExecutionPolicy,
    ) -> Path:
        require_authorized(policy)
        if artifact not in self.approved_input_artifacts:
            raise ContractError("source input artifact is not explicitly approved")
        candidate = artifact.path
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ContractError("source input artifact is unavailable") from exc
        if resolved != candidate or candidate.is_symlink():
            raise ContractError("source input artifact and its path components must not be symlinks")
        try:
            status = candidate.lstat()
        except OSError as exc:
            raise ContractError("source input artifact is unavailable") from exc
        if not S_ISREG(status.st_mode):
            raise ContractError("source input artifact must be a regular file")
        actual = sha256()
        try:
            with candidate.open("rb") as stream:
                opened = fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size) != (
                    status.st_dev, status.st_ino, status.st_mode, status.st_size
                ):
                    raise ContractError("source input artifact changed before verification")
                while chunk := stream.read(1024 * 1024):
                    actual.update(chunk)
                completed = fstat(stream.fileno())
        except OSError as exc:
            raise ContractError("source input artifact could not be verified") from exc
        try:
            final = candidate.lstat()
        except OSError as exc:
            raise ContractError("source input artifact changed after verification") from exc
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns
        )
        if identity(status) != identity(opened) or identity(opened) != identity(completed) or identity(completed) != identity(final):
            raise ContractError("source input artifact changed during verification")
        if actual.hexdigest() != artifact.sha256:
            raise ContractError("source input artifact SHA-256 mismatch")
        return candidate

    def set_thread_environment(
        self,
        name: str,
        value: str,
        policy: ExecutionPolicy,
    ) -> None:
        require_authorized(policy)
        if name not in ALLOWED_THREAD_ENVIRONMENT_KEYS:
            raise ContractError("source algorithm requested an unreviewed thread environment key")
        rendered = str(value)
        if not rendered.isdigit() or int(rendered) < 1:
            raise ContractError("thread environment value must be a positive integer")
        self.thread_environment_setter(name, rendered)

@dataclass(frozen=True)
class TrainingArtifacts:
    output_directory: Path
    model: Path
    threshold: Path
    feature_importances: Path
    tile_predictions: Path
    image_predictions: Path
    precision_recall_curve: Path
    roc_curve: Path
    confusion_matrix: Path
    metrics: Path
    dropped_columns: Path
    configuration: Path
    history: Path
    progress: Path

def _artifacts(
    ROUND_DIR: Path, MODEL_PATH: Path, THRESH_JSON_PATH: Path,
    FI_CSV_PATH: Path, TILE_PREDS_CSV_PATH: Path, IMG_PREDS_CSV_PATH: Path,
    PR_CURVE_CSV_PATH: Path, ROC_CURVE_CSV_PATH: Path, CONF_MAT_CSV_PATH: Path,
    METRICS_JSON_PATH: Path, DROPPED_COLS_JSON: Path, CONFIG_JSON_PATH: Path,
    HISTORY_CSV_PATH: Path, PROGRESS_PNG_PATH: Path,
) -> TrainingArtifacts:
    return TrainingArtifacts(
        ROUND_DIR, MODEL_PATH, THRESH_JSON_PATH, FI_CSV_PATH, TILE_PREDS_CSV_PATH,
        IMG_PREDS_CSV_PATH, PR_CURVE_CSV_PATH, ROC_CURVE_CSV_PATH,
        CONF_MAT_CSV_PATH, METRICS_JSON_PATH, DROPPED_COLS_JSON,
        CONFIG_JSON_PATH, HISTORY_CSV_PATH, PROGRESS_PNG_PATH,
    )

@dataclass(frozen=True)
class HybridContinuationState:
    CONFIG_JSON_PATH: Path
    CONF_MAT_CSV_PATH: Path
    CSV_PATH: str
    split_manifest: SourceInputArtifact
    previous_tile_predictions: SourceInputArtifact
    previous_threshold: SourceInputArtifact
    DROPPED_COLS_JSON: Path
    EARLY_STOP_ROUNDS: int
    EARLY_STOP_VAL_FRAC: float
    FI_CSV_PATH: Path
    GROUP_POS_FRAC: float
    HISTORY_CSV_PATH: Path
    IMG_PREDS_CSV_PATH: Path
    METRICS_JSON_PATH: Path
    MODEL_NAME: str
    MODEL_PATH: Path
    NAME_COL: str
    N_ESTIMATORS_BIG: int
    N_SPLITS_CV: int
    POS_LABEL: int
    PROGRESS_PNG_PATH: Path
    PR_CURVE_CSV_PATH: Path
    RANDOM_STATE: int
    ROC_CURVE_CSV_PATH: Path
    ROUND_DIR: Path
    ROUND_TAG: str
    TEST_SIZE: float
    THRESH_JSON_PATH: Path
    TILE_PREDS_CSV_PATH: Path
    TOPK_FEATURES_TO_PRINT: int
    TQDMCallback: type[object]
    Timer: type[object]
    USE_REVIEW_WEIGHTING: bool
    X_test: TableValue
    X_train: TableValue
    total_started_at: float
    build_smote: Callable[..., object]
    compute_group_labels: Callable[..., object]
    df: TableValue
    fit_with_maybe_callbacks: Callable[..., object]
    grid_n_jobs: int
    groups_test: ArrayValue
    groups_train: ArrayValue
    make_weights_after_smote: Callable[..., object]
    num_cols: tuple[str, ...]
    pipe: EstimatorValue
    review_weights_train: ArrayValue
    smote_global: EstimatorValue | None
    test_idx: ArrayValue
    to_drop: frozenset[str]
    train_idx: ArrayValue
    xgb_kwargs: Mapping[str, object]
    y_test: ArrayValue
    y_train: ArrayValue

    def __post_init__(self) -> None:
        if any(not isinstance(artifact, SourceInputArtifact) for artifact in (self.split_manifest, self.previous_tile_predictions, self.previous_threshold)):
            raise ContractError('NB7 hybrid continuation artifacts must remain typed')
        object.__setattr__(self, 'num_cols', tuple(self.num_cols))
        object.__setattr__(self, 'to_drop', frozenset(self.to_drop))
        object.__setattr__(self, 'xgb_kwargs', MappingProxyType(dict(self.xgb_kwargs)))

@dataclass(frozen=True)
class AugmentedContinuationState:
    AUG_CFG: Mapping[str, object]
    CONFIG_JSON_PATH: Path
    CONF_MAT_CSV_PATH: Path
    CSV_PATH: str
    split_manifest: SourceInputArtifact
    previous_tile_predictions: SourceInputArtifact
    previous_threshold: SourceInputArtifact
    DROPPED_COLS_JSON: Path
    EARLY_STOP_ROUNDS: int
    EARLY_STOP_VAL_FRAC: float
    FI_CSV_PATH: Path
    GROUP_POS_FRAC: float
    HISTORY_CSV_PATH: Path
    IMG_PREDS_CSV_PATH: Path
    METRICS_JSON_PATH: Path
    MODEL_NAME: str
    MODEL_PATH: Path
    NAME_COL: str
    N_ESTIMATORS_BIG: int
    N_SPLITS_CV: int
    POS_LABEL: int
    PROGRESS_PNG_PATH: Path
    PR_CURVE_CSV_PATH: Path
    RANDOM_STATE: int
    ROC_CURVE_CSV_PATH: Path
    ROUND_DIR: Path
    ROUND_TAG: str
    TEST_SIZE: float
    THRESH_JSON_PATH: Path
    TILE_PREDS_CSV_PATH: Path
    TOPK_FEATURES_TO_PRINT: int
    TQDMCallback: type[object]
    Timer: type[object]
    USE_REVIEW_WEIGHTING: bool
    X_test: TableValue
    X_train: TableValue
    total_started_at: float
    apply_augs_in_3A: Callable[..., object]
    build_smote: Callable[..., object]
    compute_group_labels: Callable[..., object]
    df: TableValue
    fit_with_maybe_callbacks: Callable[..., object]
    grid_n_jobs: int
    groups_test: ArrayValue
    groups_train: ArrayValue
    make_weights_after_smote: Callable[..., object]
    num_cols: tuple[str, ...]
    review_weights_train: ArrayValue
    test_idx: ArrayValue
    to_drop: frozenset[str]
    train_idx: ArrayValue
    xgb_base: EstimatorValue
    xgb_kwargs: Mapping[str, object]
    y_test: ArrayValue
    y_train: ArrayValue

    def __post_init__(self) -> None:
        if any(not isinstance(artifact, SourceInputArtifact) for artifact in (self.split_manifest, self.previous_tile_predictions, self.previous_threshold)):
            raise ContractError('NB7 augmented continuation artifacts must remain typed')
        object.__setattr__(self, 'AUG_CFG', MappingProxyType(dict(self.AUG_CFG)))
        object.__setattr__(self, 'num_cols', tuple(self.num_cols))
        object.__setattr__(self, 'to_drop', frozenset(self.to_drop))
        object.__setattr__(self, 'xgb_kwargs', MappingProxyType(dict(self.xgb_kwargs)))

# SOURCE_CELL: NB-LIVE-0007-C0000
# SOURCE_STATEMENT_MAP: sanitized-body -> source_algorithm_0007_0000
@dataclass(frozen=True)
class SourceInputs_0007_0000:
    output_root: Path
    training_features: SourceInputArtifact
    split_manifest: SourceInputArtifact
    previous_tile_predictions: SourceInputArtifact
    previous_threshold: SourceInputArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path) or not self.output_root.is_absolute():
            raise ContractError("training output root must be absolute")
        if any(
            not isinstance(artifact, SourceInputArtifact)
            for artifact in (
                self.training_features,
                self.split_manifest,
                self.previous_tile_predictions,
                self.previous_threshold,
            )
        ):
            raise ContractError("NB7 inputs must be typed source input artifacts")

@dataclass(frozen=True)
class SourceResult_0007_0000:
    artifacts: TrainingArtifacts
    continuation: HybridContinuationState

def source_algorithm_0007_0000(
    inputs: SourceInputs_0007_0000,
    services: SourceAlgorithmServices,
    policy: ExecutionPolicy,
) -> SourceResult_0007_0000:
    require_authorized(policy)
    _training_features = services.preflight_input_artifact(inputs.training_features, policy)
    _split_manifest = services.preflight_input_artifact(inputs.split_manifest, policy)
    _manifest_train, _manifest_validation, _manifest_test = _read_split_groups(
        _split_manifest
    )
    _verified_previous_artifacts = (
        services.preflight_input_artifact(inputs.previous_tile_predictions, policy),
        services.preflight_input_artifact(inputs.previous_threshold, policy),
    )
    _previous_tile_predictions = _verified_previous_artifacts[0]
    _previous_threshold = _verified_previous_artifacts[1]
    PREV_TILE_PREDS_PATH = str(_previous_tile_predictions)
    PREV_THRESH_JSON_PATH = str(_previous_threshold)
    import os, re, json, warnings, time
    from pathlib import Path
    np = services.dependencies.np
    pd = services.dependencies.pd
    loguniform = services.dependencies.loguniform
    randint = services.dependencies.randint
    uniform = services.dependencies.uniform
    GroupShuffleSplit = services.dependencies.GroupShuffleSplit
    GroupKFold = services.dependencies.GroupKFold
    StratifiedShuffleSplit = services.dependencies.StratifiedShuffleSplit
    RandomizedSearchCV = services.dependencies.RandomizedSearchCV
    roc_auc_score = services.dependencies.roc_auc_score
    average_precision_score = services.dependencies.average_precision_score
    precision_recall_curve = services.dependencies.precision_recall_curve
    roc_curve = services.dependencies.roc_curve
    confusion_matrix = services.dependencies.confusion_matrix
    Pipeline = services.dependencies.Pipeline
    SimpleImputer = services.dependencies.SimpleImputer
    DummyClassifier = services.dependencies.DummyClassifier
    tqdm = services.dependencies.tqdm
    from contextlib import contextmanager
    joblib = services.dependencies.joblib
    SMOTE = services.dependencies.SMOTE
    ImbPipeline = services.dependencies.ImbPipeline
    warnings.filterwarnings('ignore', message='This pattern is interpreted as a regular expression', category=UserWarning)

    @contextmanager
    def tqdm_joblib(tqdm_object):
        """progress for joblib (e.g., Grid/RandomizedSearchCV)"""

        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):

            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)
        old_cb = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_cb
            tqdm_object.close()

    class Timer:

        def __init__(self, label):
            self.label = label

        def __enter__(self):
            self.t0 = time.perf_counter()
            services.report(f'[TIMER] {self.label} ...')
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            services.report(f'[TIMER] {self.label} took {dt:.2f} sec')
    __T_TOTAL0 = time.perf_counter()
    CSV_PATH = str(_training_features)
    OUT_DIR_ROOT = inputs.output_root
    MODEL_NAME = 'cj_ultra_tilesafe_xgb_r16_hybrid.pkl'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_SPLITS_CV = 5
    POS_LABEL = 1
    TOPK_FEATURES_TO_PRINT = 30
    EARLY_STOP_ROUNDS = 50
    EARLY_STOP_VAL_FRAC = 0.3
    N_ESTIMATORS_BIG = 2000
    USE_REVIEW_WEIGHTING = True
    WEIGHT_CORRECT = 0.7
    WEIGHT_WRONG = 1.0
    AUTO_DETECT_PREV_ROUND = False
    services.set_thread_environment('OMP_NUM_THREADS', '1', policy)
    services.set_thread_environment('MKL_NUM_THREADS', '1', policy)
    prepare_source_output_directory(OUT_DIR_ROOT)
    m = re.search('_r(\\d+)\\b', MODEL_NAME)
    ROUND_TAG = f'r{m.group(1)}' if m else 'r1'
    m2 = re.search('xgb_(.+?)\\.pkl$', os.path.basename(MODEL_NAME))
    MODEL_DIR_TAG = m2.group(1) if m2 else ROUND_TAG
    ROUND_DIR = OUT_DIR_ROOT / MODEL_DIR_TAG
    prepare_source_output_directory(ROUND_DIR)
    MODEL_PATH = ROUND_DIR / MODEL_NAME
    THRESH_JSON_PATH = ROUND_DIR / f'threshold_{ROUND_TAG}.json'
    FI_CSV_PATH = ROUND_DIR / 'feature_importances.csv'
    TILE_PREDS_CSV_PATH = ROUND_DIR / 'tile_preds.csv'
    IMG_PREDS_CSV_PATH = ROUND_DIR / 'image_preds.csv'
    PR_CURVE_CSV_PATH = ROUND_DIR / 'pr_curve.csv'
    ROC_CURVE_CSV_PATH = ROUND_DIR / 'roc_curve.csv'
    METRICS_JSON_PATH = ROUND_DIR / 'metrics.json'
    CONF_MAT_CSV_PATH = ROUND_DIR / 'confusion_matrix.csv'
    DROPPED_COLS_JSON = ROUND_DIR / 'dropped_cols.json'
    CONFIG_JSON_PATH = ROUND_DIR / 'config.json'
    HISTORY_CSV_PATH = OUT_DIR_ROOT / 'history.csv'
    PROGRESS_PNG_PATH = OUT_DIR_ROOT / 'progress.png'
    GROUP_POS_FRAC = 0.5

    def compute_group_labels(groups_vec, y_vec, frac=GROUP_POS_FRAC):
        gstat = pd.DataFrame({'g': groups_vec, 'y': y_vec}).groupby('g')['y'].agg(['max', 'mean'])
        labels_any = (gstat['max'] > 0).astype(int)
        if labels_any.nunique() == 1:
            return (gstat.index.to_numpy(), labels_any.to_numpy())
        return (gstat.index.to_numpy(), labels_any.to_numpy())
    import inspect

    def fit_with_maybe_callbacks(model, X, y, **kw):
        """Pass callbacks only if supported."""
        sig = inspect.signature(model.fit)
        if 'callbacks' not in sig.parameters and 'callbacks' in kw:
            kw.pop('callbacks', None)
        return model.fit(X, y, **kw)

    def leakage_scan_vectorized(df: pd.DataFrame, y: np.ndarray, drop_always: set, leak_patterns: list[str], corr_thr: float=0.999):
        num_df = df.drop(columns=list(drop_always), errors='ignore')
        num_df = num_df.select_dtypes(include=[np.number]).astype(np.float32)
        pat = '|'.join((f'(?:{p})' for p in leak_patterns))
        name_hit = num_df.columns.to_series().str.contains(pat, case=False, regex=True, na=False)
        drop_by_name = set(num_df.columns[name_hit])
        num_df_f = num_df.fillna(0.0)
        y_s = pd.Series(y.astype(np.float32), index=num_df_f.index)
        eq_y = num_df_f.eq(y_s, axis=0).all(axis=0)
        eq_inv = num_df_f.eq(1.0 - y_s, axis=0).all(axis=0)
        drop_by_equal = set(num_df_f.columns[eq_y | eq_inv])
        const_mask = num_df_f.nunique(dropna=False) <= 1
        corr = num_df_f.loc[:, ~const_mask].corrwith(y_s, method='pearson')
        corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        drop_by_corr = set(corr.index[corr.abs() > float(corr_thr)])
        to_drop = drop_by_name | drop_by_equal | drop_by_corr
        keep_cols = [c for c in num_df.columns if c not in to_drop and (not bool(const_mask.get(c, False)))]
        return (keep_cols, sorted(to_drop))
    df = pd.read_csv(CSV_PATH)

    def pick_name_col(df: pd.DataFrame) -> str:
        for c in ['image', 'file_name', 'filename', 'file']:
            if c in df.columns:
                return c
        raise ValueError("CSV must contain one of: 'image', 'file_name', 'filename', 'file'.")
    NAME_COL = pick_name_col(df)

    def original_image_group(name: str) -> str:
        base = Path(str(name)).name
        stem = Path(base).stem
        for pat in ['^(?P<root>.+?)_y\\d{1,8}x\\d{1,8}$', '^(?P<root>.+?)_x\\d{1,8}_y\\d{1,8}$', '^(?P<root>.+?)(?:__tile-\\d+|-tile-\\d+)$']:
            m = re.match(pat, stem, flags=re.IGNORECASE)
            if m:
                return m.group('root')
        return stem
    groups = df[NAME_COL].astype(str).map(original_image_group).values
    y = df['label'].astype(int).values

    def pick_groups_tile_balanced(groups_vec: np.ndarray, test_frac: float=0.2, random_state: int=42, tol: float=0.01, max_tries: int=2000):
        """زیرمجموعه\u200cای از گروه\u200cها را برای تست انتخاب می\u200cکند تا مجموع تایل\u200cها ≈ test_frac کل شود (بدون شکستن گروه\u200cها)."""
        uniq, inv = np.unique(groups_vec, return_inverse=True)
        counts = np.bincount(inv).astype(int)
        total = int(counts.sum())
        target = int(round(total * test_frac))
        rs = np.random.RandomState(random_state)
        best_sel, best_sum, best_diff = (None, 0, float('inf'))

        def greedy_from(order_idx):
            s = 0
            sel = set()
            for idx in order_idx:
                c = int(counts[idx])
                if abs(s + c - target) <= abs(s - target):
                    sel.add(uniq[idx])
                    s += c
            return (sel, s)
        orders = [np.argsort(-counts), np.argsort(counts)]
        for t in range(max_tries):
            if t < len(orders):
                order = orders[t]
            else:
                order = rs.permutation(len(uniq))
            sel, s = greedy_from(order)
            diff = abs(s - target)
            if diff < best_diff:
                best_sel, best_sum, best_diff = (sel, s, diff)
                if diff <= max(1, int(target * tol)):
                    break
        test_groups = set(best_sel)
        train_groups = set(uniq) - test_groups
        return (train_groups, test_groups, dict(zip(uniq, counts)), target, best_sum)
    groups = df[NAME_COL].astype(str).map(original_image_group).values
    y = df['label'].astype(int).values
    drop_always = {NAME_COL, 'image', 'file_name', 'filename', 'file', 'id', 'ann_id', 'label', 'group', 'area_px', 'perim_sqrt', 'reviewed'}
    LEAK_PATTERNS = ['^class(_?id)?$', '^category(_?id)?$', '^cat(_?id)?$', '^cid$', '^name$', '^target$', '^y$', '.*(_|^)class(_|$).*', '.*(_|^)category(_|$).*', '.*(_|^)catname(_|$).*', '.*(_|^)classname(_|$).*']
    ' #before\nnum_cols, dropped_cols = leakage_scan_vectorized(\n    df, y, drop_always, LEAK_PATTERNS, corr_thr=0.999\n)\nif dropped_cols:\n    print("[INFO] dropping potential leakage columns:", dropped_cols)\nif not num_cols:\n    raise RuntimeError("No usable numeric features after leakage/variance filters.")\nX = df[num_cols].copy().astype(np.float32)\n'
    num_cols, dropped_cols = leakage_scan_vectorized(df, y, drop_always, LEAK_PATTERNS, corr_thr=0.999)
    if dropped_cols:
        services.report('[INFO] dropping potential leakage columns:', dropped_cols)
    MUST_KEEP = ['pred_iou', 'stability', 'embed_sim', 'g_quality', 'g_light', 'g_color', 'g_shape', 'g_embed', 'g_robust', 'g_maha', 'g_border']
    PCA_KEEP = [c for c in df.columns if c.startswith('embed_pca_')]

    def _safe_promote(cols, df, y):
        np = services.dependencies.np
        pd = services.dependencies.pd
        ys = pd.Series(pd.to_numeric(y, errors='coerce'), index=df.index).astype(float).fillna(0.0)
        keep = []
        for c in cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            if s.nunique(dropna=False) <= 1:
                continue
            if s.equals(ys) or s.equals(1.0 - ys):
                continue
            r = np.corrcoef(s.to_numpy(), ys.to_numpy())[0, 1]
            if np.isfinite(r) and abs(r) > 0.999:
                continue
            keep.append(c)
        return keep
    keepers = _safe_promote(MUST_KEEP + PCA_KEEP, df, y)
    num_cols = sorted(set(num_cols).union(keepers))
    services.report(f'[FEATURES] final columns after union with gates/PCA: {len(num_cols)}')
    services.report('[FEATURES] gates in use:', [c for c in MUST_KEEP if c in num_cols])
    if not num_cols:
        raise RuntimeError('No usable numeric features after leakage/variance filters.')
    X = df[num_cols].copy().astype(np.float32)
    to_drop = set(dropped_cols)
    g_tr_set, g_te_set = _resolve_manifest_groups(
        set(groups.tolist()),
        _manifest_train,
        _manifest_validation,
        _manifest_test,
    )
    split_mode = 'SealedManifest(GroupPure)'
    train_idx = np.where(np.isin(groups, list(g_tr_set)))[0]
    test_idx = np.where(np.isin(groups, list(g_te_set)))[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ContractError('sealed split manifest produced an empty row partition')
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise ContractError('sealed split manifest produced group leakage')
    observed_held_out_ratio = len(test_idx) / (len(train_idx) + len(test_idx))
    if abs(observed_held_out_ratio - TEST_SIZE) > 0.01:
        raise ContractError('sealed split manifest held-out ratio exceeds the allowed tolerance')
    rs = np.random.RandomState(RANDOM_STATE)
    train_idx = rs.permutation(train_idx)
    test_idx = rs.permutation(test_idx)
    X_train, X_test = (X.iloc[train_idx], X.iloc[test_idx])
    y_train, y_test = (y[train_idx], y[test_idx])
    groups_train, groups_test = (groups[train_idx], groups[test_idx])
    frac = len(test_idx) / (len(train_idx) + len(test_idx))
    services.report(f'Train size: {len(train_idx)}, Test size: {len(test_idx)} | Split={split_mode} | tiles_test_frac={frac:.3f} | target={TEST_SIZE:.3f}')

    def _find_prev_round_dir(root: Path, current_round_tag: str) -> Path | None:
        raise ContractError('automatic previous-round discovery is disabled; explicit verified artifacts are required')

    def _load_prev_preds_and_thr(prev_preds_path: Path, prev_thr_path: Path, name_col: str) -> dict[str, int] | None:
        try:
            prev = pd.read_csv(prev_preds_path)
            thr = json.load(open(prev_thr_path))['threshold']
            if name_col not in prev.columns or 'proba' not in prev.columns:
                return None
            prev = prev[[name_col, 'proba']].copy()
            prev['pred'] = (prev['proba'] >= float(thr)).astype(int)
            return dict(zip(prev[name_col].astype(str), prev['pred'].astype(int)))
        except Exception:
            return None

    def build_review_weights_for_train(df: pd.DataFrame, name_col: str, train_idx: np.ndarray, y_train: np.ndarray, round_dir_root: Path, current_round_tag: str, weight_correct: float=0.7, weight_wrong: float=1.0) -> np.ndarray:
        """0.7 for items predicted correctly in previous round, 1.0 otherwise."""
        prev_preds_path = Path(PREV_TILE_PREDS_PATH) if PREV_TILE_PREDS_PATH else None
        prev_thr_path = Path(PREV_THRESH_JSON_PATH) if PREV_THRESH_JSON_PATH else None
        mapping = None
        if prev_preds_path is not None and prev_thr_path is not None:
            mapping = _load_prev_preds_and_thr(prev_preds_path, prev_thr_path, name_col)
        names_train = df.iloc[train_idx][name_col].astype(str).values
        w = np.full(len(train_idx), weight_wrong, dtype=np.float32)
        if mapping:
            preds_prev = np.array([mapping.get(n, -1) for n in names_train], dtype=int)
            known_mask = preds_prev >= 0
            if known_mask.any():
                correct = preds_prev[known_mask] == y_train[known_mask]
                w_sub = np.where(correct, weight_correct, weight_wrong).astype(np.float32)
                w[known_mask] = w_sub
        return w

    def make_weights_after_smote(original_weights: np.ndarray, y_original: np.ndarray, y_resampled: np.ndarray) -> np.ndarray:
        """Assume imblearn appends synthetic samples after originals."""
        n_orig = len(y_original)
        n_res = len(y_resampled)
        w = np.empty(n_res, dtype=np.float32)
        w[:n_orig] = original_weights[:n_orig]
        pos_mask = y_original == 1
        w_pos_mean = float(original_weights[pos_mask].mean()) if pos_mask.any() else float(original_weights.mean())
        if n_res > n_orig:
            w[n_orig:] = w_pos_mean
        return w
    review_weights_train = np.ones_like(y_train, dtype=np.float32)
    if USE_REVIEW_WEIGHTING:
        review_weights_train = build_review_weights_for_train(df=df, name_col=NAME_COL, train_idx=train_idx, y_train=y_train, round_dir_root=OUT_DIR_ROOT, current_round_tag=ROUND_TAG, weight_correct=WEIGHT_CORRECT, weight_wrong=WEIGHT_WRONG)
        services.report(f'[WEIGHTS] review-weights built. mean={review_weights_train.mean():.3f}')
        review_mask_train = df.iloc[train_idx].get('reviewed', 0).astype(int).values == 1
        review_weights_train = np.where(review_mask_train, review_weights_train, 1.0).astype(np.float32)
        services.report(f'[WEIGHTS] using review-only mask. frac_reviewed={review_mask_train.mean():.3f}')
    XGBClassifier = services.dependencies.XGBClassifier
    xgb = services.dependencies.xgb
    version = services.dependencies.version
    try:
        TrainingCallback = services.dependencies.TrainingCallback
    except Exception:
        TrainingCallback = object

    class TQDMCallback(TrainingCallback):

        def __init__(self, total: int, desc: str='XGBoost'):
            self.total = int(total)
            self.desc = str(desc)
            self.pbar = None

        def before_training(self, model):
            self.pbar = tqdm(total=self.total, desc=self.desc, leave=False)
            return model

        def after_iteration(self, model, epoch: int, evals_log: dict[str, dict[str, float]]):
            if self.pbar is not None:
                self.pbar.update(1)
            return False

        def after_training(self, model):
            if self.pbar is not None:
                self.pbar.close()
                return model
    use_cuda = services.device_policy.mode == 'cuda'
    if use_cuda and services.dependencies.cupy is None:
        raise ContractError('CUDA mode requires an explicitly injected CuPy dependency')
    N_CPU = services.cpu_count or 1
    if use_cuda:
        grid_n_jobs = 1
        xgb_n_jobs = max(1, min(8, N_CPU))
    else:
        grid_n_jobs = max(1, min(4, N_CPU))
        xgb_n_jobs = max(1, N_CPU // max(1, grid_n_jobs))
    services.set_thread_environment('OMP_NUM_THREADS', str(xgb_n_jobs), policy)
    services.set_thread_environment('MKL_NUM_THREADS', str(xgb_n_jobs), policy)
    xgb_kwargs = dict(objective='binary:logistic', eval_metric='aucpr', n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, min_child_weight=1.0, reg_lambda=1.0, scale_pos_weight=1.0, random_state=RANDOM_STATE, n_jobs=xgb_n_jobs, verbosity=0)
    if version.parse(xgb.__version__) >= version.parse('2.0.0'):
        xgb_kwargs['tree_method'] = 'hist'
        xgb_kwargs['max_bin'] = 256
        if use_cuda:
            xgb_kwargs['device'] = 'cuda'
    else:
        xgb_kwargs['tree_method'] = 'gpu_hist' if use_cuda else 'hist'
    xgb_base = XGBClassifier(**xgb_kwargs)

    def build_smote(y_like: np.ndarray, pos_label: int=POS_LABEL, sampling_strategy: float=0.5):
        pos_count = int((y_like == pos_label).sum())
        if pos_count < 2:
            return None
        k = max(1, min(5, pos_count - 1))
        return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=RANDOM_STATE)
    smote_global = build_smote(y_train, pos_label=POS_LABEL, sampling_strategy=0.5)
    pipe = ImbPipeline([('imp', SimpleImputer(strategy='median')), ('smote', smote_global if smote_global is not None else 'passthrough'), ('clf', xgb_base)])
    services.report(f'[CUDA] xgboost={xgb.__version__} | use_cuda={use_cuda} | tree_method={xgb_kwargs.get('tree_method')} | device={xgb_kwargs.get('device', 'cpu')}')
    return SourceResult_0007_0000(
        artifacts=_artifacts(ROUND_DIR, MODEL_PATH, THRESH_JSON_PATH, FI_CSV_PATH, TILE_PREDS_CSV_PATH, IMG_PREDS_CSV_PATH, PR_CURVE_CSV_PATH, ROC_CURVE_CSV_PATH, CONF_MAT_CSV_PATH, METRICS_JSON_PATH, DROPPED_COLS_JSON, CONFIG_JSON_PATH, HISTORY_CSV_PATH, PROGRESS_PNG_PATH),
        continuation=HybridContinuationState(
            CONFIG_JSON_PATH=CONFIG_JSON_PATH,
            CONF_MAT_CSV_PATH=CONF_MAT_CSV_PATH,
            CSV_PATH=CSV_PATH,
            split_manifest=inputs.split_manifest,
            previous_tile_predictions=inputs.previous_tile_predictions,
            previous_threshold=inputs.previous_threshold,
            DROPPED_COLS_JSON=DROPPED_COLS_JSON,
            EARLY_STOP_ROUNDS=EARLY_STOP_ROUNDS,
            EARLY_STOP_VAL_FRAC=EARLY_STOP_VAL_FRAC,
            FI_CSV_PATH=FI_CSV_PATH,
            GROUP_POS_FRAC=GROUP_POS_FRAC,
            HISTORY_CSV_PATH=HISTORY_CSV_PATH,
            IMG_PREDS_CSV_PATH=IMG_PREDS_CSV_PATH,
            METRICS_JSON_PATH=METRICS_JSON_PATH,
            MODEL_NAME=MODEL_NAME,
            MODEL_PATH=MODEL_PATH,
            NAME_COL=NAME_COL,
            N_ESTIMATORS_BIG=N_ESTIMATORS_BIG,
            N_SPLITS_CV=N_SPLITS_CV,
            POS_LABEL=POS_LABEL,
            PROGRESS_PNG_PATH=PROGRESS_PNG_PATH,
            PR_CURVE_CSV_PATH=PR_CURVE_CSV_PATH,
            RANDOM_STATE=RANDOM_STATE,
            ROC_CURVE_CSV_PATH=ROC_CURVE_CSV_PATH,
            ROUND_DIR=ROUND_DIR,
            ROUND_TAG=ROUND_TAG,
            TEST_SIZE=TEST_SIZE,
            THRESH_JSON_PATH=THRESH_JSON_PATH,
            TILE_PREDS_CSV_PATH=TILE_PREDS_CSV_PATH,
            TOPK_FEATURES_TO_PRINT=TOPK_FEATURES_TO_PRINT,
            TQDMCallback=TQDMCallback,
            Timer=Timer,
            USE_REVIEW_WEIGHTING=USE_REVIEW_WEIGHTING,
            X_test=X_test,
            X_train=X_train,
            total_started_at=__T_TOTAL0,
            build_smote=build_smote,
            compute_group_labels=compute_group_labels,
            df=df,
            fit_with_maybe_callbacks=fit_with_maybe_callbacks,
            grid_n_jobs=grid_n_jobs,
            groups_test=groups_test,
            groups_train=groups_train,
            make_weights_after_smote=make_weights_after_smote,
            num_cols=tuple(str(item) for item in num_cols),
            pipe=pipe,
            review_weights_train=review_weights_train,
            smote_global=smote_global,
            test_idx=test_idx,
            to_drop=frozenset(str(item) for item in to_drop),
            train_idx=train_idx,
            xgb_kwargs=MappingProxyType(dict(xgb_kwargs)),
            y_test=y_test,
            y_train=y_train,
        ),
    )

# SOURCE_CELL: NB-LIVE-0007-C0001
# SOURCE_STATEMENT_MAP: sanitized-body -> source_algorithm_0007_0001
@dataclass(frozen=True)
class SourceInputs_0007_0001:
    state: HybridContinuationState

@dataclass(frozen=True)
class SourceResult_0007_0001:
    artifacts: TrainingArtifacts

def source_algorithm_0007_0001(
    inputs: SourceInputs_0007_0001,
    services: SourceAlgorithmServices,
    policy: ExecutionPolicy,
) -> SourceResult_0007_0001:
    require_authorized(policy)
    state: HybridContinuationState = inputs.state
    DummyClassifier = services.dependencies.DummyClassifier
    GroupShuffleSplit = services.dependencies.GroupShuffleSplit
    ImbPipeline = services.dependencies.ImbPipeline
    Pipeline = services.dependencies.Pipeline
    RandomizedSearchCV = services.dependencies.RandomizedSearchCV
    SimpleImputer = services.dependencies.SimpleImputer
    XGBClassifier = services.dependencies.XGBClassifier
    average_precision_score = services.dependencies.average_precision_score
    confusion_matrix = services.dependencies.confusion_matrix
    loguniform = services.dependencies.loguniform
    np = services.dependencies.np
    pd = services.dependencies.pd
    precision_recall_curve = services.dependencies.precision_recall_curve
    randint = services.dependencies.randint
    roc_auc_score = services.dependencies.roc_auc_score
    roc_curve = services.dependencies.roc_curve
    tqdm = services.dependencies.tqdm
    uniform = services.dependencies.uniform
    CONFIG_JSON_PATH = state.CONFIG_JSON_PATH
    CONF_MAT_CSV_PATH = state.CONF_MAT_CSV_PATH
    CSV_PATH = state.CSV_PATH
    DROPPED_COLS_JSON = state.DROPPED_COLS_JSON
    EARLY_STOP_ROUNDS = state.EARLY_STOP_ROUNDS
    EARLY_STOP_VAL_FRAC = state.EARLY_STOP_VAL_FRAC
    FI_CSV_PATH = state.FI_CSV_PATH
    GROUP_POS_FRAC = state.GROUP_POS_FRAC
    HISTORY_CSV_PATH = state.HISTORY_CSV_PATH
    IMG_PREDS_CSV_PATH = state.IMG_PREDS_CSV_PATH
    METRICS_JSON_PATH = state.METRICS_JSON_PATH
    MODEL_NAME = state.MODEL_NAME
    MODEL_PATH = state.MODEL_PATH
    NAME_COL = state.NAME_COL
    N_ESTIMATORS_BIG = state.N_ESTIMATORS_BIG
    N_SPLITS_CV = state.N_SPLITS_CV
    POS_LABEL = state.POS_LABEL
    PROGRESS_PNG_PATH = state.PROGRESS_PNG_PATH
    PR_CURVE_CSV_PATH = state.PR_CURVE_CSV_PATH
    RANDOM_STATE = state.RANDOM_STATE
    ROC_CURVE_CSV_PATH = state.ROC_CURVE_CSV_PATH
    ROUND_DIR = state.ROUND_DIR
    ROUND_TAG = state.ROUND_TAG
    TEST_SIZE = state.TEST_SIZE
    THRESH_JSON_PATH = state.THRESH_JSON_PATH
    TILE_PREDS_CSV_PATH = state.TILE_PREDS_CSV_PATH
    TOPK_FEATURES_TO_PRINT = state.TOPK_FEATURES_TO_PRINT
    TQDMCallback = state.TQDMCallback
    Timer = state.Timer
    USE_REVIEW_WEIGHTING = state.USE_REVIEW_WEIGHTING
    X_test = state.X_test
    X_train = state.X_train
    __T_TOTAL0 = state.total_started_at
    build_smote = state.build_smote
    compute_group_labels = state.compute_group_labels
    df = state.df
    fit_with_maybe_callbacks = state.fit_with_maybe_callbacks
    grid_n_jobs = state.grid_n_jobs
    groups_test = state.groups_test
    groups_train = state.groups_train
    make_weights_after_smote = state.make_weights_after_smote
    num_cols = list(state.num_cols)
    pipe = state.pipe
    review_weights_train = state.review_weights_train
    smote_global = state.smote_global
    test_idx = state.test_idx
    to_drop = set(state.to_drop)
    train_idx = state.train_idx
    xgb_kwargs = dict(state.xgb_kwargs)
    y_test = state.y_test
    y_train = state.y_train
    StratifiedKFold = services.dependencies.StratifiedKFold
    GroupKFold = services.dependencies.GroupKFold
    used_cv = False
    cv_best_pr = None
    best_params: dict[str, object] = {}
    param_distributions = {'clf__max_depth': randint(3, 10), 'clf__min_child_weight': loguniform(0.5, 10.0), 'clf__subsample': uniform(0.7, 0.3), 'clf__colsample_bytree': uniform(0.7, 0.3), 'clf__reg_lambda': loguniform(0.1, 10.0), 'clf__reg_alpha': loguniform(0.001, 1.0), 'clf__gamma': loguniform(0.001, 1.0), 'clf__learning_rate': loguniform(0.02, 0.2), 'clf__max_bin': [256]}
    if smote_global is not None:
        param_distributions.update({'smote__sampling_strategy': [0.4, 0.5, 0.6], 'smote__k_neighbors': randint(2, 6)})
    if len(np.unique(y_train)) < 2:
        services.report('[WARN] Only one class in TRAIN → DummyClassifier.')
        best_model = Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', DummyClassifier(strategy='constant', constant=int(np.unique(y_train)[0])))])
        best_model.fit(X_train, y_train)
    else:
        if len(np.unique(groups_train)) >= 2:
            uniq_groups_cv, labels_cv = compute_group_labels(groups_train, y_train, frac=GROUP_POS_FRAC)
            n_pos_g = int((labels_cv == 1).sum())
            n_neg_g = int((labels_cv == 0).sum())
            services.report(f'[CV] Groups: total={len(uniq_groups_cv)} | pos_groups={n_pos_g} | neg_groups={n_neg_g}')

            def indices_from_groups(g_tr_set, g_va_set):
                tr_idx = np.where(np.isin(groups_train, list(g_tr_set)))[0]
                va_idx = np.where(np.isin(groups_train, list(g_va_set)))[0]
                return (tr_idx, va_idx)
            cv_pairs = []
            used_val_group_signatures = set()
            max_tries = 400
            if len(np.unique(labels_cv)) == 2:
                for target_k in [5, 4, 3, 2]:
                    tries = 0
                    with tqdm(total=max_tries, desc=f'Build stratified folds (k={target_k})', leave=False) as pbar:
                        while len(cv_pairs) < target_k and tries < max_tries:
                            skf = StratifiedKFold(n_splits=target_k, shuffle=True, random_state=RANDOM_STATE + tries)
                            for gi_tr, gi_va in skf.split(uniq_groups_cv, labels_cv):
                                g_tr = set(uniq_groups_cv[gi_tr])
                                g_va = set(uniq_groups_cv[gi_va])
                                tr_idx, va_idx = indices_from_groups(g_tr, g_va)
                                if len(np.unique(y_train[va_idx])) < 2 or len(np.unique(y_train[tr_idx])) < 2:
                                    continue
                                sig = tuple(sorted(g_va))
                                if sig in used_val_group_signatures:
                                    continue
                                used_val_group_signatures.add(sig)
                                cv_pairs.append((tr_idx, va_idx))
                                if len(cv_pairs) >= target_k:
                                    break
                            tries += 1
                            pbar.update(1)
                    if len(cv_pairs) >= 2:
                        services.report(f'[INFO] Using {len(cv_pairs)} CV folds (stratified target={target_k}).')
                        break
            else:
                for target_k in [5, 4, 3, 2]:
                    if target_k > len(np.unique(groups_train)):
                        continue
                    gkf = GroupKFold(n_splits=target_k)
                    tmp_pairs = []
                    for tr_idx, va_idx in gkf.split(X_train, y_train, groups_train):
                        if len(np.unique(y_train[va_idx])) < 2 or len(np.unique(y_train[tr_idx])) < 2:
                            continue
                        tmp_pairs.append((tr_idx, va_idx))
                    if len(tmp_pairs) >= 2:
                        cv_pairs = tmp_pairs[:target_k]
                        services.report(f'[INFO] Using {len(cv_pairs)} CV folds from GroupKFold(k={target_k}).')
                        break
            if len(cv_pairs) < 2:
                rs = np.random.RandomState(RANDOM_STATE)
                perm = rs.permutation(np.unique(groups_train))
                cut = max(1, int(0.2 * len(perm)))
                g_va = set(perm[:cut])
                g_tr = set(perm[cut:])
                tr_idx = np.where(np.isin(groups_train, list(g_tr)))[0]
                va_idx = np.where(np.isin(groups_train, list(g_va)))[0]
                if len(np.unique(y_train[va_idx])) < 2 or len(np.unique(y_train[tr_idx])) < 2:
                    tr_idx, va_idx = (va_idx, tr_idx)
                cv_pairs = [(tr_idx, va_idx)]
                services.report('[WARN] Could not form ≥2 valid folds; using 1 group split (≈80/20).')
            N_ITER_RANDOM = 50
            gs = RandomizedSearchCV(estimator=pipe, param_distributions=param_distributions, n_iter=N_ITER_RANDOM, scoring='average_precision', cv=cv_pairs, n_jobs=grid_n_jobs, verbose=2, error_score=np.nan, refit=False, random_state=RANDOM_STATE)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=UserWarning)
                    with Timer('RandomizedSearchCV.fit'):
                        gs.fit(X_train, y_train)
                best_params = gs.best_params_
                services.report('Best PR-AUC (cv):', gs.best_score_)
                services.report('Best params:', best_params)
                cv_best_pr = float(gs.best_score_)
                used_cv = True
            except Exception as e:
                services.report(f'[WARN] RandomizedSearch failed ({e}). Falling back to base params.')
                best_params = {}
        else:
            services.report('[WARN] Not enough groups for GroupKFold. Skipping CV.')
            best_params = {}

        def refit_with_es_return_pipeline(base_params: dict[str, object], X_tr: pd.DataFrame, y_tr: np.ndarray, groups_tr: np.ndarray, review_weights_tr: np.ndarray) -> ImbPipeline:
            if len(np.unique(groups_tr)) >= 2 and len(y_tr) >= 50:
                gss_inner = GroupShuffleSplit(n_splits=1, test_size=EARLY_STOP_VAL_FRAC, random_state=RANDOM_STATE + 123)
                tr_idx, val_idx = next(gss_inner.split(X_tr, y_tr, groups_tr))
                if len(np.unique(y_tr[val_idx])) < 2:
                    idx = np.arange(len(X_tr))
                    rs = np.random.RandomState(RANDOM_STATE + 123)
                    rs.shuffle(idx)
                    cut = max(1, int(len(idx) * EARLY_STOP_VAL_FRAC))
                    val_idx, tr_idx = (idx[:cut], idx[cut:])
            else:
                idx = np.arange(len(X_tr))
                rs = np.random.RandomState(RANDOM_STATE + 123)
                rs.shuffle(idx)
                cut = max(1, int(len(idx) * EARLY_STOP_VAL_FRAC))
                val_idx, tr_idx = (idx[:cut], idx[cut:])
            X_tr2, y_tr2 = (X_tr.iloc[tr_idx], y_tr[tr_idx])
            X_val, y_val = (X_tr.iloc[val_idx], y_tr[val_idx])
            w_tr2_review = review_weights_tr[tr_idx].astype(np.float32)
            imp = SimpleImputer(strategy='median')
            X_tr2_imp = imp.fit_transform(X_tr2)
            X_val_imp = imp.transform(X_val)
            smote_local = build_smote(y_tr2, pos_label=POS_LABEL, sampling_strategy=0.5)
            if smote_local is not None:
                X_tr2_fit, y_tr2_fit = smote_local.fit_resample(X_tr2_imp, y_tr2)
                w_tr2_fit = make_weights_after_smote(w_tr2_review, y_tr2, y_tr2_fit)
            else:
                X_tr2_fit, y_tr2_fit = (X_tr2_imp, y_tr2)
                w_tr2_fit = w_tr2_review
            xgb_kwargs_local = xgb_kwargs.copy()
            xgb_kwargs_local['scale_pos_weight'] = 1.0
            clf = XGBClassifier(**xgb_kwargs_local)
            if base_params:
                clf.set_params(**{k.replace('clf__', ''): v for k, v in base_params.items()})
            clf.set_params(n_estimators=N_ESTIMATORS_BIG, early_stopping_rounds=EARLY_STOP_ROUNDS)
            try:
                with Timer('Refit with Early-Stopping (inner split)'):
                    fit_with_maybe_callbacks(clf, X_tr2_fit, y_tr2_fit, eval_set=[(X_val_imp, y_val)], verbose=False, callbacks=[xgb.callback.EvaluationMonitor(period=25)], sample_weight=w_tr2_fit)
                used = getattr(clf, 'best_iteration', None)
                if used is None:
                    used = getattr(clf, 'best_ntree_limit', None)
                services.report(f'[ES] early stopping applied; effective trees ≈ {used}')
            except Exception as e:
                services.report(f'[WARN] ES fit failed ({e}); fitting WITHOUT ES.')
                clf.set_params(early_stopping_rounds=None)
                with Timer('Refit WITHOUT ES (inner split)'):
                    fit_with_maybe_callbacks(clf, X_tr2_fit, y_tr2_fit, callbacks=[TQDMCallback(total=clf.get_params().get('n_estimators', 300), desc='XGB[No-ES]')], sample_weight=w_tr2_fit)
            final_steps = [('imp', imp)]
            final_steps.append(('smote', smote_local if smote_local is not None else 'passthrough'))
            final_steps.append(('clf', clf))
            return ImbPipeline(final_steps)
        with Timer('Assemble best_model via ES refit'):
            best_model = refit_with_es_return_pipeline(best_params, X_train, y_train, groups_train, review_weights_tr=review_weights_train)
    clf = best_model.named_steps.get('clf', None)
    if clf is not None:
        used_trees = getattr(clf, 'best_iteration', None)
        used_trees = int(used_trees) + 1 if used_trees is not None else getattr(clf, 'best_ntree_limit', None)
        if used_trees is not None:
            try:
                best_model.set_output(transform='pandas')
            except Exception:
                services.report('[REFIT] pandas output configuration unavailable; continuing with estimator defaults.')
            best_model.set_params(clf__early_stopping_rounds=None, clf__callbacks=None, clf__n_estimators=int(used_trees), clf__scale_pos_weight=1.0)
            services.report(f'[REFIT] Re-training on FULL TRAIN with n_estimators={int(used_trees)}')
            with Timer('Full-train refit'):
                clf_final = best_model.named_steps['clf']
                imp_final = best_model.named_steps['imp']
                smote_final = best_model.named_steps.get('smote', None)
                X_train_imp = imp_final.fit_transform(X_train)
                if smote_final is not None and smote_final != 'passthrough':
                    smote_final = build_smote(y_train, pos_label=POS_LABEL, sampling_strategy=0.5)
                    if smote_final is not None:
                        X_train_fit, y_train_fit = smote_final.fit_resample(X_train_imp, y_train)
                        sample_weight_full_fit = make_weights_after_smote(review_weights_train, y_train, y_train_fit)
                    else:
                        X_train_fit, y_train_fit = (X_train_imp, y_train)
                        sample_weight_full_fit = review_weights_train
                else:
                    X_train_fit, y_train_fit = (X_train_imp, y_train)
                    sample_weight_full_fit = review_weights_train
                fit_with_maybe_callbacks(clf_final, X_train_fit, y_train_fit, callbacks=[xgb.callback.EvaluationMonitor(period=25)], sample_weight=sample_weight_full_fit)
                steps = [('imp', imp_final)]
                steps.append(('smote', smote_final if smote_final is not None else 'passthrough'))
                steps.append(('clf', clf_final))
                best_model = ImbPipeline(steps)
        else:
            services.report('[REFIT] Could not detect best_iteration/best_ntree_limit; skipping full-train refit.')
    try:
        clf_fitted = best_model.named_steps.get('clf', None)
        params_now = getattr(clf_fitted, 'get_xgb_params', lambda: {})()
        services.report(f'[CUDA] Final booster params → tree_method={params_now.get('tree_method')} | device={params_now.get('device')}')
    except Exception as e:
        services.report(f'[CUDA] Could not retrieve booster params: {e}')

    def pick_threshold_by_pr(y_true, proba):
        prec, rec, thr = precision_recall_curve(y_true, proba)
        if len(thr) == 0:
            return (0.5, prec, rec, thr)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_i = int(np.argmax(f1s))
        return (float(thr[max(best_i - 1, 0)]), prec, rec, thr)

    def predict_proba_batched(model, Xdf: pd.DataFrame, batch: int=50000, desc: str='Predict proba') -> np.ndarray:
        n = len(Xdf)
        out = np.empty(n, dtype=np.float32)
        for i in tqdm(range(0, n, batch), desc=desc, leave=False):
            j = min(i + batch, n)
            out[i:j] = model.predict_proba(Xdf.iloc[i:j])[:, 1]
        return out
    tile_metrics: dict[str, object] = {}
    img_metrics: dict[str, object] = {}
    thr_use = 0.5
    split_used = 'test' if len(test_idx) > 0 and len(np.unique(y_test)) >= 2 else 'train'
    if split_used == 'test':
        with Timer('Predict proba on TEST'):
            proba_test = predict_proba_batched(best_model, X_test, desc='Predict TEST')
        roc = roc_auc_score(y_test, proba_test)
        pr = average_precision_score(y_test, proba_test)
        thr_use, prec, rec, thr = pick_threshold_by_pr(y_test, proba_test)
        y_pred = (proba_test >= thr_use).astype(int)
        services.report(f'Tile-level: ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}')
        services.report(f'Best F1 via PR on TEST at threshold≈{thr_use:.3f}')
        cm = confusion_matrix(y_test, y_pred)
        test_df = df.iloc[test_idx].copy()
        test_df['__group'] = groups_test
        test_df['__proba'] = proba_test
        test_df['__y'] = y_test
        img_agg = test_df.groupby('__group').agg(y_img=('__y', 'max'), p_img=('__proba', 'max')).reset_index()
        try:
            roc_img = roc_auc_score(img_agg['y_img'], img_agg['p_img'])
        except Exception:
            roc_img = float('nan')
        pr_img = average_precision_score(img_agg['y_img'], img_agg['p_img'])
        services.report(f'Image-level: ROC-AUC={roc_img:.4f}  PR-AUC={pr_img:.4f}')
        tile_metrics = {'roc_auc_test': float(roc), 'pr_auc_test': float(pr)}
        img_metrics = {'roc_auc_test': float(roc_img), 'pr_auc_test': float(pr_img)}
        tmp = df.iloc[test_idx][[NAME_COL]].copy()
        tmp['group'] = groups_test
        tmp['y_true'] = y_test
        tmp['proba'] = proba_test
        tmp.to_csv(TILE_PREDS_CSV_PATH, index=False)
        img_agg.rename(columns={'__group': 'group', 'y_img': 'y_true', 'p_img': 'proba'}, inplace=True)
        img_agg.to_csv(IMG_PREDS_CSV_PATH, index=False)
        pr_df = pd.DataFrame({'precision': prec[:-1], 'recall': rec[:-1], 'threshold': thr})
        pr_df.to_csv(PR_CURVE_CSV_PATH, index=False)
        fpr, tpr, roc_thr = roc_curve(y_test, proba_test)
        pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': roc_thr}).to_csv(ROC_CURVE_CSV_PATH, index=False)
        pd.DataFrame(cm, index=['true_0', 'true_1'], columns=['pred_0', 'pred_1']).to_csv(CONF_MAT_CSV_PATH)
    else:
        services.report('[WARN] No independent TEST split; selecting threshold on TRAIN (optimistic).')
        with Timer('Predict proba on TRAIN'):
            proba_tr = predict_proba_batched(best_model, X_train, desc='Predict TRAIN')
        thr_use, prec, rec, thr = pick_threshold_by_pr(y_train, proba_tr)
        try:
            roc_tr = roc_auc_score(y_train, proba_tr) if len(np.unique(y_train)) >= 2 else float('nan')
            pr_tr = average_precision_score(y_train, proba_tr)
            services.report(f'(TRAIN diag) Tile-level: ROC-AUC={roc_tr:.4f}  PR-AUC={pr_tr:.4f}')
            tile_metrics = {'roc_auc_train': float(roc_tr), 'pr_auc_train': float(pr_tr)}
        except Exception:
            tile_metrics = {'roc_auc_train': None, 'pr_auc_train': None}
            services.report('[WARN] TRAIN diagnostic metrics were unavailable.')
        tmp = df.iloc[train_idx][[NAME_COL]].copy()
        tmp['group'] = groups_train
        tmp['y_true'] = y_train
        tmp['proba'] = proba_tr
        tmp.to_csv(TILE_PREDS_CSV_PATH, index=False)
        pd.DataFrame({'precision': prec[:-1], 'recall': rec[:-1], 'threshold': thr}).to_csv(PR_CURVE_CSV_PATH, index=False)
        fpr, tpr, roc_thr = roc_curve(y_train, proba_tr)
        pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': roc_thr}).to_csv(ROC_CURVE_CSV_PATH, index=False)
    joblib = services.dependencies.joblib
    xgb = services.dependencies.xgb
    meta = {'csv_path': CSV_PATH, 'round_tag': ROUND_TAG, 'pos_label': int(POS_LABEL), 'xgboost_version': xgb.__version__, 'use_cuda_try': bool(services.device_policy.mode == 'cuda'), 'used_cv': bool(used_cv), 'cv_best_pr_auc': None if cv_best_pr is None else float(cv_best_pr), 'tile_metrics': tile_metrics, 'img_metrics': img_metrics, 'class_balance_train': np.bincount(y_train).tolist(), 'class_balance_test': np.bincount(y_test).tolist() if len(test_idx) > 0 else None, 'split_used_for_threshold': split_used, 'test_size': float(TEST_SIZE), 'random_state': int(RANDOM_STATE), 'features_count': len(num_cols), 'early_stopping_rounds': int(EARLY_STOP_ROUNDS), 'early_stop_val_frac': float(EARLY_STOP_VAL_FRAC), 'best_iteration': int(getattr(getattr(best_model, 'named_steps', {}).get('clf', object()), 'best_iteration', -1)) if hasattr(best_model, 'named_steps') else -1, 'hybrid_mode': True, 'review_weight_mean': float(np.mean(review_weights_train)) if USE_REVIEW_WEIGHTING else None}
    clf_for_meta = best_model.named_steps.get('clf', None)
    if clf_for_meta is not None:
        meta['used_trees'] = int(getattr(clf_for_meta, 'n_estimators', 0))
        bi = getattr(clf_for_meta, 'best_iteration', None)
        if bi is not None:
            meta['es_best_iteration'] = int(bi)
        meta['early_stopping_rounds'] = int(EARLY_STOP_ROUNDS)
    joblib.dump({'pipeline': best_model, 'features': num_cols, 'threshold': float(thr_use), 'meta': meta}, MODEL_PATH)
    with open(THRESH_JSON_PATH, 'w') as f:
        json.dump({'threshold': float(thr_use)}, f, indent=2)
    with open(DROPPED_COLS_JSON, 'w') as f:
        json.dump(sorted(list(to_drop)), f, indent=2)
    with open(CONFIG_JSON_PATH, 'w') as f:
        json.dump({'MODEL_NAME': MODEL_NAME, 'ROUND_DIR': str(ROUND_DIR), 'ROUND_TAG': ROUND_TAG, 'TEST_SIZE': TEST_SIZE, 'RANDOM_STATE': RANDOM_STATE, 'N_SPLITS_CV': N_SPLITS_CV, 'POS_LABEL': POS_LABEL, 'TOPK_FEATURES_TO_PRINT': TOPK_FEATURES_TO_PRINT, 'used_features': num_cols, 'split_mode': 'SealedManifest(GroupPure)', 'split_manifest': str(state.split_manifest.path), 'split_manifest_sha256': state.split_manifest.sha256, 'previous_tile_predictions': str(state.previous_tile_predictions.path), 'previous_tile_predictions_sha256': state.previous_tile_predictions.sha256, 'previous_threshold': str(state.previous_threshold.path), 'previous_threshold_sha256': state.previous_threshold.sha256}, f, indent=2)
    services.report(f'[OK] saved model to: {MODEL_PATH}')
    services.report(f'[OK] saved threshold JSON to: {THRESH_JSON_PATH}')
    try:
        clf_fitted = best_model.named_steps.get('clf', None)
        importances = getattr(clf_fitted, 'feature_importances_', None)
        if importances is not None:
            fi_full = pd.Series(importances, index=num_cols).sort_values(ascending=False)
            services.report('\nTop feature importances:')
            services.report(fi_full.head(TOPK_FEATURES_TO_PRINT).round(6).to_string())
            fi_full.to_csv(FI_CSV_PATH)
            services.report(f'[OK] saved full feature importances to: {FI_CSV_PATH}')
    except Exception as e:
        warnings.warn(f'could not compute/save feature importances: {e}')
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump({'round_tag': ROUND_TAG, 'tile_metrics': tile_metrics, 'img_metrics': img_metrics, 'threshold': float(thr_use), 'split_used_for_threshold': split_used}, f, indent=2)
    hist_row = {'round': ROUND_TAG, 'features_csv': CSV_PATH, 'threshold': float(thr_use), 'used_cv': bool(used_cv), 'cv_best_pr_auc': None if cv_best_pr is None else float(cv_best_pr), 'tile_pr_auc': float(tile_metrics.get('pr_auc_test', tile_metrics.get('pr_auc_train', np.nan))), 'tile_roc_auc': float(tile_metrics.get('roc_auc_test', tile_metrics.get('roc_auc_train', np.nan))), 'img_pr_auc': float(img_metrics.get('pr_auc_test', np.nan)), 'img_roc_auc': float(img_metrics.get('roc_auc_test', np.nan)), 'train_pos': int(np.bincount(y_train)[1] if len(np.bincount(y_train)) > 1 else 0), 'train_neg': int(np.bincount(y_train)[0] if len(np.bincount(y_train)) > 0 else 0), 'n_tiles_train': int(len(y_train)), 'n_tiles_test': int(len(y_test)), 'n_groups_train': int(len(np.unique(groups_train))), 'n_groups_test': int(len(np.unique(groups_test))) if len(test_idx) > 0 else 0}
    if HISTORY_CSV_PATH.exists():
        hist = pd.read_csv(HISTORY_CSV_PATH)
        hist = hist[hist['round'] != ROUND_TAG]
        hist = pd.concat([hist, pd.DataFrame([hist_row])], ignore_index=True)
    else:
        hist = pd.DataFrame([hist_row])
    hist = hist.sort_values(by='round', key=lambda s: s.str.extract('r(\\d+)', expand=False).astype(float), ignore_index=True)
    hist.to_csv(HISTORY_CSV_PATH, index=False)
    services.report(f'[OK] history updated: {HISTORY_CSV_PATH}')
    try:
        plt = services.dependencies.plt
        h = pd.read_csv(HISTORY_CSV_PATH)
        h['_rnum'] = h['round'].str.extract('r(\\d+)', expand=False).astype(float)
        plt.figure()
        if 'tile_pr_auc' in h.columns:
            plt.plot(h['_rnum'].values, h['tile_pr_auc'].values, marker='o', label='Tile PR-AUC')
        if 'img_pr_auc' in h.columns and h['img_pr_auc'].notna().any():
            plt.plot(h['_rnum'].values, h['img_pr_auc'].values, marker='o', linestyle='--', label='Image PR-AUC')
        plt.xlabel('Round')
        plt.ylabel('PR-AUC')
        plt.title('Active Learning Progress (PR-AUC)')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PROGRESS_PNG_PATH, dpi=150)
        plt.close()
        services.report(f'[OK] saved progress plot: {PROGRESS_PNG_PATH}')
    except Exception as e:
        services.report(f'[WARN] could not render progress plot: {e}')
    services.report(f'[TIMER] TOTAL took {time.perf_counter() - __T_TOTAL0:.2f} sec')
    return SourceResult_0007_0001(_artifacts(ROUND_DIR, MODEL_PATH, THRESH_JSON_PATH, FI_CSV_PATH, TILE_PREDS_CSV_PATH, IMG_PREDS_CSV_PATH, PR_CURVE_CSV_PATH, ROC_CURVE_CSV_PATH, CONF_MAT_CSV_PATH, METRICS_JSON_PATH, DROPPED_COLS_JSON, CONFIG_JSON_PATH, HISTORY_CSV_PATH, PROGRESS_PNG_PATH))

# SOURCE_CELL: NB-LIVE-0007-C0006
# SOURCE_STATEMENT_MAP: sanitized-body -> source_algorithm_0007_0006
@dataclass(frozen=True)
class SourceInputs_0007_0006:
    output_root: Path
    training_features: SourceInputArtifact
    split_manifest: SourceInputArtifact
    previous_tile_predictions: SourceInputArtifact
    previous_threshold: SourceInputArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path) or not self.output_root.is_absolute():
            raise ContractError("training output root must be absolute")
        if any(
            not isinstance(artifact, SourceInputArtifact)
            for artifact in (
                self.training_features,
                self.split_manifest,
                self.previous_tile_predictions,
                self.previous_threshold,
            )
        ):
            raise ContractError("NB7 inputs must be typed source input artifacts")

@dataclass(frozen=True)
class SourceResult_0007_0006:
    artifacts: TrainingArtifacts
    continuation: AugmentedContinuationState

def source_algorithm_0007_0006(
    inputs: SourceInputs_0007_0006,
    services: SourceAlgorithmServices,
    policy: ExecutionPolicy,
) -> SourceResult_0007_0006:
    require_authorized(policy)
    _training_features = services.preflight_input_artifact(inputs.training_features, policy)
    _split_manifest = services.preflight_input_artifact(inputs.split_manifest, policy)
    _manifest_train, _manifest_validation, _manifest_test = _read_split_groups(
        _split_manifest
    )
    _verified_previous_artifacts = (
        services.preflight_input_artifact(inputs.previous_tile_predictions, policy),
        services.preflight_input_artifact(inputs.previous_threshold, policy),
    )
    _previous_tile_predictions = _verified_previous_artifacts[0]
    _previous_threshold = _verified_previous_artifacts[1]
    PREV_TILE_PREDS_PATH = str(_previous_tile_predictions)
    PREV_THRESH_JSON_PATH = str(_previous_threshold)
    import os, re, json, warnings, time
    from pathlib import Path
    np = services.dependencies.np
    pd = services.dependencies.pd
    loguniform = services.dependencies.loguniform
    randint = services.dependencies.randint
    uniform = services.dependencies.uniform
    GroupShuffleSplit = services.dependencies.GroupShuffleSplit
    GroupKFold = services.dependencies.GroupKFold
    StratifiedShuffleSplit = services.dependencies.StratifiedShuffleSplit
    RandomizedSearchCV = services.dependencies.RandomizedSearchCV
    roc_auc_score = services.dependencies.roc_auc_score
    average_precision_score = services.dependencies.average_precision_score
    precision_recall_curve = services.dependencies.precision_recall_curve
    roc_curve = services.dependencies.roc_curve
    confusion_matrix = services.dependencies.confusion_matrix
    Pipeline = services.dependencies.Pipeline
    SimpleImputer = services.dependencies.SimpleImputer
    DummyClassifier = services.dependencies.DummyClassifier
    tqdm = services.dependencies.tqdm
    from contextlib import contextmanager
    joblib = services.dependencies.joblib
    SMOTE = services.dependencies.SMOTE
    ImbPipeline = services.dependencies.ImbPipeline
    warnings.filterwarnings('ignore', message='This pattern is interpreted as a regular expression', category=UserWarning)

    @contextmanager
    def tqdm_joblib(tqdm_object):
        """progress for joblib (e.g., Grid/RandomizedSearchCV)"""

        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):

            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)
        old_cb = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_cb
            tqdm_object.close()

    class Timer:

        def __init__(self, label):
            self.label = label

        def __enter__(self):
            self.t0 = time.perf_counter()
            services.report(f'[TIMER] {self.label} ...')
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            services.report(f'[TIMER] {self.label} took {dt:.2f} sec')
    __T_TOTAL0 = time.perf_counter()
    CSV_PATH = str(_training_features)
    OUT_DIR_ROOT = inputs.output_root
    MODEL_NAME = 'cj_ultra_tilesafe_xgb_r16_hybrid.pkl'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_SPLITS_CV = 5
    POS_LABEL = 1
    TOPK_FEATURES_TO_PRINT = 30
    EARLY_STOP_ROUNDS = 50
    EARLY_STOP_VAL_FRAC = 0.3
    N_ESTIMATORS_BIG = 2000
    USE_REVIEW_WEIGHTING = True
    WEIGHT_CORRECT = 0.7
    WEIGHT_WRONG = 1.0
    AUTO_DETECT_PREV_ROUND = False
    AUG_CFG = {'apply_in_3A': True, 'exclude_discrete_max_nunique': 10, 'mixup': {'enabled': True, 'alpha': 0.2, 'mult': 0.5}, 'dropout': {'enabled': True, 'p': 0.05, 'mult': 1.0, 'strategy': 'median_by_class'}, 'jitter': {'enabled': True, 'mult': 0.5, 'sigma': 0.01, 'per_feature': True, 'clip_q': (0.001, 0.999)}, 'shuffle_rows': True}
    services.set_thread_environment('OMP_NUM_THREADS', '1', policy)
    services.set_thread_environment('MKL_NUM_THREADS', '1', policy)
    prepare_source_output_directory(OUT_DIR_ROOT)
    m = re.search('_r(\\d+)\\b', MODEL_NAME)
    ROUND_TAG = f'r{m.group(1)}' if m else 'r1'
    m2 = re.search('xgb_(.+?)\\.pkl$', os.path.basename(MODEL_NAME))
    MODEL_DIR_TAG = m2.group(1) if m2 else ROUND_TAG
    ROUND_DIR = OUT_DIR_ROOT / MODEL_DIR_TAG
    prepare_source_output_directory(ROUND_DIR)
    MODEL_PATH = ROUND_DIR / MODEL_NAME
    THRESH_JSON_PATH = ROUND_DIR / f'threshold_{ROUND_TAG}.json'
    FI_CSV_PATH = ROUND_DIR / 'feature_importances.csv'
    TILE_PREDS_CSV_PATH = ROUND_DIR / 'tile_preds.csv'
    IMG_PREDS_CSV_PATH = ROUND_DIR / 'image_preds.csv'
    PR_CURVE_CSV_PATH = ROUND_DIR / 'pr_curve.csv'
    ROC_CURVE_CSV_PATH = ROUND_DIR / 'roc_curve.csv'
    METRICS_JSON_PATH = ROUND_DIR / 'metrics.json'
    CONF_MAT_CSV_PATH = ROUND_DIR / 'confusion_matrix.csv'
    DROPPED_COLS_JSON = ROUND_DIR / 'dropped_cols.json'
    CONFIG_JSON_PATH = ROUND_DIR / 'config.json'
    HISTORY_CSV_PATH = OUT_DIR_ROOT / 'history.csv'
    PROGRESS_PNG_PATH = OUT_DIR_ROOT / 'progress.png'
    GROUP_POS_FRAC = 0.5

    def compute_group_labels(groups_vec, y_vec, frac=GROUP_POS_FRAC):
        gstat = pd.DataFrame({'g': groups_vec, 'y': y_vec}).groupby('g')['y'].agg(['max', 'mean'])
        labels_any = (gstat['max'] > 0).astype(int)
        if labels_any.nunique() == 1:
            return (gstat.index.to_numpy(), labels_any.to_numpy())
        return (gstat.index.to_numpy(), labels_any.to_numpy())
    import inspect

    def fit_with_maybe_callbacks(model, X, y, **kw):
        sig = inspect.signature(model.fit)
        if 'callbacks' not in sig.parameters and 'callbacks' in kw:
            kw.pop('callbacks', None)
        return model.fit(X, y, **kw)

    def leakage_scan_vectorized(df: pd.DataFrame, y: np.ndarray, drop_always: set, leak_patterns: list[str], corr_thr: float=0.999):
        num_df = df.drop(columns=list(drop_always), errors='ignore')
        num_df = num_df.select_dtypes(include=[np.number]).astype(np.float32)
        pat = '|'.join((f'(?:{p})' for p in leak_patterns))
        name_hit = num_df.columns.to_series().str.contains(pat, case=False, regex=True, na=False)
        drop_by_name = set(num_df.columns[name_hit])
        num_df_f = num_df.fillna(0.0)
        y_s = pd.Series(y.astype(np.float32), index=num_df_f.index)
        eq_y = num_df_f.eq(y_s, axis=0).all(axis=0)
        eq_inv = num_df_f.eq(1.0 - y_s, axis=0).all(axis=0)
        drop_by_equal = set(num_df_f.columns[eq_y | eq_inv])
        const_mask = num_df_f.nunique(dropna=False) <= 1
        corr = num_df_f.loc[:, ~const_mask].corrwith(y_s, method='pearson')
        corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        drop_by_corr = set(corr.index[corr.abs() > float(corr_thr)])
        to_drop = drop_by_name | drop_by_equal | drop_by_corr
        keep_cols = [c for c in num_df.columns if c not in to_drop and (not bool(const_mask.get(c, False)))]
        return (keep_cols, sorted(to_drop))
    df = pd.read_csv(CSV_PATH)

    def pick_name_col(df: pd.DataFrame) -> str:
        for c in ['image', 'file_name', 'filename', 'file']:
            if c in df.columns:
                return c
        raise ValueError("CSV must contain one of: 'image', 'file_name', 'filename', 'file'.")
    NAME_COL = pick_name_col(df)

    def original_image_group(name: str) -> str:
        base = Path(str(name)).name
        stem = Path(base).stem
        for pat in ['^(?P<root>.+?)_y\\d{1,8}x\\d{1,8}$', '^(?P<root>.+?)_x\\d{1,8}_y\\d{1,8}$', '^(?P<root>.+?)(?:__tile-\\d+|-tile-\\d+)$']:
            m = re.match(pat, stem, flags=re.IGNORECASE)
            if m:
                return m.group('root')
        return stem
    groups = df[NAME_COL].astype(str).map(original_image_group).values
    y = df['label'].astype(int).values
    drop_always = {NAME_COL, 'image', 'file_name', 'filename', 'file', 'id', 'ann_id', 'label', 'group', 'area_px', 'perim_sqrt', 'reviewed'}
    LEAK_PATTERNS = ['^class(_?id)?$', '^category(_?id)?$', '^cat(_?id)?$', '^cid$', '^name$', '^target$', '^y$', '.*(_|^)class(_|$).*', '.*(_|^)category(_|$).*', '.*(_|^)catname(_|$).*', '.*(_|^)classname(_|$).*']
    num_cols, dropped_cols = leakage_scan_vectorized(df, y, drop_always, LEAK_PATTERNS, corr_thr=0.999)
    if dropped_cols:
        services.report('[INFO] dropping potential leakage columns:', dropped_cols)
    MUST_KEEP = ['pred_iou', 'stability', 'embed_sim', 'g_quality', 'g_light', 'g_color', 'g_shape', 'g_embed', 'g_robust', 'g_maha', 'g_border']
    PCA_KEEP = [c for c in df.columns if c.startswith('embed_pca_')]

    def _safe_promote(cols, df, y):
        ys = pd.Series(pd.to_numeric(y, errors='coerce'), index=df.index).astype(float).fillna(0.0)
        keep = []
        for c in cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            if s.nunique(dropna=False) <= 1:
                continue
            if s.equals(ys) or s.equals(1.0 - ys):
                continue
            r = np.corrcoef(s.to_numpy(), ys.to_numpy())[0, 1]
            if np.isfinite(r) and abs(r) > 0.999:
                continue
            keep.append(c)
        return keep
    keepers = _safe_promote(MUST_KEEP + PCA_KEEP, df, y)
    num_cols = sorted(set(num_cols).union(keepers))
    services.report(f'[FEATURES] final columns after union with gates/PCA: {len(num_cols)}')
    services.report('[FEATURES] gates in use:', [c for c in MUST_KEEP if c in num_cols])
    if not num_cols:
        raise RuntimeError('No usable numeric features after leakage/variance filters.')
    X = df[num_cols].copy().astype(np.float32)

    def pick_groups_tile_balanced(groups_vec: np.ndarray, test_frac: float=0.2, random_state: int=42, tol: float=0.01, max_tries: int=2000):
        uniq, inv = np.unique(groups_vec, return_inverse=True)
        counts = np.bincount(inv).astype(int)
        total = int(counts.sum())
        target = int(round(total * test_frac))
        rs = np.random.RandomState(random_state)
        best_sel, best_sum, best_diff = (None, 0, float('inf'))

        def greedy_from(order_idx):
            s = 0
            sel = set()
            for idx in order_idx:
                c = int(counts[idx])
                if abs(s + c - target) <= abs(s - target):
                    sel.add(uniq[idx])
                    s += c
            return (sel, s)
        orders = [np.argsort(-counts), np.argsort(counts)]
        for t in range(max_tries):
            order = orders[t] if t < len(orders) else rs.permutation(len(uniq))
            sel, s = greedy_from(order)
            diff = abs(s - target)
            if diff < best_diff:
                best_sel, best_sum, best_diff = (sel, s, diff)
                if diff <= max(1, int(target * tol)):
                    break
        test_groups = set(best_sel)
        train_groups = set(uniq) - test_groups
        return (train_groups, test_groups, dict(zip(uniq, counts)), target, best_sum)
    g_tr_set, g_te_set = _resolve_manifest_groups(
        set(groups.tolist()),
        _manifest_train,
        _manifest_validation,
        _manifest_test,
    )
    split_mode = 'SealedManifest(GroupPure)'
    train_idx = np.where(np.isin(groups, list(g_tr_set)))[0]
    test_idx = np.where(np.isin(groups, list(g_te_set)))[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ContractError('sealed split manifest produced an empty row partition')
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise ContractError('sealed split manifest produced group leakage')
    observed_held_out_ratio = len(test_idx) / (len(train_idx) + len(test_idx))
    if abs(observed_held_out_ratio - TEST_SIZE) > 0.01:
        raise ContractError('sealed split manifest held-out ratio exceeds the allowed tolerance')
    if AUG_CFG.get('shuffle_rows', True):
        rs = np.random.RandomState(RANDOM_STATE)
        train_idx = rs.permutation(train_idx)
        test_idx = rs.permutation(test_idx)
    X_train, X_test = (X.iloc[train_idx], X.iloc[test_idx])
    y_train, y_test = (y[train_idx], y[test_idx])
    groups_train, groups_test = (groups[train_idx], groups[test_idx])
    frac = len(test_idx) / (len(train_idx) + len(test_idx))
    services.report(f'Train size: {len(train_idx)}, Test size: {len(test_idx)} | Split={split_mode} | tiles_test_frac={frac:.3f} | target={TEST_SIZE:.3f}')

    def _find_prev_round_dir(root: Path, current_round_tag: str) -> Path | None:
        raise ContractError('automatic previous-round discovery is disabled; explicit verified artifacts are required')

    def _load_prev_preds_and_thr(prev_preds_path: Path, prev_thr_path: Path, name_col: str) -> dict[str, int] | None:
        try:
            prev = pd.read_csv(prev_preds_path)
            thr = json.load(open(prev_thr_path))['threshold']
            if name_col not in prev.columns or 'proba' not in prev.columns:
                return None
            prev = prev[[name_col, 'proba']].copy()
            prev['pred'] = (prev['proba'] >= float(thr)).astype(int)
            return dict(zip(prev[name_col].astype(str), prev['pred'].astype(int)))
        except Exception:
            return None

    def build_review_weights_for_train(df: pd.DataFrame, name_col: str, train_idx: np.ndarray, y_train: np.ndarray, round_dir_root: Path, current_round_tag: str, weight_correct: float=0.7, weight_wrong: float=1.0) -> np.ndarray:
        prev_preds_path = Path(PREV_TILE_PREDS_PATH)
        prev_thr_path = Path(PREV_THRESH_JSON_PATH)
        mapping = None
        if prev_preds_path is not None and prev_thr_path is not None:
            mapping = _load_prev_preds_and_thr(prev_preds_path, prev_thr_path, name_col)
        names_train = df.iloc[train_idx][name_col].astype(str).values
        w = np.full(len(train_idx), weight_wrong, dtype=np.float32)
        if mapping:
            preds_prev = np.array([mapping.get(n, -1) for n in names_train], dtype=int)
            known_mask = preds_prev >= 0
            if known_mask.any():
                correct = preds_prev[known_mask] == y_train[known_mask]
                w_sub = np.where(correct, weight_correct, weight_wrong).astype(np.float32)
                w[known_mask] = w_sub
        return w

    def make_weights_after_smote(original_weights: np.ndarray, y_original: np.ndarray, y_resampled: np.ndarray) -> np.ndarray:
        n_orig = len(y_original)
        n_res = len(y_resampled)
        w = np.empty(n_res, dtype=np.float32)
        w[:n_orig] = original_weights[:n_orig]
        pos_mask = y_original == 1
        w_pos_mean = float(original_weights[pos_mask].mean()) if pos_mask.any() else float(original_weights.mean())
        if n_res > n_orig:
            w[n_orig:] = w_pos_mean
        return w
    review_weights_train = np.ones_like(y_train, dtype=np.float32)
    if USE_REVIEW_WEIGHTING:
        review_weights_train = build_review_weights_for_train(df=df, name_col=NAME_COL, train_idx=train_idx, y_train=y_train, round_dir_root=OUT_DIR_ROOT, current_round_tag=ROUND_TAG, weight_correct=WEIGHT_CORRECT, weight_wrong=WEIGHT_WRONG)
        services.report(f'[WEIGHTS] review-weights built. mean={review_weights_train.mean():.3f}')
        review_mask_train = df.iloc[train_idx].get('reviewed', 0).astype(int).values == 1
        review_weights_train = np.where(review_mask_train, review_weights_train, 1.0).astype(np.float32)
        services.report(f'[WEIGHTS] using review-only mask. frac_reviewed={review_mask_train.mean():.3f}')

    def _cont_indices_from_df(df_like: pd.DataFrame, max_nunique: int) -> np.ndarray:
        nunq = df_like.nunique().to_numpy()
        return np.where(nunq > int(max_nunique))[0]

    def mixup_same_class_with_groups(X, y, w, g, alpha=0.2, mult=0.5, exclude_discrete_max_nunique=10, rs=None):
        if rs is None:
            rs = np.random.RandomState(42)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        w = np.asarray(w, dtype=np.float32)
        g = np.asarray(g)
        nunq = pd.DataFrame(X).nunique().to_numpy()
        cont_idx = np.where(nunq > int(exclude_discrete_max_nunique))[0]
        if cont_idx.size == 0:
            return (X, y, w, g)
        Xc_list, yc_list, wc_list, gc_list = ([X], [y], [w], [g])
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            n = len(idx)
            if n < 2:
                continue
            m = int(np.ceil(n * float(mult)))
            i1 = rs.choice(idx, size=m, replace=n < m)
            i2 = rs.choice(idx, size=m, replace=n < m)
            lam = rs.beta(alpha, alpha, size=m).astype(np.float32)
            Xn = X[i1].copy()
            diff = (X[i2] - X[i1]).astype(np.float32)
            Xn[:, cont_idx] = (X[i1][:, cont_idx] + lam[:, None] * diff[:, cont_idx]).astype(np.float32)
            yn = np.full(m, cls, dtype=int)
            wn = 0.5 * (w[i1] + w[i2])
            gn = np.where(rs.rand(m) < 0.5, g[i1], g[i2])
            Xc_list.append(Xn)
            yc_list.append(yn)
            wc_list.append(wn)
            gc_list.append(gn)
        Xo = np.vstack(Xc_list).astype(np.float32)
        yo = np.concatenate(yc_list).astype(int)
        wo = np.concatenate(wc_list).astype(np.float32)
        go = np.concatenate(gc_list)
        return (Xo, yo, wo, go)

    def dropout_augment_tabular_with_groups(X, y, w, g, p=0.05, mult=1.0, strategy='median_by_class', exclude_discrete_max_nunique=10, rs=None):
        if rs is None:
            rs = np.random.RandomState(42)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        w = np.asarray(w, dtype=np.float32)
        g = np.asarray(g)
        n, d = X.shape
        nunq = pd.DataFrame(X).nunique().to_numpy()
        cont_idx = np.where(nunq > int(exclude_discrete_max_nunique))[0]
        if cont_idx.size == 0:
            return (X, y, w, g)
        m = int(np.ceil(n * float(mult)))
        sel = rs.choice(np.arange(n), size=m, replace=n < m)
        Xn = X[sel].copy()
        yn = y[sel].copy()
        wn = w[sel].copy()
        gn = g[sel].copy()
        if strategy.startswith('median'):
            if 'class' in strategy and len(np.unique(y)) >= 2:
                med = {int(c): np.median(X[y == c][:, cont_idx], axis=0) for c in np.unique(y)}
                fills = np.vstack([med[int(c)] for c in yn])
            else:
                med_all = np.median(X[:, cont_idx], axis=0)
                fills = np.repeat(med_all[None, :], m, axis=0)
        else:
            fills = np.zeros((m, cont_idx.size), dtype=np.float32)
        mask = rs.rand(m, cont_idx.size) < float(p)
        Xn[:, cont_idx] = np.where(mask, fills, Xn[:, cont_idx])
        Xo = np.vstack([X, Xn]).astype(np.float32)
        yo = np.concatenate([y, yn]).astype(int)
        wo = np.concatenate([w, wn]).astype(np.float32)
        go = np.concatenate([g, gn])
        return (Xo, yo, wo, go)

    def jitter_augment_tabular_with_groups(X, y, w, g, cont_idx, mult=0.5, sigma=0.01, per_feature=True, clip_low=None, clip_high=None, rs=None):
        if rs is None:
            rs = np.random.RandomState(42)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        w = np.asarray(w, dtype=np.float32)
        g = np.asarray(g)
        n, d = X.shape
        if cont_idx.size == 0:
            return (X, y, w, g)
        m = int(np.ceil(n * float(mult)))
        sel = rs.choice(np.arange(n), size=m, replace=n < m)
        Xn = X[sel].copy()
        yn = y[sel].copy()
        wn = w[sel].copy()
        gn = g[sel].copy()
        if per_feature:
            std = X[:, cont_idx].std(axis=0, ddof=0) + 1e-06
            noise = rs.normal(0.0, 1.0, size=(m, cont_idx.size)).astype(np.float32)
            Xn[:, cont_idx] = Xn[:, cont_idx] + noise * (float(sigma) * std)
        else:
            gstd = float(np.std(X[:, cont_idx])) + 1e-06
            noise = rs.normal(0.0, 1.0, size=(m, cont_idx.size)).astype(np.float32)
            Xn[:, cont_idx] = Xn[:, cont_idx] + noise * (float(sigma) * gstd)
        if clip_low is not None and clip_high is not None:
            Xn[:, cont_idx] = np.minimum(np.maximum(Xn[:, cont_idx], clip_low), clip_high)
        Xo = np.vstack([X, Xn]).astype(np.float32)
        yo = np.concatenate([y, yn]).astype(int)
        wo = np.concatenate([w, wn]).astype(np.float32)
        go = np.concatenate([g, gn])
        return (Xo, yo, wo, go)

    def apply_augs_in_3A(X_df, y_np, w_np, g_np, cfg, random_state=42):
        """Impute -> تعیین ستون\u200cهای پیوسته -> mixup -> dropout -> jitter -> بازگشت DataFrame/np"""
        rs = np.random.RandomState(random_state)
        imp = SimpleImputer(strategy='median')
        X_imp = imp.fit_transform(X_df)
        cont_idx = _cont_indices_from_df(X_df, max_nunique=cfg['exclude_discrete_max_nunique'])
        ql, qh = (None, None)
        if cfg['jitter']['enabled'] and cont_idx.size > 0:
            ql = np.quantile(X_imp[:, cont_idx], cfg['jitter']['clip_q'][0], axis=0)
            qh = np.quantile(X_imp[:, cont_idx], cfg['jitter']['clip_q'][1], axis=0)
        Xo, yo, wo, go = (X_imp, y_np.copy(), w_np.copy(), g_np.copy())
        if cfg['mixup']['enabled']:
            Xo, yo, wo, go = mixup_same_class_with_groups(Xo, yo, wo, go, alpha=cfg['mixup']['alpha'], mult=cfg['mixup']['mult'], exclude_discrete_max_nunique=cfg['exclude_discrete_max_nunique'], rs=rs)
            services.report(f'[AUG] mixup -> +{len(yo) - len(y_np)} samples')
        if cfg['dropout']['enabled']:
            Xo, yo, wo, go = dropout_augment_tabular_with_groups(Xo, yo, wo, go, p=cfg['dropout']['p'], mult=cfg['dropout']['mult'], strategy=cfg['dropout']['strategy'], exclude_discrete_max_nunique=cfg['exclude_discrete_max_nunique'], rs=rs)
            services.report(f'[AUG] dropout -> total {len(yo)} samples')
        if cfg['jitter']['enabled']:
            Xo, yo, wo, go = jitter_augment_tabular_with_groups(Xo, yo, wo, go, cont_idx=cont_idx, mult=cfg['jitter']['mult'], sigma=cfg['jitter']['sigma'], per_feature=cfg['jitter']['per_feature'], clip_low=ql, clip_high=qh, rs=rs)
            services.report(f'[AUG] jitter -> total {len(yo)} samples')
        X_df_aug = pd.DataFrame(Xo, columns=X_df.columns)
        return (X_df_aug, yo, wo, go)
    if AUG_CFG.get('apply_in_3A', True):
        X_train, y_train, review_weights_train, groups_train = apply_augs_in_3A(X_df=X_train, y_np=y_train, w_np=review_weights_train, g_np=groups_train, cfg=AUG_CFG, random_state=RANDOM_STATE)
        services.report(f'[AUG] Done. Train now: {len(y_train)} rows (incl. augmented).')
    else:
        services.report('[AUG] Skipped in 3A (you can apply inside 3B ES/final refit).')
    XGBClassifier = services.dependencies.XGBClassifier
    xgb = services.dependencies.xgb
    version = services.dependencies.version
    try:
        TrainingCallback = services.dependencies.TrainingCallback
    except Exception:
        TrainingCallback = object

    class TQDMCallback(TrainingCallback):

        def __init__(self, total: int, desc: str='XGBoost'):
            self.total = int(total)
            self.desc = str(desc)
            self.pbar = None

        def before_training(self, model):
            self.pbar = tqdm(total=self.total, desc=self.desc, leave=False)
            return model

        def after_iteration(self, model, epoch: int, evals_log: dict[str, dict[str, float]]):
            if self.pbar is not None:
                self.pbar.update(1)
            return False

        def after_training(self, model):
            if self.pbar is not None:
                self.pbar.close()
                return model
    use_cuda = services.device_policy.mode == 'cuda'
    if use_cuda and services.dependencies.cupy is None:
        raise ContractError('CUDA mode requires an explicitly injected CuPy dependency')
    N_CPU = services.cpu_count or 1
    if use_cuda:
        grid_n_jobs = 1
        xgb_n_jobs = max(1, min(8, N_CPU))
    else:
        grid_n_jobs = max(1, min(4, N_CPU))
        xgb_n_jobs = max(1, N_CPU // max(1, grid_n_jobs))
    services.set_thread_environment('OMP_NUM_THREADS', str(xgb_n_jobs), policy)
    services.set_thread_environment('MKL_NUM_THREADS', str(xgb_n_jobs), policy)
    xgb_kwargs = dict(objective='binary:logistic', eval_metric='aucpr', n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, min_child_weight=1.0, reg_lambda=1.0, scale_pos_weight=1.0, random_state=RANDOM_STATE, n_jobs=xgb_n_jobs, verbosity=0)
    if version.parse(xgb.__version__) >= version.parse('2.0.0'):
        xgb_kwargs['tree_method'] = 'hist'
        xgb_kwargs['max_bin'] = 256
        if use_cuda:
            xgb_kwargs['device'] = 'cuda'
    else:
        xgb_kwargs['tree_method'] = 'gpu_hist' if use_cuda else 'hist'
    xgb_base = XGBClassifier(**xgb_kwargs)

    def build_smote(y_like: np.ndarray, pos_label: int=POS_LABEL, sampling_strategy: float=0.5):
        pos_count = int((y_like == pos_label).sum())
        if pos_count < 2:
            return None
        k = max(1, min(5, pos_count - 1))
        return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=RANDOM_STATE)
    smote_global = build_smote(np.asarray(y_train), pos_label=POS_LABEL, sampling_strategy=0.5)
    pipe = ImbPipeline([('imp', SimpleImputer(strategy='median')), ('smote', smote_global if smote_global is not None else 'passthrough'), ('clf', xgb_base)])
    services.report(f'[CUDA] xgboost={xgb.__version__} | use_cuda={use_cuda} | tree_method={xgb_kwargs.get('tree_method')} | device={xgb_kwargs.get('device', 'cpu')}')
    return SourceResult_0007_0006(
        artifacts=_artifacts(ROUND_DIR, MODEL_PATH, THRESH_JSON_PATH, FI_CSV_PATH, TILE_PREDS_CSV_PATH, IMG_PREDS_CSV_PATH, PR_CURVE_CSV_PATH, ROC_CURVE_CSV_PATH, CONF_MAT_CSV_PATH, METRICS_JSON_PATH, DROPPED_COLS_JSON, CONFIG_JSON_PATH, HISTORY_CSV_PATH, PROGRESS_PNG_PATH),
        continuation=AugmentedContinuationState(
            AUG_CFG=MappingProxyType(dict(AUG_CFG)),
            CONFIG_JSON_PATH=CONFIG_JSON_PATH,
            CONF_MAT_CSV_PATH=CONF_MAT_CSV_PATH,
            CSV_PATH=CSV_PATH,
            split_manifest=inputs.split_manifest,
            previous_tile_predictions=inputs.previous_tile_predictions,
            previous_threshold=inputs.previous_threshold,
            DROPPED_COLS_JSON=DROPPED_COLS_JSON,
            EARLY_STOP_ROUNDS=EARLY_STOP_ROUNDS,
            EARLY_STOP_VAL_FRAC=EARLY_STOP_VAL_FRAC,
            FI_CSV_PATH=FI_CSV_PATH,
            GROUP_POS_FRAC=GROUP_POS_FRAC,
            HISTORY_CSV_PATH=HISTORY_CSV_PATH,
            IMG_PREDS_CSV_PATH=IMG_PREDS_CSV_PATH,
            METRICS_JSON_PATH=METRICS_JSON_PATH,
            MODEL_NAME=MODEL_NAME,
            MODEL_PATH=MODEL_PATH,
            NAME_COL=NAME_COL,
            N_ESTIMATORS_BIG=N_ESTIMATORS_BIG,
            N_SPLITS_CV=N_SPLITS_CV,
            POS_LABEL=POS_LABEL,
            PROGRESS_PNG_PATH=PROGRESS_PNG_PATH,
            PR_CURVE_CSV_PATH=PR_CURVE_CSV_PATH,
            RANDOM_STATE=RANDOM_STATE,
            ROC_CURVE_CSV_PATH=ROC_CURVE_CSV_PATH,
            ROUND_DIR=ROUND_DIR,
            ROUND_TAG=ROUND_TAG,
            TEST_SIZE=TEST_SIZE,
            THRESH_JSON_PATH=THRESH_JSON_PATH,
            TILE_PREDS_CSV_PATH=TILE_PREDS_CSV_PATH,
            TOPK_FEATURES_TO_PRINT=TOPK_FEATURES_TO_PRINT,
            TQDMCallback=TQDMCallback,
            Timer=Timer,
            USE_REVIEW_WEIGHTING=USE_REVIEW_WEIGHTING,
            X_test=X_test,
            X_train=X_train,
            total_started_at=__T_TOTAL0,
            apply_augs_in_3A=apply_augs_in_3A,
            build_smote=build_smote,
            compute_group_labels=compute_group_labels,
            df=df,
            fit_with_maybe_callbacks=fit_with_maybe_callbacks,
            grid_n_jobs=grid_n_jobs,
            groups_test=groups_test,
            groups_train=groups_train,
            make_weights_after_smote=make_weights_after_smote,
            num_cols=tuple(str(item) for item in num_cols),
            review_weights_train=review_weights_train,
            test_idx=test_idx,
            to_drop=frozenset(str(item) for item in dropped_cols),
            train_idx=train_idx,
            xgb_base=xgb_base,
            xgb_kwargs=MappingProxyType(dict(xgb_kwargs)),
            y_test=y_test,
            y_train=y_train,
        ),
    )

# SOURCE_CELL: NB-LIVE-0007-C0007
# SOURCE_STATEMENT_MAP: sanitized-body -> source_algorithm_0007_0007
@dataclass(frozen=True)
class SourceInputs_0007_0007:
    state: AugmentedContinuationState

@dataclass(frozen=True)
class SourceResult_0007_0007:
    artifacts: TrainingArtifacts

def source_algorithm_0007_0007(
    inputs: SourceInputs_0007_0007,
    services: SourceAlgorithmServices,
    policy: ExecutionPolicy,
) -> SourceResult_0007_0007:
    require_authorized(policy)
    state: AugmentedContinuationState = inputs.state
    DummyClassifier = services.dependencies.DummyClassifier
    GroupShuffleSplit = services.dependencies.GroupShuffleSplit
    ImbPipeline = services.dependencies.ImbPipeline
    Pipeline = services.dependencies.Pipeline
    RandomizedSearchCV = services.dependencies.RandomizedSearchCV
    SimpleImputer = services.dependencies.SimpleImputer
    XGBClassifier = services.dependencies.XGBClassifier
    average_precision_score = services.dependencies.average_precision_score
    confusion_matrix = services.dependencies.confusion_matrix
    loguniform = services.dependencies.loguniform
    np = services.dependencies.np
    pd = services.dependencies.pd
    precision_recall_curve = services.dependencies.precision_recall_curve
    randint = services.dependencies.randint
    roc_auc_score = services.dependencies.roc_auc_score
    roc_curve = services.dependencies.roc_curve
    tqdm = services.dependencies.tqdm
    uniform = services.dependencies.uniform
    AUG_CFG = dict(state.AUG_CFG)
    CONFIG_JSON_PATH = state.CONFIG_JSON_PATH
    CONF_MAT_CSV_PATH = state.CONF_MAT_CSV_PATH
    CSV_PATH = state.CSV_PATH
    DROPPED_COLS_JSON = state.DROPPED_COLS_JSON
    EARLY_STOP_ROUNDS = state.EARLY_STOP_ROUNDS
    EARLY_STOP_VAL_FRAC = state.EARLY_STOP_VAL_FRAC
    FI_CSV_PATH = state.FI_CSV_PATH
    GROUP_POS_FRAC = state.GROUP_POS_FRAC
    HISTORY_CSV_PATH = state.HISTORY_CSV_PATH
    IMG_PREDS_CSV_PATH = state.IMG_PREDS_CSV_PATH
    METRICS_JSON_PATH = state.METRICS_JSON_PATH
    MODEL_NAME = state.MODEL_NAME
    MODEL_PATH = state.MODEL_PATH
    NAME_COL = state.NAME_COL
    N_ESTIMATORS_BIG = state.N_ESTIMATORS_BIG
    N_SPLITS_CV = state.N_SPLITS_CV
    POS_LABEL = state.POS_LABEL
    PROGRESS_PNG_PATH = state.PROGRESS_PNG_PATH
    PR_CURVE_CSV_PATH = state.PR_CURVE_CSV_PATH
    RANDOM_STATE = state.RANDOM_STATE
    ROC_CURVE_CSV_PATH = state.ROC_CURVE_CSV_PATH
    ROUND_DIR = state.ROUND_DIR
    ROUND_TAG = state.ROUND_TAG
    TEST_SIZE = state.TEST_SIZE
    THRESH_JSON_PATH = state.THRESH_JSON_PATH
    TILE_PREDS_CSV_PATH = state.TILE_PREDS_CSV_PATH
    TOPK_FEATURES_TO_PRINT = state.TOPK_FEATURES_TO_PRINT
    TQDMCallback = state.TQDMCallback
    Timer = state.Timer
    USE_REVIEW_WEIGHTING = state.USE_REVIEW_WEIGHTING
    X_test = state.X_test
    X_train = state.X_train
    __T_TOTAL0 = state.total_started_at
    apply_augs_in_3A = state.apply_augs_in_3A
    build_smote = state.build_smote
    compute_group_labels = state.compute_group_labels
    df = state.df
    fit_with_maybe_callbacks = state.fit_with_maybe_callbacks
    grid_n_jobs = state.grid_n_jobs
    groups_test = state.groups_test
    groups_train = state.groups_train
    make_weights_after_smote = state.make_weights_after_smote
    num_cols = list(state.num_cols)
    review_weights_train = state.review_weights_train
    test_idx = state.test_idx
    to_drop = set(state.to_drop)
    train_idx = state.train_idx
    xgb_base = state.xgb_base
    xgb_kwargs = dict(state.xgb_kwargs)
    y_test = state.y_test
    y_train = state.y_train
    StratifiedKFold = services.dependencies.StratifiedKFold
    GroupKFold = services.dependencies.GroupKFold
    used_cv = False
    cv_best_pr = None
    best_params: dict[str, object] = {}
    _AUG = None
    try:
        _AUG = AUG_CFG
    except NameError:
        _AUG = None
    if _AUG is not None and (not _AUG.get('apply_in_3A', True)):
        try:
            X_train, y_train, review_weights_train, groups_train = apply_augs_in_3A(X_df=X_train, y_np=y_train, w_np=review_weights_train, g_np=groups_train, cfg=_AUG, random_state=RANDOM_STATE)
            services.report(f'[AUG][3B] Applied (pre-CV). Train now: {len(y_train)} rows.')
        except Exception as e:
            services.report(f'[AUG][3B] Skipped due to error: {e}')
    else:
        services.report('[AUG][3B] No extra augmentation (already applied in 3A or disabled).')
    try:
        smote_global = build_smote(np.asarray(y_train), pos_label=POS_LABEL, sampling_strategy=0.5)
    except NameError as e:
        raise RuntimeError('build_smote not found (Cell3A must be run first).') from e
    try:
        xgb_base
    except NameError as e:
        raise RuntimeError('xgb_base not found (Cell3A must be run first).') from e
    pipe = ImbPipeline([('imp', SimpleImputer(strategy='median')), ('smote', smote_global if smote_global is not None else 'passthrough'), ('clf', xgb_base)])
    param_distributions = {'clf__max_depth': randint(3, 10), 'clf__min_child_weight': loguniform(0.5, 10.0), 'clf__subsample': uniform(0.7, 0.3), 'clf__colsample_bytree': uniform(0.7, 0.3), 'clf__reg_lambda': loguniform(0.1, 10.0), 'clf__reg_alpha': loguniform(0.001, 1.0), 'clf__gamma': loguniform(0.001, 1.0), 'clf__learning_rate': loguniform(0.02, 0.2), 'clf__max_bin': [256]}
    if smote_global is not None:
        param_distributions.update({'smote__sampling_strategy': [0.4, 0.5, 0.6], 'smote__k_neighbors': randint(2, 6)})
    if len(np.unique(y_train)) < 2:
        services.report('[WARN] Only one class in TRAIN ? DummyClassifier.')
        best_model = Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', DummyClassifier(strategy='constant', constant=int(np.unique(y_train)[0])))])
        best_model.fit(X_train, y_train)
    else:
        if len(np.unique(groups_train)) >= 2:
            uniq_groups_cv, labels_cv = compute_group_labels(groups_train, y_train, frac=GROUP_POS_FRAC)
            n_pos_g = int((labels_cv == 1).sum())
            n_neg_g = int((labels_cv == 0).sum())
            services.report(f'[CV] Groups: total={len(uniq_groups_cv)} | pos_groups={n_pos_g} | neg_groups={n_neg_g}')

            def indices_from_groups(g_tr_set, g_va_set):
                tr_idx = np.where(np.isin(groups_train, list(g_tr_set)))[0]
                va_idx = np.where(np.isin(groups_train, list(g_va_set)))[0]
                return (tr_idx, va_idx)
            cv_pairs = []
            used_val_group_signatures = set()
            max_tries = 400
            if len(np.unique(labels_cv)) == 2:
                for target_k in [5, 4, 3, 2]:
                    tries = 0
                    with tqdm(total=max_tries, desc=f'Build stratified folds (k={target_k})', leave=False) as pbar:
                        while len(cv_pairs) < target_k and tries < max_tries:
                            skf = StratifiedKFold(n_splits=target_k, shuffle=True, random_state=RANDOM_STATE + tries)
                            for gi_tr, gi_va in skf.split(uniq_groups_cv, labels_cv):
                                g_tr = set(uniq_groups_cv[gi_tr])
                                g_va = set(uniq_groups_cv[gi_va])
                                tr_idx, va_idx = indices_from_groups(g_tr, g_va)
                                if len(np.unique(y_train[va_idx])) < 2 or len(np.unique(y_train[tr_idx])) < 2:
                                    continue
                                sig = tuple(sorted(g_va))
                                if sig in used_val_group_signatures:
                                    continue
                                used_val_group_signatures.add(sig)
                                cv_pairs.append((tr_idx, va_idx))
                                if len(cv_pairs) >= target_k:
                                    break
                            tries += 1
                            pbar.update(1)
                    if len(cv_pairs) >= 2:
                        services.report(f'[INFO] Using {len(cv_pairs)} CV folds (stratified target={target_k}).')
                        break
            else:
                for target_k in [5, 4, 3, 2]:
                    if target_k > len(np.unique(groups_train)):
                        continue
                    gkf = GroupKFold(n_splits=target_k)
                    tmp_pairs = []
                    for tr_idx, va_idx in gkf.split(X_train, y_train, groups_train):
                        if len(np.unique(y_train[va_idx])) < 2 or len(np.unique(y_train[tr_idx])) < 2:
                            continue
                        tmp_pairs.append((tr_idx, va_idx))
                    if len(tmp_pairs) >= 2:
                        cv_pairs = tmp_pairs[:target_k]
                        services.report(f'[INFO] Using {len(cv_pairs)} CV folds from GroupKFold(k={target_k}).')
                        break
            if len(cv_pairs) < 2:
                rs = np.random.RandomState(RANDOM_STATE)
                perm = rs.permutation(np.unique(groups_train))
                cut = max(1, int(0.2 * len(perm)))
                g_va = set(perm[:cut])
                g_tr = set(perm[cut:])
                tr_idx = np.where(np.isin(groups_train, list(g_tr)))[0]
                va_idx = np.where(np.isin(groups_train, list(g_va)))[0]
                if len(np.unique(y_train[va_idx])) < 2 or len(np.unique(y_train[tr_idx])) < 2:
                    tr_idx, va_idx = (va_idx, tr_idx)
                cv_pairs = [(tr_idx, va_idx)]
                services.report('[WARN] Could not form ≥2 valid folds; using 1 group split (~80/20).')
            N_ITER_RANDOM = 50
            gs = RandomizedSearchCV(estimator=pipe, param_distributions=param_distributions, n_iter=N_ITER_RANDOM, scoring='average_precision', cv=cv_pairs, n_jobs=grid_n_jobs, verbose=2, error_score=np.nan, refit=False, random_state=RANDOM_STATE)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=UserWarning)
                    with Timer('RandomizedSearchCV.fit'):
                        gs.fit(X_train, y_train)
                best_params = gs.best_params_
                services.report('Best PR-AUC (cv):', gs.best_score_)
                services.report('Best params:', best_params)
                cv_best_pr = float(gs.best_score_)
                used_cv = True
            except Exception as e:
                services.report(f'[WARN] RandomizedSearch failed ({e}). Falling back to base params.')
                best_params = {}
        else:
            services.report('[WARN] Not enough groups for GroupKFold. Skipping CV.')
            best_params = {}

        def refit_with_es_return_pipeline(base_params: dict[str, object], X_tr: pd.DataFrame, y_tr: np.ndarray, groups_tr: np.ndarray, review_weights_tr: np.ndarray) -> ImbPipeline:
            if len(np.unique(groups_tr)) >= 2 and len(y_tr) >= 50:
                gss_inner = GroupShuffleSplit(n_splits=1, test_size=EARLY_STOP_VAL_FRAC, random_state=RANDOM_STATE + 123)
                tr_idx, val_idx = next(gss_inner.split(X_tr, y_tr, groups_tr))
                if len(np.unique(y_tr[val_idx])) < 2:
                    idx = np.arange(len(X_tr))
                    rs = np.random.RandomState(RANDOM_STATE + 123)
                    rs.shuffle(idx)
                    cut = max(1, int(len(idx) * EARLY_STOP_VAL_FRAC))
                    val_idx, tr_idx = (idx[:cut], idx[cut:])
            else:
                idx = np.arange(len(X_tr))
                rs = np.random.RandomState(RANDOM_STATE + 123)
                rs.shuffle(idx)
                cut = max(1, int(len(idx) * EARLY_STOP_VAL_FRAC))
                val_idx, tr_idx = (idx[:cut], idx[cut:])
            X_tr2, y_tr2 = (X_tr.iloc[tr_idx], y_tr[tr_idx])
            X_val, y_val = (X_tr.iloc[val_idx], y_tr[val_idx])
            w_tr2_review = review_weights_tr[tr_idx].astype(np.float32)
            if _AUG is not None and (not _AUG.get('apply_in_3A', True)):
                try:
                    X_tr2, y_tr2, w_tr2_review, _ = apply_augs_in_3A(X_df=X_tr2, y_np=y_tr2, w_np=w_tr2_review, g_np=groups_tr[tr_idx], cfg=_AUG, random_state=RANDOM_STATE + 11)
                    services.report(f'[AUG][ES] Applied on inner-train fold. Size now: {len(y_tr2)}')
                except Exception as e:
                    services.report(f'[AUG][ES] Skipped due to error: {e}')
            imp = SimpleImputer(strategy='median')
            X_tr2_imp = imp.fit_transform(X_tr2)
            X_val_imp = imp.transform(X_val)
            smote_local = build_smote(y_tr2, pos_label=POS_LABEL, sampling_strategy=0.5)
            if smote_local is not None:
                X_tr2_fit, y_tr2_fit = smote_local.fit_resample(X_tr2_imp, y_tr2)
                w_tr2_fit = make_weights_after_smote(w_tr2_review, y_tr2, y_tr2_fit)
            else:
                X_tr2_fit, y_tr2_fit = (X_tr2_imp, y_tr2)
                w_tr2_fit = w_tr2_review
            xgb_kwargs_local = xgb_kwargs.copy()
            xgb_kwargs_local['scale_pos_weight'] = 1.0
            clf = XGBClassifier(**xgb_kwargs_local)
            if base_params:
                clf.set_params(**{k.replace('clf__', ''): v for k, v in base_params.items()})
            clf.set_params(n_estimators=N_ESTIMATORS_BIG, early_stopping_rounds=EARLY_STOP_ROUNDS)
            try:
                with Timer('Refit with Early-Stopping (inner split)'):
                    fit_with_maybe_callbacks(clf, X_tr2_fit, y_tr2_fit, eval_set=[(X_val_imp, y_val)], verbose=False, callbacks=[xgb.callback.EvaluationMonitor(period=25)], sample_weight=w_tr2_fit)
                used = getattr(clf, 'best_iteration', None)
                if used is None:
                    used = getattr(clf, 'best_ntree_limit', None)
                services.report(f'[ES] early stopping applied; effective trees ≈ {used}')
            except Exception as e:
                services.report(f'[WARN] ES fit failed ({e}); fitting WITHOUT ES.')
                clf.set_params(early_stopping_rounds=None)
                with Timer('Refit WITHOUT ES (inner split)'):
                    fit_with_maybe_callbacks(clf, X_tr2_fit, y_tr2_fit, callbacks=[TQDMCallback(total=clf.get_params().get('n_estimators', 300), desc='XGB[No-ES]')], sample_weight=w_tr2_fit)
            final_steps = [('imp', imp)]
            final_steps.append(('smote', smote_local if smote_local is not None else 'passthrough'))
            final_steps.append(('clf', clf))
            return ImbPipeline(final_steps)
        with Timer('Assemble best_model via ES refit'):
            best_model = refit_with_es_return_pipeline(best_params, X_train, y_train, groups_train, review_weights_tr=review_weights_train)
    clf = best_model.named_steps.get('clf', None)
    if clf is not None:
        used_trees = getattr(clf, 'best_iteration', None)
        used_trees = int(used_trees) + 1 if used_trees is not None else getattr(clf, 'best_ntree_limit', None)
        if used_trees is not None:
            try:
                best_model.set_output(transform='pandas')
            except Exception:
                services.report('[REFIT] pandas output configuration unavailable; continuing with estimator defaults.')
            best_model.set_params(clf__early_stopping_rounds=None, clf__callbacks=None, clf__n_estimators=int(used_trees), clf__scale_pos_weight=1.0)
            services.report(f'[REFIT] Re-training on FULL TRAIN with n_estimators={int(used_trees)}')
            with Timer('Full-train refit'):
                clf_final = best_model.named_steps['clf']
                imp_final = best_model.named_steps['imp']
                smote_final = best_model.named_steps.get('smote', None)
                X_train_imp = imp_final.fit_transform(X_train)
                if smote_final is not None and smote_final != 'passthrough':
                    smote_final = build_smote(y_train, pos_label=POS_LABEL, sampling_strategy=0.5)
                    if smote_final is not None:
                        X_train_fit, y_train_fit = smote_final.fit_resample(X_train_imp, y_train)
                        sample_weight_full_fit = make_weights_after_smote(review_weights_train, y_train, y_train_fit)
                    else:
                        X_train_fit, y_train_fit = (X_train_imp, y_train)
                        sample_weight_full_fit = review_weights_train
                else:
                    X_train_fit, y_train_fit = (X_train_imp, y_train)
                    sample_weight_full_fit = review_weights_train
                fit_with_maybe_callbacks(clf_final, X_train_fit, y_train_fit, callbacks=[xgb.callback.EvaluationMonitor(period=25)], sample_weight=sample_weight_full_fit)
                steps = [('imp', imp_final)]
                steps.append(('smote', smote_final if smote_final is not None else 'passthrough'))
                steps.append(('clf', clf_final))
                best_model = ImbPipeline(steps)
        else:
            services.report('[REFIT] Could not detect best_iteration/best_ntree_limit; skipping full-train refit.')
    try:
        clf_fitted = best_model.named_steps.get('clf', None)
        params_now = getattr(clf_fitted, 'get_xgb_params', lambda: {})()
        services.report(f'[CUDA] Final booster params → tree_method={params_now.get('tree_method')} | device={params_now.get('device')}')
    except Exception as e:
        services.report(f'[CUDA] Could not retrieve booster params: {e}')

    def pick_threshold_by_pr(y_true, proba):
        prec, rec, thr = precision_recall_curve(y_true, proba)
        if len(thr) == 0:
            return (0.5, prec, rec, thr)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_i = int(np.argmax(f1s))
        return (float(thr[max(best_i - 1, 0)]), prec, rec, thr)

    def predict_proba_batched(model, Xdf: pd.DataFrame, batch: int=50000, desc: str='Predict proba') -> np.ndarray:
        n = len(Xdf)
        out = np.empty(n, dtype=np.float32)
        for i in tqdm(range(0, n, batch), desc=desc, leave=False):
            j = min(i + batch, n)
            out[i:j] = model.predict_proba(Xdf.iloc[i:j])[:, 1]
        return out
    tile_metrics: dict[str, object] = {}
    img_metrics: dict[str, object] = {}
    thr_use = 0.5
    split_used = 'test' if len(test_idx) > 0 and len(np.unique(y_test)) >= 2 else 'train'
    if split_used == 'test':
        with Timer('Predict proba on TEST'):
            proba_test = predict_proba_batched(best_model, X_test, desc='Predict TEST')
        roc = roc_auc_score(y_test, proba_test)
        pr = average_precision_score(y_test, proba_test)
        thr_use, prec, rec, thr = pick_threshold_by_pr(y_test, proba_test)
        y_pred = (proba_test >= thr_use).astype(int)
        services.report(f'Tile-level: ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}')
        services.report(f'Best F1 via PR on TEST at threshold≈{thr_use:.3f}')
        cm = confusion_matrix(y_test, y_pred)
        test_df = df.iloc[test_idx].copy()
        test_df['__group'] = groups_test
        test_df['__proba'] = proba_test
        test_df['__y'] = y_test
        img_agg = test_df.groupby('__group').agg(y_img=('__y', 'max'), p_img=('__proba', 'max')).reset_index()
        try:
            roc_img = roc_auc_score(img_agg['y_img'], img_agg['p_img'])
        except Exception:
            roc_img = float('nan')
        pr_img = average_precision_score(img_agg['y_img'], img_agg['p_img'])
        services.report(f'Image-level: ROC-AUC={roc_img:.4f}  PR-AUC={pr_img:.4f}')
        tile_metrics = {'roc_auc_test': float(roc), 'pr_auc_test': float(pr)}
        img_metrics = {'roc_auc_test': float(roc_img), 'pr_auc_test': float(pr_img)}
        tmp = df.iloc[test_idx][[NAME_COL]].copy()
        tmp['group'] = groups_test
        tmp['y_true'] = y_test
        tmp['proba'] = proba_test
        tmp.to_csv(TILE_PREDS_CSV_PATH, index=False)
        img_agg.rename(columns={'__group': 'group', 'y_img': 'y_true', 'p_img': 'proba'}, inplace=True)
        img_agg.to_csv(IMG_PREDS_CSV_PATH, index=False)
        pr_df = pd.DataFrame({'precision': prec[:-1], 'recall': rec[:-1], 'threshold': thr})
        pr_df.to_csv(PR_CURVE_CSV_PATH, index=False)
        fpr, tpr, roc_thr = roc_curve(y_test, proba_test)
        pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': roc_thr}).to_csv(ROC_CURVE_CSV_PATH, index=False)
        pd.DataFrame(cm, index=['true_0', 'true_1'], columns=['pred_0', 'pred_1']).to_csv(CONF_MAT_CSV_PATH)
    else:
        services.report('[WARN] No independent TEST split; selecting threshold on TRAIN (optimistic).')
        with Timer('Predict proba on TRAIN'):
            proba_tr = predict_proba_batched(best_model, X_train, desc='Predict TRAIN')
        thr_use, prec, rec, thr = pick_threshold_by_pr(y_train, proba_tr)
        try:
            roc_tr = roc_auc_score(y_train, proba_tr) if len(np.unique(y_train)) >= 2 else float('nan')
            pr_tr = average_precision_score(y_train, proba_tr)
            services.report(f'(TRAIN diag) Tile-level: ROC-AUC={roc_tr:.4f}  PR-AUC={pr_tr:.4f}')
            tile_metrics = {'roc_auc_train': float(roc_tr), 'pr_auc_train': float(pr_tr)}
        except Exception:
            tile_metrics = {'roc_auc_train': None, 'pr_auc_train': None}
            services.report('[WARN] TRAIN diagnostic metrics were unavailable.')
        tmp = df.iloc[train_idx][[NAME_COL]].copy()
        tmp['group'] = groups_train
        tmp['y_true'] = y_train
        tmp['proba'] = proba_tr
        tmp.to_csv(TILE_PREDS_CSV_PATH, index=False)
        pd.DataFrame({'precision': prec[:-1], 'recall': rec[:-1], 'threshold': thr}).to_csv(PR_CURVE_CSV_PATH, index=False)
        fpr, tpr, roc_thr = roc_curve(y_train, proba_tr)
        pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': roc_thr}).to_csv(ROC_CURVE_CSV_PATH, index=False)
    joblib = services.dependencies.joblib
    xgb = services.dependencies.xgb
    meta = {'csv_path': CSV_PATH, 'round_tag': ROUND_TAG, 'pos_label': int(POS_LABEL), 'xgboost_version': xgb.__version__, 'use_cuda_try': bool(services.device_policy.mode == 'cuda'), 'used_cv': bool(used_cv), 'cv_best_pr_auc': None if cv_best_pr is None else float(cv_best_pr), 'tile_metrics': tile_metrics, 'img_metrics': img_metrics, 'class_balance_train': np.bincount(y_train).tolist(), 'class_balance_test': np.bincount(y_test).tolist() if len(test_idx) > 0 else None, 'split_used_for_threshold': split_used, 'test_size': float(TEST_SIZE), 'random_state': int(RANDOM_STATE), 'features_count': len(num_cols), 'early_stopping_rounds': int(EARLY_STOP_ROUNDS), 'early_stop_val_frac': float(EARLY_STOP_VAL_FRAC), 'best_iteration': int(getattr(getattr(best_model, 'named_steps', {}).get('clf', object()), 'best_iteration', -1)) if hasattr(best_model, 'named_steps') else -1, 'hybrid_mode': True, 'review_weight_mean': float(np.mean(review_weights_train)) if USE_REVIEW_WEIGHTING else None}
    clf_for_meta = best_model.named_steps.get('clf', None)
    if clf_for_meta is not None:
        meta['used_trees'] = int(getattr(clf_for_meta, 'n_estimators', 0))
        bi = getattr(clf_for_meta, 'best_iteration', None)
        if bi is not None:
            meta['es_best_iteration'] = int(bi)
        meta['early_stopping_rounds'] = int(EARLY_STOP_ROUNDS)
    joblib.dump({'pipeline': best_model, 'features': num_cols, 'threshold': float(thr_use), 'meta': meta}, MODEL_PATH)
    with open(THRESH_JSON_PATH, 'w') as f:
        json.dump({'threshold': float(thr_use)}, f, indent=2)
    with open(DROPPED_COLS_JSON, 'w') as f:
        json.dump(sorted(list(to_drop)), f, indent=2)
    with open(CONFIG_JSON_PATH, 'w') as f:
        json.dump({'MODEL_NAME': MODEL_NAME, 'ROUND_DIR': str(ROUND_DIR), 'ROUND_TAG': ROUND_TAG, 'TEST_SIZE': TEST_SIZE, 'RANDOM_STATE': RANDOM_STATE, 'N_SPLITS_CV': N_SPLITS_CV, 'POS_LABEL': POS_LABEL, 'TOPK_FEATURES_TO_PRINT': TOPK_FEATURES_TO_PRINT, 'used_features': num_cols, 'split_mode': 'SealedManifest(GroupPure)', 'split_manifest': str(state.split_manifest.path), 'split_manifest_sha256': state.split_manifest.sha256, 'previous_tile_predictions': str(state.previous_tile_predictions.path), 'previous_tile_predictions_sha256': state.previous_tile_predictions.sha256, 'previous_threshold': str(state.previous_threshold.path), 'previous_threshold_sha256': state.previous_threshold.sha256}, f, indent=2)
    services.report(f'[OK] saved model to: {MODEL_PATH}')
    services.report(f'[OK] saved threshold JSON to: {THRESH_JSON_PATH}')
    try:
        clf_fitted = best_model.named_steps.get('clf', None)
        importances = getattr(clf_fitted, 'feature_importances_', None)
        if importances is not None:
            fi_full = pd.Series(importances, index=num_cols).sort_values(ascending=False)
            services.report('\nTop feature importances:')
            services.report(fi_full.head(TOPK_FEATURES_TO_PRINT).round(6).to_string())
            fi_full.to_csv(FI_CSV_PATH)
            services.report(f'[OK] saved full feature importances to: {FI_CSV_PATH}')
    except Exception as e:
        warnings.warn(f'could not compute/save feature importances: {e}')
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump({'round_tag': ROUND_TAG, 'tile_metrics': tile_metrics, 'img_metrics': img_metrics, 'threshold': float(thr_use), 'split_used_for_threshold': split_used}, f, indent=2)
    hist_row = {'round': ROUND_TAG, 'features_csv': CSV_PATH, 'threshold': float(thr_use), 'used_cv': bool(used_cv), 'cv_best_pr_auc': None if cv_best_pr is None else float(cv_best_pr), 'tile_pr_auc': float(tile_metrics.get('pr_auc_test', tile_metrics.get('pr_auc_train', np.nan))), 'tile_roc_auc': float(tile_metrics.get('roc_auc_test', tile_metrics.get('roc_auc_train', np.nan))), 'img_pr_auc': float(img_metrics.get('pr_auc_test', np.nan)), 'img_roc_auc': float(img_metrics.get('roc_auc_test', np.nan)), 'train_pos': int(np.bincount(y_train)[1] if len(np.bincount(y_train)) > 1 else 0), 'train_neg': int(np.bincount(y_train)[0] if len(np.bincount(y_train)) > 0 else 0), 'n_tiles_train': int(len(y_train)), 'n_tiles_test': int(len(y_test)), 'n_groups_train': int(len(np.unique(groups_train))), 'n_groups_test': int(len(np.unique(groups_test))) if len(test_idx) > 0 else 0}
    if HISTORY_CSV_PATH.exists():
        hist = pd.read_csv(HISTORY_CSV_PATH)
        hist = hist[hist['round'] != ROUND_TAG]
        hist = pd.concat([hist, pd.DataFrame([hist_row])], ignore_index=True)
    else:
        hist = pd.DataFrame([hist_row])
    hist = hist.sort_values(by='round', key=lambda s: s.str.extract('r(\\d+)', expand=False).astype(float), ignore_index=True)
    hist.to_csv(HISTORY_CSV_PATH, index=False)
    services.report(f'[OK] history updated: {HISTORY_CSV_PATH}')
    try:
        plt = services.dependencies.plt
        h = pd.read_csv(HISTORY_CSV_PATH)
        h['_rnum'] = h['round'].str.extract('r(\\d+)', expand=False).astype(float)
        plt.figure()
        if 'tile_pr_auc' in h.columns:
            plt.plot(h['_rnum'].values, h['tile_pr_auc'].values, marker='o', label='Tile PR-AUC')
        if 'img_pr_auc' in h.columns and h['img_pr_auc'].notna().any():
            plt.plot(h['_rnum'].values, h['img_pr_auc'].values, marker='o', linestyle='--', label='Image PR-AUC')
        plt.xlabel('Round')
        plt.ylabel('PR-AUC')
        plt.title('Active Learning Progress (PR-AUC)')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PROGRESS_PNG_PATH, dpi=150)
        plt.close()
        services.report(f'[OK] saved progress plot: {PROGRESS_PNG_PATH}')
    except Exception as e:
        services.report(f'[WARN] could not render progress plot: {e}')
    services.report(f'[TIMER] TOTAL took {time.perf_counter() - __T_TOTAL0:.2f} sec')
    return SourceResult_0007_0007(_artifacts(ROUND_DIR, MODEL_PATH, THRESH_JSON_PATH, FI_CSV_PATH, TILE_PREDS_CSV_PATH, IMG_PREDS_CSV_PATH, PR_CURVE_CSV_PATH, ROC_CURVE_CSV_PATH, CONF_MAT_CSV_PATH, METRICS_JSON_PATH, DROPPED_COLS_JSON, CONFIG_JSON_PATH, HISTORY_CSV_PATH, PROGRESS_PNG_PATH))

@dataclass(frozen=True)
class SourceAlgorithmBatch:
    hybrid: SourceInputs_0007_0000
    augmented: SourceInputs_0007_0006

@dataclass(frozen=True)
class SourceAlgorithmBatchResult:
    cell_ids: tuple[str, str, str, str]
    results: tuple[SourceResult_0007_0000, SourceResult_0007_0001, SourceResult_0007_0006, SourceResult_0007_0007]

def execute_xgb_nb7_source_algorithms(
    batch: SourceAlgorithmBatch,
    services: SourceAlgorithmServices,
    policy: ExecutionPolicy,
) -> SourceAlgorithmBatchResult:
    require_authorized(policy)
    hybrid_setup = source_algorithm_0007_0000(batch.hybrid, services, policy)
    hybrid_fit = source_algorithm_0007_0001(SourceInputs_0007_0001(hybrid_setup.continuation), services, policy)
    augmented_setup = source_algorithm_0007_0006(batch.augmented, services, policy)
    augmented_fit = source_algorithm_0007_0007(SourceInputs_0007_0007(augmented_setup.continuation), services, policy)
    return SourceAlgorithmBatchResult(
        SOURCE_ALGORITHM_ORDER,
        (hybrid_setup, hybrid_fit, augmented_setup, augmented_fit),
    )
