"""Authorized source-backed NB0001 XGBoost setup and fit algorithms.

The reviewed source specification is the implementation authority. Scientific libraries are
resolved only through an explicitly authorized service boundary.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from hashlib import sha256
from os import fstat
from pathlib import Path
from stat import S_ISREG
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

from ..contracts import ContractError, ExecutionPolicy, require_authorized
from .source_paths import prepare_source_output_directory

ALLOWED_THREAD_ENVIRONMENT_KEYS = frozenset({'MKL_NUM_THREADS', 'OMP_NUM_THREADS'})
SOURCE_ALGORITHM_ORDER = ('NB-LIVE-0001-C0005', 'NB-LIVE-0001-C0006')
SOURCE_BODY_SHA256: Mapping[str, str] = {
    'NB-LIVE-0001-C0005': '75b6c43d431b423ea62cfb76234fe6e0335d532e4f0968cf169a76533335b27d',
    'NB-LIVE-0001-C0006': '0c1e5f26b2e86e1cf4c3a066bc35397299ea04d0facb99f930e1cd75f7bf08e6',
}
SOURCE_OWNER_COUNT: Mapping[str, int] = {
    'NB-LIVE-0001-C0005': 674,
    'NB-LIVE-0001-C0006': 494,
}
SOURCE_TRANSFORMATION_CLASSES: Mapping[str, tuple[str, ...]] = {
    'NB-LIVE-0001-C0005': (
        'AMBIENT_CONFIG_TO_TYPED_INPUT',
        'AMBIENT_CPU_COUNT_TO_EXPLICIT_SERVICE',
        'AMBIENT_ENVIRONMENT_WRITE_TO_INJECTED_SERVICE',
        'AUTO_DISCOVERY_TO_EXPLICIT_VERIFIED_ASSETS',
        'CAPABILITY_PROBE_TO_EXPLICIT_DEVICE_POLICY',
        'GLOBAL_SYMBOL_PROBE_TO_LOCAL_SOURCE_ALGORITHM',
        'PRINT_DISPLAY_TO_INJECTED_REPORTER',
        'SCIENTIFIC_IMPORT_TO_INJECTED_DEPENDENCY',
        'SOURCE_TOKEN_TO_TYPED_ROOT',
        'TOKENIZED_TRAINING_PATH_TO_VERIFIED_ARTIFACT',
        'TYPE_ANNOTATION_NORMALIZATION',
    ),
    'NB-LIVE-0001-C0006': (
        'AMBIENT_CONFIG_TO_TYPED_CONTINUATION_STATE',
        'AUTO_DISCOVERY_TO_EXPLICIT_VERIFIED_ASSET',
        'EXPLICIT_TYPED_CONTINUATION_STATE',
        'PRINT_DISPLAY_TO_INJECTED_REPORTER',
        'SCIENTIFIC_IMPORT_TO_INJECTED_DEPENDENCY',
        'TYPE_ANNOTATION_NORMALIZATION',
    ),
}

class SourceAlgorithmDependencies(Protocol):
    GroupKFold: Any
    GroupShuffleSplit: Any
    ImbPipeline: Any
    ParameterSampler: Any
    ParserError: Any
    SMOTE: Any
    SimpleImputer: Any
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
            raise ContractError('source input artifact path must be absolute')
        if not isinstance(self.sha256, str) or len(self.sha256) != 64 or any(character not in '0123456789abcdef' for character in self.sha256):
            raise ContractError('source input artifact SHA-256 must be lowercase hexadecimal')


def _read_split_groups(path: Path) -> tuple[set[str], set[str], set[str]]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError('training split manifest cannot be read') from exc
    if not isinstance(value, Mapping):
        raise ContractError('training split manifest must be an object')
    aliases = {
        'train': ('train', 'train_groups', 'orig_train_ids'),
        'validation': ('validation', 'val', 'validation_groups', 'orig_val_ids'),
        'test': ('test', 'test_groups', 'orig_test_ids'),
    }
    allowed = {
        name for names in aliases.values() for name in names
    } | {'random_state', 'freeze_existing'}
    if set(value) - allowed:
        raise ContractError('training split manifest fields are invalid')

    def normalized(role: str) -> set[str]:
        names = aliases[role]
        present = [name for name in names if name in value]
        if len(present) > 1:
            raise ContractError(f'training split manifest has ambiguous {role} fields')
        raw: object = value[present[0]] if present else []
        if not isinstance(raw, list):
            raise ContractError('training split members must be lists')
        members: list[str] = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, (int, str)):
                raise ContractError('training split members must be integer or string identifiers')
            rendered = str(item).strip()
            if not rendered:
                raise ContractError('training split identifiers must be nonempty')
            members.append(rendered)
        if len(set(members)) != len(members):
            raise ContractError('training split identifiers must be unique')
        return set(members)

    train = normalized('train')
    validation = normalized('validation')
    test = normalized('test')
    if train & validation or train & test or validation & test:
        raise ContractError('training split groups overlap')
    if not train or not (validation or test):
        raise ContractError('training split manifest requires nonempty train and held-out groups')
    return train, validation, test


def _resolve_manifest_groups(
    known_groups: set[str],
    train: set[str],
    validation: set[str],
    test: set[str],
) -> tuple[set[str], set[str]]:
    held_out = validation | test
    if known_groups - train - held_out:
        raise ContractError('training feature groups are absent from the sealed split manifest')
    explicit_train = known_groups & train
    explicit_held_out = known_groups & held_out
    if not explicit_train or not explicit_held_out:
        raise ContractError('sealed split manifest produces an empty observed train or held-out set')
    return explicit_train, explicit_held_out

@dataclass(frozen=True)
class XgbDevicePolicy:
    mode: str = 'cpu'

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {'cpu', 'cuda'}:
            raise ContractError('XGBoost device mode must be cpu or cuda')

@dataclass(frozen=True)
class SourceAlgorithmServices:
    dependencies: SourceAlgorithmDependencies
    report: Callable[..., None]
    approved_input_artifacts: tuple[SourceInputArtifact, ...]
    cpu_count: int
    thread_environment_setter: Callable[[str, str], None]
    device_policy: XgbDevicePolicy = XgbDevicePolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.approved_input_artifacts, tuple) or any(not isinstance(artifact, SourceInputArtifact) for artifact in self.approved_input_artifacts):
            raise ContractError('approved source input artifacts must be a typed tuple')
        if not callable(self.report) or not callable(self.thread_environment_setter):
            raise ContractError('source algorithm callbacks must be callable')
        if not isinstance(self.device_policy, XgbDevicePolicy):
            raise ContractError('source algorithm device policy must be typed')
        paths = tuple(artifact.path for artifact in self.approved_input_artifacts)
        if len(paths) != len(set(paths)):
            raise ContractError('approved source input artifact paths must be unique')
        if isinstance(self.cpu_count, bool) or not isinstance(self.cpu_count, int) or self.cpu_count < 1:
            raise ContractError('source algorithm CPU count must be a positive integer')

    def preflight_input_artifact(self, artifact: SourceInputArtifact, policy: ExecutionPolicy) -> Path:
        require_authorized(policy)
        if artifact not in self.approved_input_artifacts:
            raise ContractError('source input artifact is not explicitly approved')
        candidate = artifact.path
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ContractError('source input artifact is unavailable') from exc
        if resolved != candidate or candidate.is_symlink():
            raise ContractError('source input artifact and its path components must not be symlinks')
        try:
            status = candidate.lstat()
        except OSError as exc:
            raise ContractError('source input artifact is unavailable') from exc
        if not S_ISREG(status.st_mode):
            raise ContractError('source input artifact must be a regular file')
        actual = sha256()
        try:
            with candidate.open('rb') as stream:
                opened = fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size) != (
                    status.st_dev, status.st_ino, status.st_mode, status.st_size
                ):
                    raise ContractError('source input artifact changed before verification')
                while chunk := stream.read(1024 * 1024):
                    actual.update(chunk)
                completed = fstat(stream.fileno())
        except OSError as exc:
            raise ContractError('source input artifact could not be verified') from exc
        try:
            final = candidate.lstat()
        except OSError as exc:
            raise ContractError('source input artifact changed after verification') from exc
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns
        )
        if identity(status) != identity(opened) or identity(opened) != identity(completed) or identity(completed) != identity(final):
            raise ContractError('source input artifact changed during verification')
        if actual.hexdigest() != artifact.sha256:
            raise ContractError('source input artifact SHA-256 mismatch')
        return candidate

    def set_thread_environment(self, name: str, value: str, policy: ExecutionPolicy) -> None:
        require_authorized(policy)
        if name not in ALLOWED_THREAD_ENVIRONMENT_KEYS:
            raise ContractError('source algorithm requested an unreviewed thread environment key')
        rendered = str(value)
        if not rendered.isdigit() or int(rendered) < 1:
            raise ContractError('thread environment value must be a positive integer')
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
    metrics: Path
    confusion_matrix: Path
    dropped_columns: Path
    config: Path
    history: Path
    progress: Path
    best_parameters: Path

@dataclass(frozen=True)
class TrainingContinuationState:
    training_features: SourceInputArtifact
    split_manifest: SourceInputArtifact
    previous_tile_predictions: SourceInputArtifact
    previous_threshold: SourceInputArtifact
    df: TableValue
    csv_used: Path
    csv_path: Path
    name_col: str
    groups: ArrayValue
    y: ArrayValue
    num_cols: tuple[str, ...]
    dropped_cols: tuple[str, ...]
    X_train: TableValue
    X_test: TableValue
    y_train: ArrayValue
    y_test: ArrayValue
    groups_train: ArrayValue
    groups_test: ArrayValue
    review_weights_train: ArrayValue
    test_idx: ArrayValue
    round_tag: str
    model_name: str
    artifacts: TrainingArtifacts
    xgb_kwargs: Mapping[str, object]
    xgb_base: EstimatorValue
    build_smote: Callable[..., object]
    fit_with_maybe_callbacks: Callable[..., object]
    timer_factory: Callable[[str], AbstractContextManager[object]]
    grid_n_jobs: int
    device_mode: str
    test_size: float
    random_state: int
    n_splits_cv: int
    positive_label: int
    top_features_to_report: int
    border_pixels: int
    tile_size: int
    tiny_minimum_width_height: int
    tiny_maximum_area: int
    border_weight: float
    uncertainty_margin: float
    uncertainty_multiplier: float
    drop_tiny_in_train: bool
    drop_border_from_test: bool
    early_stop_validation_fraction: float
    large_estimator_count: int
    use_review_weighting: bool
    correct_review_weight: float
    wrong_review_weight: float
    use_review_tag_weights: bool
    review_tag_weights: tuple[tuple[str, float], ...]
    drop_skip_rows: bool
    augmentation_config_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'num_cols', tuple(self.num_cols))
        object.__setattr__(self, 'dropped_cols', tuple(self.dropped_cols))
        object.__setattr__(self, 'xgb_kwargs', MappingProxyType(dict(self.xgb_kwargs)))
        object.__setattr__(self, 'review_tag_weights', tuple(self.review_tag_weights))
        if self.grid_n_jobs < 1:
            raise ContractError('grid_n_jobs must be positive')
        if self.device_mode not in {'cpu', 'cuda'}:
            raise ContractError('continuation device mode must be cpu or cuda')
        if self.device_mode == 'cpu' and self.xgb_kwargs.get('device', 'cpu') != 'cpu':
            raise ContractError('CPU continuation cannot carry CUDA XGBoost settings')
        parsed = json.loads(self.augmentation_config_json)
        if not isinstance(parsed, dict):
            raise ContractError('augmentation configuration must encode an object')

    def required_bindings(self) -> Mapping[str, object]:
        """Expose the exact Cell3A-to-Cell3B predecessor contract immutably."""

        return MappingProxyType({
            'df': self.df,
            'CSV_USED': self.csv_used,
            'CSV_PATH': self.csv_path,
            'NAME_COL': self.name_col,
            'groups': self.groups,
            'y': self.y,
            'num_cols': self.num_cols,
            'dropped_cols': self.dropped_cols,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'groups_train': self.groups_train,
            'groups_test': self.groups_test,
            'review_weights_train': self.review_weights_train,
            'ROUND_TAG': self.round_tag,
            'MODEL_NAME': self.model_name,
            'MODEL_PATH': self.artifacts.model,
            'THRESH_JSON_PATH': self.artifacts.threshold,
            'FI_CSV_PATH': self.artifacts.feature_importances,
            'TILE_PREDS_CSV_PATH': self.artifacts.tile_predictions,
            'IMG_PREDS_CSV_PATH': self.artifacts.image_predictions,
            'PR_CURVE_CSV_PATH': self.artifacts.precision_recall_curve,
            'ROC_CURVE_CSV_PATH': self.artifacts.roc_curve,
            'METRICS_JSON_PATH': self.artifacts.metrics,
            'CONF_MAT_CSV_PATH': self.artifacts.confusion_matrix,
            'DROPPED_COLS_JSON': self.artifacts.dropped_columns,
            'CONFIG_JSON_PATH': self.artifacts.config,
            'HISTORY_CSV_PATH': self.artifacts.history,
            'PROGRESS_PNG_PATH': self.artifacts.progress,
            'xgb_kwargs': self.xgb_kwargs,
            'xgb_base': self.xgb_base,
            'build_smote': self.build_smote,
            'fit_with_maybe_callbacks': self.fit_with_maybe_callbacks,
            'Timer': self.timer_factory,
        })

@dataclass(frozen=True)
class SourceInputs_0001_0005:
    output_root: Path
    training_features: SourceInputArtifact
    split_manifest: SourceInputArtifact
    previous_tile_predictions: SourceInputArtifact
    previous_threshold: SourceInputArtifact
    model_name: str = 'cj_ultra_tilesafe_xgb_r19_hybrid.pkl'
    model_directory_tag: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path) or not self.output_root.is_absolute():
            raise ContractError('training output root must be absolute')
        if any(not isinstance(artifact, SourceInputArtifact) for artifact in (self.training_features, self.split_manifest, self.previous_tile_predictions, self.previous_threshold)):
            raise ContractError('NB0001 setup inputs must be typed source input artifacts')
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ContractError('model name must be nonempty')
        if not isinstance(self.model_directory_tag, str):
            raise ContractError('model directory tag must be a string')
        if Path(self.model_name).name != self.model_name or self.model_name in {'.', '..'}:
            raise ContractError('model name must be a single filename')
        if self.model_directory_tag and (
            Path(self.model_directory_tag).name != self.model_directory_tag
            or self.model_directory_tag in {'.', '..'}
        ):
            raise ContractError('model directory tag must be a single directory name')

@dataclass(frozen=True)
class SourceResult_0001_0005:
    artifacts: TrainingArtifacts
    continuation: TrainingContinuationState

@dataclass(frozen=True)
class SourceInputs_0001_0006:
    continuation: TrainingContinuationState
    previous_best_parameters: SourceInputArtifact | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.continuation, TrainingContinuationState):
            raise ContractError('NB0001 fit requires the typed setup continuation')
        if self.previous_best_parameters is not None and not isinstance(self.previous_best_parameters, SourceInputArtifact):
            raise ContractError('previous best parameters must be a typed artifact')

@dataclass(frozen=True)
class SourceResult_0001_0006:
    artifacts: TrainingArtifacts
    selected_threshold: float
    evaluation_split: str
    selected_feature_count: int
    used_trees: int


def _artifacts(ROUND_DIR: Path, MODEL_PATH: Path, THRESH_JSON_PATH: Path, FI_CSV_PATH: Path, TILE_PREDS_CSV_PATH: Path, IMG_PREDS_CSV_PATH: Path, PR_CURVE_CSV_PATH: Path, ROC_CURVE_CSV_PATH: Path, METRICS_JSON_PATH: Path, CONF_MAT_CSV_PATH: Path, DROPPED_COLS_JSON: Path, CONFIG_JSON_PATH: Path, HISTORY_CSV_PATH: Path, PROGRESS_PNG_PATH: Path) -> TrainingArtifacts:
    return TrainingArtifacts(ROUND_DIR, MODEL_PATH, THRESH_JSON_PATH, FI_CSV_PATH, TILE_PREDS_CSV_PATH, IMG_PREDS_CSV_PATH, PR_CURVE_CSV_PATH, ROC_CURVE_CSV_PATH, METRICS_JSON_PATH, CONF_MAT_CSV_PATH, DROPPED_COLS_JSON, CONFIG_JSON_PATH, HISTORY_CSV_PATH, PROGRESS_PNG_PATH, ROUND_DIR / 'best_params_used.json')


# SOURCE_CELL: NB-LIVE-0001-C0005
# SOURCE_STATEMENT_MAP: sanitized-body -> source_algorithm_0001_0005
def source_algorithm_0001_0005(inputs: SourceInputs_0001_0005, services: SourceAlgorithmServices, policy: ExecutionPolicy) -> SourceResult_0001_0005:
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
    # SOURCE_CELL: NB-LIVE-0001-C0005
    # SOURCE_STATEMENT_MAP: SafeSMOTE/build_smote_safe -> stable persisted estimator symbols
    from .safe_smote import SafeSMOTE, build_smote_safe

    np = services.dependencies.np
    pd = services.dependencies.pd
    ParserError = services.dependencies.ParserError
    SimpleImputer = services.dependencies.SimpleImputer
    tqdm = services.dependencies.tqdm
    joblib = services.dependencies.joblib
    SMOTE = services.dependencies.SMOTE
    ImbPipeline = services.dependencies.ImbPipeline
    XGBClassifier = services.dependencies.XGBClassifier
    xgb = services.dependencies.xgb
    version = services.dependencies.version
    TrainingCallback = services.dependencies.TrainingCallback
    import os, re, json, warnings, time
    from pathlib import Path
    from contextlib import contextmanager

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
    CSV_PATH = _training_features
    OUT_DIR_ROOT = inputs.output_root
    MODEL_NAME = inputs.model_name
    PREV_TILE_PREDS_PATH = _previous_tile_predictions
    PREV_THRESH_JSON_PATH = _previous_threshold
    services.report('[CSV_PATH]', CSV_PATH)
    if not CSV_PATH.exists():
        raise FileNotFoundError(f'CSV_PATH not found: {CSV_PATH}')
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_SPLITS_CV = 5
    POS_LABEL = 1
    TOPK_FEATURES_TO_PRINT = 30
    BORDER_PX = 4
    TILE_SIZE = 512
    TINY_MIN_WH = 32
    TINY_MAX_AREA = 1500
    BORDER_WEIGHT = 0.5
    UNCERT_MARGIN = 0.05
    UNCERT_MULT = 0.7
    DROP_TINY_IN_TRAIN = False
    DROP_BORDER_FROM_TEST = True
    EARLY_STOP_ROUNDS = 50
    EARLY_STOP_VAL_FRAC = 0.3
    N_ESTIMATORS_BIG = 2000
    USE_REVIEW_WEIGHTING = True
    WEIGHT_CORRECT = 0.7
    WEIGHT_WRONG = 1.0
    AUTO_DETECT_PREV_ROUND = False
    WEIGHT_MAP = {'accept': 1.0, 'flip': 1.0, 'sus_accept': 0.4, 'sus_flip': 0.4, 'skip': 0.0}
    USE_REVIEW_TAG_WEIGHTS = True
    DROP_SKIP_ROWS = True
    AUG_CFG = {'apply_in_3A': True, 'exclude_discrete_max_nunique': 10, 'mixup': {'enabled': True, 'alpha': 0.2, 'mult': 0.5}, 'dropout': {'enabled': True, 'p': 0.05, 'mult': 1.0, 'strategy': 'median_by_class'}, 'jitter': {'enabled': True, 'mult': 0.5, 'sigma': 0.01, 'per_feature': True, 'clip_q': (0.001, 0.999)}, 'shuffle_rows': True}
    services.set_thread_environment('OMP_NUM_THREADS', '1', policy)
    services.set_thread_environment('MKL_NUM_THREADS', '1', policy)
    prepare_source_output_directory(OUT_DIR_ROOT)
    MODEL_DIR_TAG = inputs.model_directory_tag.strip()
    if not MODEL_DIR_TAG:
        m_dir = re.search('(r\\d+_[A-Za-z0-9_-]+)', str(MODEL_NAME), flags=re.IGNORECASE)
        MODEL_DIR_TAG = m_dir.group(1) if m_dir else ''
    m_r = re.search('r(\\d+)(?:\\D|$)', str(MODEL_DIR_TAG) or str(MODEL_NAME), flags=re.IGNORECASE)
    ROUND_TAG = f'r{m_r.group(1)}' if m_r else 'r1'
    if not MODEL_DIR_TAG:
        MODEL_DIR_TAG = ROUND_TAG
    ROUND_DIR = OUT_DIR_ROOT / MODEL_DIR_TAG
    prepare_source_output_directory(ROUND_DIR)
    services.report('[ROUND] MODEL_NAME   :', MODEL_NAME)
    services.report('[ROUND] MODEL_DIR_TAG:', MODEL_DIR_TAG)
    services.report('[ROUND] ROUND_TAG    :', ROUND_TAG)
    services.report('[ROUND] ROUND_DIR    :', ROUND_DIR)
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
    NORMALIZED_FEATURE_PATH = ROUND_DIR / _training_features.with_suffix('.normalized.csv').name
    if NORMALIZED_FEATURE_PATH.parent != ROUND_DIR:
        raise ContractError('normalized feature output must remain directly under the round directory')

    def normalize_features_csv_add_review_tag(csv_in: Path, csv_out: Optional[Path]=None) -> Path:
        """
        Normalize a potentially mixed-schema features CSV:
          - header may NOT have review_tag, while some rows have an extra field (tag) between reviewed and class_id.
          - output will ALWAYS contain review_tag column inserted right after reviewed.
          - rows with no tag will get blank review_tag.
        Uses csv module; safe for large files and mismatched field counts.
        """
        import csv as _csv
        if csv_out is None:
            csv_out = NORMALIZED_FEATURE_PATH
        if csv_out != NORMALIZED_FEATURE_PATH or csv_out.parent != ROUND_DIR:
            raise ContractError('normalized feature output escaped its declared round directory')
        with csv_in.open('r', newline='', encoding='utf-8', errors='replace') as fin:
            rdr = _csv.reader(fin)
            header = next(rdr, None)
            if not header:
                raise RuntimeError('CSV is empty or missing header.')
            header = [h.strip() for h in header]
            has_review_tag = 'review_tag' in header
            if has_review_tag:
                tag_idx = header.index('review_tag')
                out_header = header
                expected_n = len(out_header)
            else:
                if 'reviewed' in header:
                    tag_idx = header.index('reviewed') + 1
                elif 'class_id' in header:
                    tag_idx = header.index('class_id')
                else:
                    tag_idx = len(header)
                out_header = header[:tag_idx] + ['review_tag'] + header[tag_idx:]
                expected_n = len(out_header)
            with csv_out.open('w', newline='', encoding='utf-8') as fout:
                wtr = _csv.writer(fout)
                wtr.writerow(out_header)
                in_n = len(header)
                for row in rdr:
                    if not row:
                        continue
                    if not has_review_tag:
                        if len(row) == in_n:
                            row = row[:tag_idx] + [''] + row[tag_idx:]
                        elif len(row) == in_n + 1:
                            row = row[:]
                        elif len(row) < in_n:
                            row = row + [''] * (in_n - len(row))
                            row = row[:tag_idx] + [''] + row[tag_idx:]
                        else:
                            row = row[:in_n] + [','.join(row[in_n:])]
                            row = row[:tag_idx] + [''] + row[tag_idx:]
                    elif len(row) < expected_n:
                        row = row + [''] * (expected_n - len(row))
                    elif len(row) > expected_n:
                        row = row[:expected_n - 1] + [','.join(row[expected_n - 1:])]
                    if len(row) < expected_n:
                        row = row + [''] * (expected_n - len(row))
                    elif len(row) > expected_n:
                        row = row[:expected_n - 1] + [','.join(row[expected_n - 1:])]
                    wtr.writerow(row)
        services.report(f'[CSV] normalized -> {csv_out}')
        return csv_out

    def read_features_csv_robust(csv_path: Path) -> Tuple[pd.DataFrame, Path]:
        """
        Try pandas read_csv; if ParserError due to mixed columns, auto-normalize then read.
        Also sanitizes review_tag and enforces minimal required columns.
        """
        try:
            df = pd.read_csv(csv_path, low_memory=False, keep_default_na=False)
            used = csv_path
        except ParserError as e:
            services.report(f'[PANDAS ParserError] {e}')
            norm = normalize_features_csv_add_review_tag(
                csv_path,
                NORMALIZED_FEATURE_PATH,
            )
            if norm != NORMALIZED_FEATURE_PATH or norm.parent != ROUND_DIR:
                raise ContractError('normalizer returned an undeclared feature output path')
            df = pd.read_csv(norm, low_memory=False, keep_default_na=False)
            used = norm
        if 'review_tag' not in df.columns:
            df['review_tag'] = ''
        df['review_tag'] = df['review_tag'].astype(str).str.strip()
        df.loc[df['review_tag'].str.lower().isin({'nan', 'none'}), 'review_tag'] = ''
        if 'reviewed' in df.columns:
            df['reviewed'] = pd.to_numeric(df['reviewed'], errors='coerce').fillna(0).astype(int)
        else:
            df['reviewed'] = 0
        if 'label' not in df.columns:
            raise RuntimeError('CSV missing required column: label')
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        if 'scale' in df.columns:
            df['scale'] = pd.to_numeric(df['scale'], errors='coerce')
        return (df, used)
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

    def leakage_scan_vectorized(df: pd.DataFrame, y: np.ndarray, drop_always: set, leak_patterns: List[str], corr_thr: float=0.999):
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

    def _ensure_numeric_series(s, default=0.0):
        return pd.to_numeric(s, errors='coerce').fillna(default).astype(float)
    def border_and_tiny_masks_from_df_auto(df, name_col: str, tile_size_default: int=512, border_px: int=4, tiny_min_wh: int=32, tiny_max_area: int=1500):
        bx = _ensure_numeric_series(df.get('bbox_x', 0))
        by = _ensure_numeric_series(df.get('bbox_y', 0))
        bw = _ensure_numeric_series(df.get('bbox_w', 0))
        bh = _ensure_numeric_series(df.get('bbox_h', 0))
        if {'tile_w', 'tile_h'}.issubset(df.columns):
            tw = _ensure_numeric_series(df['tile_w']).clip(lower=1.0)
            th = _ensure_numeric_series(df['tile_h']).clip(lower=1.0)
        else:
            per_tile = pd.DataFrame({name_col: df[name_col].astype(str), 'mx_w': bx + bw, 'mx_h': by + bh}).groupby(name_col, sort=False).agg({'mx_w': 'max', 'mx_h': 'max'})
            tw = df[name_col].astype(str).map(per_tile['mx_w']).fillna(float(tile_size_default)).clip(lower=1.0)
            th = df[name_col].astype(str).map(per_tile['mx_h']).fillna(float(tile_size_default)).clip(lower=1.0)
        border_frac = float(border_px) / float(tile_size_default)
        tiny_min_frac = float(tiny_min_wh) / float(tile_size_default)
        tiny_area_frac = float(tiny_max_area) / float(tile_size_default * tile_size_default)
        is_border = ((bx <= border_frac * tw) | (by <= border_frac * th) | (bx + bw >= (1.0 - border_frac) * tw) | (by + bh >= (1.0 - border_frac) * th)).to_numpy()
        is_tiny = ((bw < tiny_min_frac * tw) | (bh < tiny_min_frac * th) | (bw * bh < tiny_area_frac * (tw * th))).to_numpy()
        return (is_border, is_tiny)
    with Timer('read_features_csv_robust'):
        df, CSV_USED = read_features_csv_robust(CSV_PATH)
    services.report('[CSV_USED]', CSV_USED)
    services.report('[DF]', df.shape, '| cols:', len(df.columns))
    USE_SCALE_FEATURE = False
    if 'scale' in df.columns:
        df['scale'] = pd.to_numeric(df['scale'], errors='coerce')
    else:
        USE_SCALE_FEATURE = False

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
    drop_always = {NAME_COL, 'image', 'file_name', 'filename', 'file', 'id', 'ann_id', 'label', 'group', 'area_px', 'perim_sqrt', 'reviewed', 'review_tag', 'class_id', 'class_name'}
    if not USE_SCALE_FEATURE:
        drop_always.add('scale')
    LEAK_PATTERNS = ['^class(_?id)?$', '^category(_?id)?$', '^cat(_?id)?$', '^cid$', '^name$', '^target$', '^y$', '.*(_|^)class(_|$).*', '.*(_|^)category(_|$).*', '.*(_|^)catname(_|$).*', '.*(_|^)classname(_|$).*']
    num_cols, dropped_cols = leakage_scan_vectorized(df, y, drop_always, LEAK_PATTERNS, corr_thr=0.999)
    if dropped_cols:
        services.report('[INFO] dropping potential leakage columns:', dropped_cols[:50], '...' if len(dropped_cols) > 50 else '')
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
    services.report(f'[FEATURES] final numeric columns: {len(num_cols)}')
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
        test_groups = set(best_sel) if best_sel is not None else set()
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
    if not len(train_idx) or not len(test_idx):
        raise ContractError('sealed split manifest produced an empty training or held-out set')
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
    _is_b_tr, _is_t_tr = border_and_tiny_masks_from_df_auto(df.iloc[train_idx], name_col=NAME_COL, tile_size_default=TILE_SIZE, border_px=BORDER_PX, tiny_min_wh=TINY_MIN_WH, tiny_max_area=TINY_MAX_AREA)
    _is_b_te, _ = border_and_tiny_masks_from_df_auto(df.iloc[test_idx], name_col=NAME_COL, tile_size_default=TILE_SIZE, border_px=BORDER_PX, tiny_min_wh=TINY_MIN_WH, tiny_max_area=TINY_MAX_AREA)
    if DROP_TINY_IN_TRAIN and _is_t_tr.any():
        _drop_tr = _is_t_tr & (y_train == 0)
        services.report(f'[FILTER] drop tiny NEG in TRAIN: {int(_drop_tr.sum())} rows')
        _keep_tr = ~_drop_tr
        X_train = X_train.iloc[_keep_tr]
        y_train = y_train[_keep_tr]
        groups_train = groups_train[_keep_tr]
        train_idx = train_idx[_keep_tr]
        _is_b_tr = _is_b_tr[_keep_tr]
        _is_t_tr = _is_t_tr[_keep_tr]
    if DROP_BORDER_FROM_TEST and _is_b_te.any():
        _keep_te = ~_is_b_te
        services.report(f'[FILTER] drop border in TEST: {int(_is_b_te.sum())} rows')
        X_test = X_test.iloc[_keep_te]
        y_test = y_test[_keep_te]
        groups_test = groups_test[_keep_te]
        test_idx = test_idx[_keep_te]
    frac = len(test_idx) / (len(train_idx) + len(test_idx))
    services.report(f'Train size: {len(train_idx)}, Test size: {len(test_idx)} | Split={split_mode} | tiles_test_frac={frac:.3f} | target={TEST_SIZE:.3f}')

    def _extract_round_num_from_name(s: str) -> Optional[int]:
        m = re.search('r(\\d+)(?:\\D|$)', str(s), flags=re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _find_prev_round_dir(root: Path, current_round_tag: str) -> Optional[Path]:
        raise ContractError('automatic previous-round discovery is disabled; explicit verified artifacts are required')

    def _load_prev_preds_and_thr(prev_preds_path: Path, prev_thr_path: Path, name_col: str) -> Optional[Dict[str, Tuple[int, float]]]:
        try:
            prev = pd.read_csv(prev_preds_path, low_memory=False, keep_default_na=False)
            with open(prev_thr_path, 'r', encoding='utf-8') as f:
                thr_obj = json.load(f)
            thr = float(thr_obj.get('threshold', thr_obj.get('thr', 0.5)))
            if name_col not in prev.columns or 'proba' not in prev.columns:
                return None
            prev2 = prev[[name_col, 'proba']].copy()
            prev2['proba'] = pd.to_numeric(prev2['proba'], errors='coerce')
            prev2 = prev2.groupby(name_col, as_index=False, sort=False)['proba'].max()
            prev2['pred'] = (prev2['proba'] >= thr).astype(int)
            mapping = {str(n): (int(p), float(s)) for n, p, s in zip(prev2[name_col], prev2['pred'], prev2['proba'])}
            mapping['__thr__'] = (0, float(thr))
            return mapping
        except Exception as e:
            services.report(f'[PREV] load failed: {e}')
            return None

    def build_review_weights_for_train(df: pd.DataFrame, name_col: str, train_idx: np.ndarray, y_train: np.ndarray, round_dir_root: Path, current_round_tag: str, weight_correct: float=0.7, weight_wrong: float=1.0) -> np.ndarray:
        prev_preds_path = Path(PREV_TILE_PREDS_PATH) if PREV_TILE_PREDS_PATH else None
        prev_thr_path = Path(PREV_THRESH_JSON_PATH) if PREV_THRESH_JSON_PATH else None
        mapping = None
        if prev_preds_path is not None and prev_thr_path is not None and prev_preds_path.exists() and prev_thr_path.exists():
            mapping = _load_prev_preds_and_thr(prev_preds_path, prev_thr_path, name_col)
        w = np.full(len(train_idx), float(weight_wrong), dtype=np.float32)
        if not mapping:
            return w
        thr = float(mapping.get('__thr__', (0, 0.5))[1])
        names_train = df.iloc[train_idx][name_col].astype(str).values
        preds_prev = np.full(len(train_idx), -1, dtype=int)
        prob_prev = np.full(len(train_idx), np.nan, dtype=float)
        for i, n in enumerate(names_train):
            v = mapping.get(str(n), None)
            if v is None:
                continue
            preds_prev[i] = int(v[0])
            prob_prev[i] = float(v[1])
        known = preds_prev >= 0
        if known.any():
            correct = preds_prev[known] == y_train[known]
            w_sub = np.where(correct, float(weight_correct), float(weight_wrong)).astype(np.float32)
            close = np.abs(prob_prev[known] - thr) <= float(UNCERT_MARGIN)
            w_sub = w_sub * np.where(close, float(UNCERT_MULT), 1.0).astype(np.float32)
            w[known] = w_sub
        return w
    review_weights_train = np.ones_like(y_train, dtype=np.float32)
    review_mask_train = np.zeros_like(y_train, dtype=bool)
    if USE_REVIEW_WEIGHTING:
        review_weights_train = build_review_weights_for_train(df=df, name_col=NAME_COL, train_idx=train_idx, y_train=y_train, round_dir_root=OUT_DIR_ROOT, current_round_tag=ROUND_TAG, weight_correct=WEIGHT_CORRECT, weight_wrong=WEIGHT_WRONG)
        services.report(f'[WEIGHTS] prev-round weights built. mean={review_weights_train.mean():.3f}')
        review_mask_train = df.iloc[train_idx]['reviewed'].astype(int).values == 1
        review_weights_train = np.where(review_mask_train, review_weights_train, 1.0).astype(np.float32)
        services.report(f'[WEIGHTS] review-only mask applied. frac_reviewed={review_mask_train.mean():.3f}')
    if _is_b_tr.any() and review_mask_train.any():
        review_weights_train = review_weights_train * np.where(review_mask_train & _is_b_tr, float(BORDER_WEIGHT), 1.0).astype(np.float32)
        services.report(f'[WEIGHTS] applied border penalty on reviewed rows (x{BORDER_WEIGHT}).')
    TINY_NEG_WEIGHT = 0.6
    review_weights_train = review_weights_train * np.where(_is_t_tr & (y_train == 0), float(TINY_NEG_WEIGHT), 1.0).astype(np.float32)
    services.report(f'[WEIGHTS] applied tiny-neg weight x{TINY_NEG_WEIGHT}')

    def _get_review_tag_series(df_slice: pd.DataFrame) -> pd.Series:
        if 'review_tag' not in df_slice.columns:
            return pd.Series([''] * len(df_slice), index=df_slice.index)
        s = df_slice['review_tag'].astype(str).str.strip().str.lower()
        s = s.where(~s.isin({'nan', 'none'}), '')
        return s
    if USE_REVIEW_TAG_WEIGHTS:
        tags_tr = _get_review_tag_series(df.iloc[train_idx])
        mult_tr = tags_tr.map(lambda t: float(WEIGHT_MAP.get(t, 1.0))).astype(np.float32).to_numpy()
        if DROP_SKIP_ROWS:
            keep = mult_tr > 0.0
            if (~keep).any():
                n_drop = int((~keep).sum())
                services.report(f'[FILTER] drop skip-tag rows in TRAIN: {n_drop}')
                X_train = X_train.iloc[keep]
                y_train = y_train[keep]
                groups_train = groups_train[keep]
                review_weights_train = review_weights_train[keep]
                _is_b_tr = _is_b_tr[keep]
                _is_t_tr = _is_t_tr[keep]
                train_idx = train_idx[keep]
                mult_tr = mult_tr[keep]
        review_weights_train = review_weights_train.astype(np.float32) * mult_tr.astype(np.float32)
        services.report(f'[WEIGHTS] applied review_tag multipliers. mean={review_weights_train.mean():.3f}')
        tags_te = _get_review_tag_series(df.iloc[test_idx])
        mult_te = tags_te.map(lambda t: float(WEIGHT_MAP.get(t, 1.0))).astype(np.float32).to_numpy()
        if DROP_SKIP_ROWS:
            keep = mult_te > 0.0
            if (~keep).any():
                n_drop = int((~keep).sum())
                services.report(f'[FILTER] drop skip-tag rows in TEST: {n_drop}')
                X_test = X_test.iloc[keep]
                y_test = y_test[keep]
                groups_test = groups_test[keep]
                test_idx = test_idx[keep]
                mult_te = mult_te[keep]

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
        return (np.vstack(Xc_list).astype(np.float32), np.concatenate(yc_list).astype(int), np.concatenate(wc_list).astype(np.float32), np.concatenate(gc_list))

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
        return (np.vstack([X, Xn]).astype(np.float32), np.concatenate([y, yn]).astype(int), np.concatenate([w, wn]).astype(np.float32), np.concatenate([g, gn]))

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
        return (np.vstack([X, Xn]).astype(np.float32), np.concatenate([y, yn]).astype(int), np.concatenate([w, wn]).astype(np.float32), np.concatenate([g, gn]))

    def apply_augs_in_3A(X_df, y_np, w_np, g_np, cfg, random_state=42):
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
            services.report(f'[AUG] mixup -> total {len(yo)} samples')
        if cfg['dropout']['enabled']:
            Xo, yo, wo, go = dropout_augment_tabular_with_groups(Xo, yo, wo, go, p=cfg['dropout']['p'], mult=cfg['dropout']['mult'], strategy=cfg['dropout']['strategy'], exclude_discrete_max_nunique=cfg['exclude_discrete_max_nunique'], rs=rs)
            services.report(f'[AUG] dropout -> total {len(yo)} samples')
        if cfg['jitter']['enabled']:
            Xo, yo, wo, go = jitter_augment_tabular_with_groups(Xo, yo, wo, go, cont_idx=cont_idx, mult=cfg['jitter']['mult'], sigma=cfg['jitter']['sigma'], per_feature=cfg['jitter']['per_feature'], clip_low=ql, clip_high=qh, rs=rs)
            services.report(f'[AUG] jitter -> total {len(yo)} samples')
        X_df_aug = pd.DataFrame(Xo, columns=X_df.columns)
        return (X_df_aug, yo, wo, go)
    X_train_preaug = X_train.copy()
    y_train_preaug = y_train.copy()
    groups_train_preaug = groups_train.copy()
    review_weights_train_preaug = review_weights_train.copy()
    if AUG_CFG.get('apply_in_3A', True):
        X_train, y_train, review_weights_train, groups_train = apply_augs_in_3A(X_df=X_train, y_np=y_train, w_np=review_weights_train, g_np=groups_train, cfg=AUG_CFG, random_state=RANDOM_STATE)
        services.report(f'[AUG] Done. Train now: {len(y_train)} rows (incl. augmented).')
    else:
        services.report('[AUG] Skipped in 3A (you can apply inside 3B).')

    class TQDMCallback(TrainingCallback):

        def __init__(self, total: int, desc: str='XGBoost'):
            self.total = int(total)
            self.desc = str(desc)
            self.pbar = None

        def before_training(self, model):
            self.pbar = tqdm(total=self.total, desc=self.desc, leave=False)
            return model

        def after_iteration(self, model, epoch: int, evals_log: Dict[str, Dict[str, float]]):
            if self.pbar is not None:
                self.pbar.update(1)
            return False

        def after_training(self, model):
            if self.pbar is not None:
                self.pbar.close()
                return model
    use_cuda = services.device_policy.mode == 'cuda'
    if use_cuda and services.dependencies.cupy is None:
        raise ContractError('CUDA mode requires an injected CuPy dependency')
    N_CPU = services.cpu_count
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
    smote_global = build_smote_safe(y_train, sampling_strategy=0.5, k_neighbors=5, random_state=RANDOM_STATE)
    pipe = ImbPipeline([('imp', SimpleImputer(strategy='median')), ('smote', smote_global if smote_global is not None else 'passthrough'), ('clf', xgb_base)])
    services.report(f'[CUDA] xgboost={xgb.__version__} | use_cuda={use_cuda} | tree_method={xgb_kwargs.get('tree_method')} | device={xgb_kwargs.get('device', 'cpu')}')
    services.report('[PIPE] ready. X_train:', getattr(X_train, 'shape', None), '| y_train:', y_train.shape, '| w_train:', review_weights_train.shape)

    artifacts = _artifacts(ROUND_DIR, MODEL_PATH, THRESH_JSON_PATH, FI_CSV_PATH, TILE_PREDS_CSV_PATH, IMG_PREDS_CSV_PATH, PR_CURVE_CSV_PATH, ROC_CURVE_CSV_PATH, METRICS_JSON_PATH, CONF_MAT_CSV_PATH, DROPPED_COLS_JSON, CONFIG_JSON_PATH, HISTORY_CSV_PATH, PROGRESS_PNG_PATH)
    continuation = TrainingContinuationState(
        inputs.training_features, inputs.split_manifest, inputs.previous_tile_predictions, inputs.previous_threshold,
        df, Path(CSV_USED), Path(CSV_PATH), NAME_COL, groups, y, tuple(num_cols), tuple(dropped_cols),
        X_train, X_test, y_train, y_test, groups_train, groups_test, review_weights_train, test_idx,
        ROUND_TAG, MODEL_NAME, artifacts, xgb_kwargs, xgb_base, build_smote, fit_with_maybe_callbacks,
        Timer, grid_n_jobs, services.device_policy.mode, TEST_SIZE, RANDOM_STATE, N_SPLITS_CV, POS_LABEL, TOPK_FEATURES_TO_PRINT,
        BORDER_PX, TILE_SIZE, TINY_MIN_WH, TINY_MAX_AREA, BORDER_WEIGHT, UNCERT_MARGIN, UNCERT_MULT,
        DROP_TINY_IN_TRAIN, DROP_BORDER_FROM_TEST, EARLY_STOP_VAL_FRAC, N_ESTIMATORS_BIG,
        USE_REVIEW_WEIGHTING, WEIGHT_CORRECT, WEIGHT_WRONG, USE_REVIEW_TAG_WEIGHTS,
        tuple((str(key), float(value)) for key, value in WEIGHT_MAP.items()), DROP_SKIP_ROWS,
        json.dumps(AUG_CFG, sort_keys=True, separators=(',', ':')),
    )
    return SourceResult_0001_0005(artifacts, continuation)


# SOURCE_CELL: NB-LIVE-0001-C0006
# SOURCE_STATEMENT_MAP: sanitized-body -> source_algorithm_0001_0006
def source_algorithm_0001_0006(inputs: SourceInputs_0001_0006, services: SourceAlgorithmServices, policy: ExecutionPolicy) -> SourceResult_0001_0006:
    require_authorized(policy)
    _previous_best_parameters = None
    if inputs.previous_best_parameters is not None:
        _previous_best_parameters = services.preflight_input_artifact(inputs.previous_best_parameters, policy)
    state = inputs.continuation
    if state.device_mode != services.device_policy.mode:
        raise ContractError('continuation and service device policies differ')
    use_cuda = services.device_policy.mode == 'cuda'
    if use_cuda and services.dependencies.cupy is None:
        raise ContractError('CUDA mode requires an injected CuPy dependency')
    np = services.dependencies.np
    pd = services.dependencies.pd
    joblib = services.dependencies.joblib
    plt = services.dependencies.plt
    SimpleImputer = services.dependencies.SimpleImputer
    average_precision_score = services.dependencies.average_precision_score
    roc_auc_score = services.dependencies.roc_auc_score
    precision_recall_curve = services.dependencies.precision_recall_curve
    roc_curve = services.dependencies.roc_curve
    confusion_matrix = services.dependencies.confusion_matrix
    GroupKFold = services.dependencies.GroupKFold
    GroupShuffleSplit = services.dependencies.GroupShuffleSplit
    ParameterSampler = services.dependencies.ParameterSampler
    ImbPipeline = services.dependencies.ImbPipeline
    XGBClassifier = services.dependencies.XGBClassifier
    xgb = services.dependencies.xgb
    randint = services.dependencies.randint
    loguniform = services.dependencies.loguniform
    uniform = services.dependencies.uniform
    tqdm = services.dependencies.tqdm
    import os, json, time, math, warnings
    from pathlib import Path
    warnings.filterwarnings('ignore', category=UserWarning)
    _required = [
        'df', 'CSV_USED', 'CSV_PATH', 'NAME_COL', 'groups', 'y',
        'num_cols', 'dropped_cols',
        'X_train', 'X_test', 'y_train', 'y_test', 'groups_train', 'groups_test',
        'review_weights_train',
        'ROUND_TAG', 'MODEL_NAME', 'MODEL_PATH', 'THRESH_JSON_PATH',
        'FI_CSV_PATH', 'TILE_PREDS_CSV_PATH', 'IMG_PREDS_CSV_PATH',
        'PR_CURVE_CSV_PATH', 'ROC_CURVE_CSV_PATH', 'METRICS_JSON_PATH', 'CONF_MAT_CSV_PATH',
        'DROPPED_COLS_JSON', 'CONFIG_JSON_PATH', 'HISTORY_CSV_PATH', 'PROGRESS_PNG_PATH',
        'xgb_kwargs', 'xgb_base', 'build_smote', 'fit_with_maybe_callbacks', 'Timer',
    ]
    missing = [key for key in _required if key not in state.required_bindings()]
    if missing:
        raise ContractError('typed setup continuation is missing required bindings')
    df = state.df
    CSV_USED = state.csv_used
    CSV_PATH = state.csv_path
    NAME_COL = state.name_col
    groups = state.groups
    y = state.y
    num_cols = list(state.num_cols)
    dropped_cols = list(state.dropped_cols)
    X_train = state.X_train
    X_test = state.X_test
    y_train = np.asarray(state.y_train).astype(int)
    y_test = np.asarray(state.y_test).astype(int)
    groups_train = np.asarray(state.groups_train)
    groups_test = np.asarray(state.groups_test)
    review_weights_train = np.asarray(state.review_weights_train).astype(np.float32)
    test_idx = state.test_idx if state.test_idx is not None else None
    ROUND_TAG = str(state.round_tag)
    MODEL_NAME = str(state.model_name)
    MODEL_PATH = Path(state.artifacts.model)
    THRESH_JSON_PATH = Path(state.artifacts.threshold)
    FI_CSV_PATH = Path(state.artifacts.feature_importances)
    TILE_PREDS_CSV_PATH = Path(state.artifacts.tile_predictions)
    IMG_PREDS_CSV_PATH = Path(state.artifacts.image_predictions)
    PR_CURVE_CSV_PATH = Path(state.artifacts.precision_recall_curve)
    ROC_CURVE_CSV_PATH = Path(state.artifacts.roc_curve)
    METRICS_JSON_PATH = Path(state.artifacts.metrics)
    CONF_MAT_CSV_PATH = Path(state.artifacts.confusion_matrix)
    DROPPED_COLS_JSON = Path(state.artifacts.dropped_columns)
    CONFIG_JSON_PATH = Path(state.artifacts.config)
    HISTORY_CSV_PATH = Path(state.artifacts.history)
    PROGRESS_PNG_PATH = Path(state.artifacts.progress)
    xgb_kwargs = dict(state.xgb_kwargs)
    xgb_base = state.xgb_base
    build_smote = state.build_smote
    fit_with_maybe_callbacks = state.fit_with_maybe_callbacks
    Timer = state.timer_factory
    RANDOM_STATE = int(state.random_state)
    POS_LABEL = int(state.positive_label)
    TEST_SIZE = state.test_size
    N_SPLITS_CV = int(state.n_splits_cv)
    TOPK_FEATURES_TO_PRINT = int(state.top_features_to_report)
    BORDER_PX = state.border_pixels
    TILE_SIZE = state.tile_size
    TINY_MIN_WH = state.tiny_minimum_width_height
    TINY_MAX_AREA = state.tiny_maximum_area
    BORDER_WEIGHT = state.border_weight
    UNCERT_MARGIN = state.uncertainty_margin
    UNCERT_MULT = state.uncertainty_multiplier
    DROP_TINY_IN_TRAIN = state.drop_tiny_in_train
    DROP_BORDER_FROM_TEST = state.drop_border_from_test
    EARLY_STOP_ROUNDS = 30
    EARLY_STOP_VAL_FRAC = float(state.early_stop_validation_fraction)
    N_ESTIMATORS_BIG = int(state.large_estimator_count)
    USE_REVIEW_WEIGHTING = state.use_review_weighting
    WEIGHT_CORRECT = state.correct_review_weight
    WEIGHT_WRONG = state.wrong_review_weight
    AUTO_DETECT_PREV_ROUND = False
    USE_REVIEW_TAG_WEIGHTS = state.use_review_tag_weights
    WEIGHT_MAP = dict(state.review_tag_weights)
    DROP_SKIP_ROWS = state.drop_skip_rows
    AUG_CFG = json.loads(state.augmentation_config_json)
    grid_n_jobs = int(state.grid_n_jobs)
    FORCE_HPARAM_SEARCH = False
    HPARAM_N_ITER = 30
    USE_WEIGHTED_SEARCH = True
    SEARCH_EVERY_N_ROUNDS = 500
    N_EST_SEARCH = 300

    def ensure_numeric_df(d: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = d.copy()
        for c in cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors='coerce')
        return out
    X_train = ensure_numeric_df(X_train, num_cols)
    X_test = ensure_numeric_df(X_test, num_cols)

    def build_cv_pairs(groups_vec: np.ndarray, y_vec: np.ndarray, target_k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        uniq_g = np.unique(groups_vec)
        if len(uniq_g) < 2:
            return []
        k = min(int(target_k), len(uniq_g))
        if k < 2:
            return []
        gkf = GroupKFold(n_splits=k)
        pairs = []
        for tr_idx, va_idx in gkf.split(np.zeros(len(y_vec)), y_vec, groups_vec):
            if len(np.unique(y_vec[tr_idx])) < 2 or len(np.unique(y_vec[va_idx])) < 2:
                continue
            pairs.append((tr_idx, va_idx))
        return pairs
    cv_pairs = build_cv_pairs(groups_train, y_train, N_SPLITS_CV)
    if not cv_pairs:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE + 999)
        tr_idx, va_idx = next(gss.split(np.zeros(len(y_train)), y_train, groups_train))
        cv_pairs = [(tr_idx, va_idx)]
    services.report(f'[CV] folds: {len(cv_pairs)}')

    def make_weights_after_smote(original_w: np.ndarray, y_original: np.ndarray, y_resampled: np.ndarray) -> np.ndarray:
        y_original = np.asarray(y_original)
        y_resampled = np.asarray(y_resampled)
        w0 = np.asarray(original_w, dtype=np.float32)
        n0 = len(y_original)
        n1 = len(y_resampled)
        w = np.empty(n1, dtype=np.float32)
        w[:n0] = w0[:n0]
        if n1 > n0:
            synth_y = y_resampled[n0:]
            if synth_y.size > 0:
                cls = int(pd.Series(synth_y).mode().iloc[0])
                m = float(w0[y_original == cls].mean()) if np.any(y_original == cls) else float(w0.mean())
                w[n0:] = m
            else:
                w[n0:] = float(w0.mean())
        return w

    def pick_threshold_by_pr_f1(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, Dict[str, float], pd.DataFrame]:
        prec, rec, thr = precision_recall_curve(y_true, proba)
        if thr.size == 0:
            return (0.5, {'best_f1': float('nan'), 'best_precision': float(prec[-1]), 'best_recall': float(rec[-1])}, pd.DataFrame({'precision': prec, 'recall': rec, 'threshold': np.r_[np.nan]}))
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        bi = int(np.nanargmax(f1))
        thr_use = float(thr[max(bi - 1, 0)])
        pr_df = pd.DataFrame({'precision': prec, 'recall': rec, 'threshold': np.r_[np.nan, thr]})
        stats = {'best_f1': float(f1[bi]), 'best_precision': float(prec[bi]), 'best_recall': float(rec[bi])}
        return (thr_use, stats, pr_df)

    def _extract_round_num(s: str) -> Optional[int]:
        import re
        m = re.search('r(\\d+)(?:\\D|$)', str(s), flags=re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _model_root() -> Path:
        return MODEL_PATH.parent.parent

    def _find_prev_dir(root: Path, cur_round_tag: str) -> Optional[Path]:
        raise ContractError('automatic previous-round discovery is disabled; configure previous_best_parameters explicitly')

    def _load_prev_best_params() -> Dict[str, object]:
        if _previous_best_parameters is None:
            return {}
        try:
            obj = json.loads(_previous_best_parameters.read_text(encoding='utf-8'))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    prev_best_params = _load_prev_best_params()
    cur_r = _extract_round_num(ROUND_TAG) or _extract_round_num(MODEL_NAME) or 0
    do_search = bool(FORCE_HPARAM_SEARCH)
    if not do_search and SEARCH_EVERY_N_ROUNDS and (cur_r > 0):
        do_search = cur_r % int(SEARCH_EVERY_N_ROUNDS) == 0
    services.report(f'[HP] do_search={do_search} | prev_params={len(prev_best_params)} keys')
    param_distributions = {'clf__max_depth': randint(3, 10), 'clf__min_child_weight': loguniform(0.5, 10.0), 'clf__subsample': uniform(0.7, 0.3), 'clf__colsample_bytree': uniform(0.7, 0.3), 'clf__reg_lambda': loguniform(0.1, 10.0), 'clf__reg_alpha': loguniform(0.001, 1.0), 'clf__gamma': loguniform(0.001, 1.0), 'clf__learning_rate': loguniform(0.02, 0.2)}
    if 'max_bin' in xgb_kwargs:
        param_distributions['clf__max_bin'] = [xgb_kwargs['max_bin']]
    smote_possible = build_smote(np.asarray(y_train), pos_label=POS_LABEL, sampling_strategy=0.5) is not None
    if smote_possible:
        param_distributions.update({'smote__sampling_strategy': [0.4, 0.5, 0.6], 'smote__k_neighbors': randint(2, 6)})

    def safe_smote_fit_resample(X_imp, y_tr, ss, kn):
        """
        Returns (X_fit, y_fit, sm_or_None). Skips SMOTE if not needed or invalid.
        """
        if not smote_possible:
            return (X_imp, y_tr, None)
        y_tr = np.asarray(y_tr)
        n_pos = int((y_tr == POS_LABEL).sum())
        n_neg = int((y_tr != POS_LABEL).sum())
        n_min = min(n_pos, n_neg)
        n_maj = max(n_pos, n_neg)
        if n_maj <= 0 or n_min <= 1:
            return (X_imp, y_tr, None)
        cur_ratio = n_min / float(n_maj)
        ss = float(ss) if ss is not None else 0.5
        if cur_ratio >= ss - 1e-09:
            return (X_imp, y_tr, None)
        if kn is not None:
            kn = int(kn)
            kn = max(1, min(kn, n_min - 1))
        sm = build_smote(y_tr, pos_label=POS_LABEL, sampling_strategy=ss)
        if sm is None:
            return (X_imp, y_tr, None)
        if kn is not None and hasattr(sm, 'k_neighbors'):
            sm.k_neighbors = kn
        try:
            X_fit, y_fit = sm.fit_resample(X_imp, y_tr)
            return (X_fit, y_fit, sm)
        except ValueError:
            return (X_imp, y_tr, None)

    def weighted_search(n_iter: int, random_state: int) -> Tuple[Dict[str, object], float]:
        sampler = list(ParameterSampler(param_distributions, n_iter=int(n_iter), random_state=int(random_state)))
        best_score = -1.0
        best_params: Dict[str, object] = {}
        for params in tqdm(sampler, desc='Weighted HP search', total=len(sampler)):
            fold_scores = []
            for tr_idx, va_idx in cv_pairs:
                X_tr = X_train.iloc[tr_idx]
                y_tr = y_train[tr_idx]
                X_va = X_train.iloc[va_idx]
                y_va = y_train[va_idx]
                w_tr = review_weights_train[tr_idx].astype(np.float32)
                imp = SimpleImputer(strategy='median')
                X_tr_imp = imp.fit_transform(X_tr)
                X_va_imp = imp.transform(X_va)
                X_fit, y_fit, w_fit = (X_tr_imp, y_tr, w_tr)
                if smote_possible:
                    ss = float(params.get('smote__sampling_strategy', 0.5))
                    kn = params.get('smote__k_neighbors', None)
                    X_fit, y_fit, sm_used = safe_smote_fit_resample(X_tr_imp, y_tr, ss, kn)
                    w_fit = make_weights_after_smote(w_tr, y_tr, y_fit) if sm_used is not None else w_tr
                clf = XGBClassifier(**xgb_kwargs)
                clf_only = {k.replace('clf__', ''): v for k, v in params.items() if k.startswith('clf__')}
                if clf_only:
                    clf.set_params(**clf_only)
                clf.set_params(n_estimators=int(N_EST_SEARCH), early_stopping_rounds=int(EARLY_STOP_ROUNDS), eval_metric='aucpr')
                clf.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=[(X_va_imp, y_va)], verbose=False)
                proba = clf.predict_proba(X_va_imp)[:, 1]
                sc = float(average_precision_score(y_va, proba))
                fold_scores.append(sc)
            mean_sc = float(np.mean(fold_scores)) if fold_scores else float('nan')
            if np.isfinite(mean_sc) and mean_sc > best_score:
                best_score = mean_sc
                best_params = dict(params)
        return (best_params, float(best_score))

    def unweighted_search(n_iter: int, random_state: int) -> Tuple[Dict[str, object], float]:
        sampler = list(ParameterSampler(param_distributions, n_iter=int(n_iter), random_state=int(random_state)))
        best_score = -1.0
        best_params: Dict[str, object] = {}
        total = len(sampler)
        for i, params in enumerate(sampler, start=1):
            fold_scores = []
            for tr_idx, va_idx in cv_pairs:
                X_tr = X_train.iloc[tr_idx]
                y_tr = y_train[tr_idx]
                X_va = X_train.iloc[va_idx]
                y_va = y_train[va_idx]
                imp = SimpleImputer(strategy='median')
                X_tr_imp = imp.fit_transform(X_tr)
                X_va_imp = imp.transform(X_va)
                if smote_possible:
                    ss = float(params.get('smote__sampling_strategy', 0.5))
                    kn = params.get('smote__k_neighbors', None)
                    X_fit, y_fit, _ = safe_smote_fit_resample(X_tr_imp, y_tr, ss, kn)
                else:
                    X_fit, y_fit = (X_tr_imp, y_tr)
                clf = XGBClassifier(**xgb_kwargs)
                clf_only = {k.replace('clf__', ''): v for k, v in params.items() if k.startswith('clf__')}
                if clf_only:
                    clf.set_params(**clf_only)
                clf.set_params(n_estimators=int(N_EST_SEARCH), early_stopping_rounds=int(EARLY_STOP_ROUNDS), eval_metric='aucpr')
                clf.fit(X_fit, y_fit, eval_set=[(X_va_imp, y_va)], verbose=False)
                proba = clf.predict_proba(X_va_imp)[:, 1]
                fold_scores.append(float(average_precision_score(y_va, proba)))
            mean_sc = float(np.mean(fold_scores)) if fold_scores else float('nan')
            if np.isfinite(mean_sc) and mean_sc > best_score:
                best_score = mean_sc
                best_params = dict(params)
            if i == 1 or i % 5 == 0 or i == total:
                services.report(f'[SEARCH] {i}/{total} done | best_pr={best_score:.5f}', flush=True)
        return (best_params, float(best_score))
    best_params: Dict[str, object] = {}
    cv_best_pr = None
    param_source = 'base'
    if len(np.unique(y_train)) < 2:
        services.report('[WARN] Train has only one class; skipping XGB training.')
        best_params = {}
    elif not do_search and prev_best_params:
        best_params = dict(prev_best_params)
        param_source = 'prev_best'
    elif do_search and USE_WEIGHTED_SEARCH:
        with Timer('weighted_search'):
            best_params, cv_best_pr = weighted_search(HPARAM_N_ITER, RANDOM_STATE)
        param_source = 'weighted_search'
    elif do_search and (not USE_WEIGHTED_SEARCH):
        with Timer('unweighted_search'):
            best_params, cv_best_pr = unweighted_search(HPARAM_N_ITER, RANDOM_STATE)
        param_source = 'unweighted_search'
    else:
        best_params = {}
        param_source = 'base'
    services.report('[HP] source:', param_source)
    if best_params:
        services.report('[HP] best_params:', best_params)
    if cv_best_pr is not None:
        services.report('[HP] cv_best_pr_auc:', cv_best_pr)
    prepare_source_output_directory(MODEL_PATH.parent)
    BEST_PARAMS_USED_PATH = MODEL_PATH.parent / 'best_params_used.json'

    def _jsonable(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    def _clean_params_for_json(params: Dict[str, object]) -> Dict[str, object]:
        return {k: _jsonable(v) for k, v in (params or {}).items()}
    BEST_PARAMS_USED_PATH.write_text(json.dumps(_clean_params_for_json(best_params), indent=2), encoding='utf-8')

    def refit_es_pipeline(X_tr_df: pd.DataFrame, y_tr_np: np.ndarray, g_tr_np: np.ndarray, w_tr_np: np.ndarray, params: Dict[str, object]) -> ImbPipeline:
        if len(np.unique(g_tr_np)) >= 2 and len(y_tr_np) >= 50:
            gss = GroupShuffleSplit(n_splits=1, test_size=float(EARLY_STOP_VAL_FRAC), random_state=RANDOM_STATE + 123)
            tr_idx, va_idx = next(gss.split(np.zeros(len(y_tr_np)), y_tr_np, g_tr_np))
        else:
            idx = np.arange(len(y_tr_np))
            rs = np.random.RandomState(RANDOM_STATE + 123)
            rs.shuffle(idx)
            cut = max(1, int(len(idx) * float(EARLY_STOP_VAL_FRAC)))
            va_idx, tr_idx = (idx[:cut], idx[cut:])
        if len(np.unique(y_tr_np[va_idx])) < 2:
            idx = np.arange(len(y_tr_np))
            rs = np.random.RandomState(RANDOM_STATE + 777)
            rs.shuffle(idx)
            cut = max(1, int(len(idx) * float(EARLY_STOP_VAL_FRAC)))
            va_idx, tr_idx = (idx[:cut], idx[cut:])
        X_tr = X_tr_df.iloc[tr_idx]
        y_tr = y_tr_np[tr_idx]
        w_tr = w_tr_np[tr_idx].astype(np.float32)
        X_va = X_tr_df.iloc[va_idx]
        y_va = y_tr_np[va_idx]
        imp = SimpleImputer(strategy='median')
        X_tr_imp = imp.fit_transform(X_tr)
        X_va_imp = imp.transform(X_va)
        X_fit, y_fit, w_fit = (X_tr_imp, y_tr, w_tr)
        sm_used = None
        if smote_possible:
            ss = float(params.get('smote__sampling_strategy', 0.5))
            kn = params.get('smote__k_neighbors', None)
            X_fit, y_fit, sm_used = safe_smote_fit_resample(X_tr_imp, y_tr, ss, kn)
            if sm_used is not None:
                w_fit = make_weights_after_smote(w_tr, y_tr, y_fit)
        clf = XGBClassifier(**xgb_kwargs)
        clf_only = {k.replace('clf__', ''): v for k, v in params.items() if k.startswith('clf__')}
        if clf_only:
            clf.set_params(**clf_only)
        clf.set_params(n_estimators=int(N_ESTIMATORS_BIG), early_stopping_rounds=int(EARLY_STOP_ROUNDS), eval_metric='aucpr')
        with Timer('fit ES'):
            fit_with_maybe_callbacks(clf, X_fit, y_fit, sample_weight=w_fit, eval_set=[(X_va_imp, y_va)], verbose=False, callbacks=[xgb.callback.EvaluationMonitor(period=25)])
        return ImbPipeline([('imp', imp), ('smote', sm_used if sm_used is not None else 'passthrough'), ('clf', clf)])

    def full_train_refit(pipe_es: ImbPipeline, X_tr_df: pd.DataFrame, y_tr_np: np.ndarray, w_tr_np: np.ndarray, params: Dict[str, object]) -> Tuple[ImbPipeline, int]:
        imp = pipe_es.named_steps['imp']
        clf_es: XGBClassifier = pipe_es.named_steps['clf']
        best_iter = getattr(clf_es, 'best_iteration', None)
        used_trees = int(best_iter) + 1 if best_iter is not None else int(getattr(clf_es, 'n_estimators', N_ESTIMATORS_BIG))
        X_imp = imp.fit_transform(X_tr_df)
        X_fit, y_fit, w_fit = (X_imp, y_tr_np, w_tr_np.astype(np.float32))
        sm_used = None
        if smote_possible:
            ss = float(params.get('smote__sampling_strategy', 0.5))
            kn = params.get('smote__k_neighbors', None)
            X_fit, y_fit, sm_used = safe_smote_fit_resample(X_imp, y_tr_np, ss, kn)
            if sm_used is not None:
                w_fit = make_weights_after_smote(w_tr_np.astype(np.float32), y_tr_np, y_fit)
        clf = XGBClassifier(**xgb_kwargs)
        clf_only = {k.replace('clf__', ''): v for k, v in params.items() if k.startswith('clf__')}
        if clf_only:
            clf.set_params(**clf_only)
        clf.set_params(n_estimators=int(used_trees), early_stopping_rounds=None, eval_metric='aucpr')
        with Timer('full-train fit'):
            fit_with_maybe_callbacks(clf, X_fit, y_fit, sample_weight=w_fit, verbose=False, callbacks=[xgb.callback.EvaluationMonitor(period=25)])
        pipe_final = ImbPipeline([('imp', imp), ('smote', sm_used if sm_used is not None else 'passthrough'), ('clf', clf)])
        return (pipe_final, int(used_trees))
    if len(np.unique(y_train)) < 2:
        raise RuntimeError('TRAIN is single-class after filtering; cannot train XGB. (Check skip-dropping / split / labels)')
    with Timer('refit_es_pipeline'):
        pipe_es = refit_es_pipeline(X_train, y_train, groups_train, review_weights_train, best_params)
    with Timer('full_train_refit'):
        best_model, used_trees = full_train_refit(pipe_es, X_train, y_train, review_weights_train, best_params)
    services.report(f'[MODEL] used_trees={used_trees}')

    def predict_proba_batched(model, Xdf: pd.DataFrame, batch: int=50000, desc: str='Predict') -> np.ndarray:
        n = len(Xdf)
        out = np.empty(n, dtype=np.float32)
        for i in tqdm(range(0, n, batch), desc=desc, leave=False):
            j = min(i + batch, n)
            out[i:j] = model.predict_proba(Xdf.iloc[i:j])[:, 1]
        return out
    use_test = len(y_test) > 0 and len(np.unique(y_test)) >= 2
    eval_split = 'test' if use_test else 'train'
    services.report('[EVAL] split:', eval_split)
    if use_test:
        proba_eval = predict_proba_batched(best_model, X_test, desc='Predict TEST')
        y_eval = y_test
        g_eval = groups_test
    else:
        services.report('[WARN] TEST unusable (empty or single-class). Using TRAIN for threshold/metrics (optimistic).')
        proba_eval = predict_proba_batched(best_model, X_train, desc='Predict TRAIN')
        y_eval = y_train
        g_eval = groups_train
    tile_pr_auc = float(average_precision_score(y_eval, proba_eval))
    tile_roc_auc = float(roc_auc_score(y_eval, proba_eval)) if len(np.unique(y_eval)) >= 2 else float('nan')
    thr_use, thr_stats, pr_df = pick_threshold_by_pr_f1(y_eval, proba_eval)
    y_pred = (proba_eval >= thr_use).astype(int)
    cm = confusion_matrix(y_eval, y_pred)
    services.report(f'[TILE] PR-AUC={tile_pr_auc:.4f} | ROC-AUC={tile_roc_auc:.4f} | thr={thr_use:.4f} | F1*={thr_stats.get('best_f1', float('nan')):.4f}')
    fpr, tpr, roc_thr = roc_curve(y_eval, proba_eval)
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': roc_thr})
    tile_preds_df = None
    img_preds_df = None
    img_pr_auc = float('nan')
    img_roc_auc = float('nan')

    def _safe_col(s: str) -> str:
        return s if s in df.columns else ''
    if eval_split == 'test':
        test_idx = state.test_idx if state.test_idx is not None else None
        if test_idx is None or len(test_idx) != len(y_test):
            base = pd.DataFrame({NAME_COL: np.arange(len(y_test)).astype(str)})
            base['group'] = groups_test
            base['label'] = y_test
            base['proba'] = proba_eval
            base['pred'] = (base['proba'] >= thr_use).astype(int)
            tile_preds_df = base
        else:
            base = df.iloc[np.asarray(test_idx)].copy()
            base['group'] = groups_test
            base['label'] = y_test
            base['proba'] = proba_eval
            base['pred'] = (base['proba'] >= thr_use).astype(int)
            keep_cols = [c for c in [NAME_COL, 'group', 'label', 'proba', 'pred', 'reviewed', 'review_tag', 'scale', 'ann_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'] if c in base.columns]
            tile_preds_df = base[keep_cols].copy()
        agg = tile_preds_df.groupby('group', as_index=False).agg(y_img=('label', 'max'), p_img=('proba', 'max'), n_tiles=('proba', 'size'))
        agg['pred_img'] = (agg['p_img'] >= thr_use).astype(int)
        img_preds_df = agg
        if len(np.unique(agg['y_img'].values)) >= 2:
            img_pr_auc = float(average_precision_score(agg['y_img'].values, agg['p_img'].values))
            img_roc_auc = float(roc_auc_score(agg['y_img'].values, agg['p_img'].values))
            services.report(f'[IMG ] PR-AUC={img_pr_auc:.4f} | ROC-AUC={img_roc_auc:.4f}')
        else:
            services.report('[IMG ] single-class at image-level; skipping AUCs.')
    clf_final: XGBClassifier = best_model.named_steps['clf']
    fi = getattr(clf_final, 'feature_importances_', None)
    fi_df = None
    if fi is not None and len(fi) == len(num_cols):
        fi_df = pd.DataFrame({'feature': num_cols, 'importance': fi.astype(float)})
        fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
        services.report('[FI] top features:')
        services.report(fi_df.head(min(TOPK_FEATURES_TO_PRINT, len(fi_df))))
    else:
        services.report('[FI] not available (or length mismatch).')
    for p in [MODEL_PATH.parent, THRESH_JSON_PATH.parent, FI_CSV_PATH.parent, TILE_PREDS_CSV_PATH.parent, IMG_PREDS_CSV_PATH.parent, PR_CURVE_CSV_PATH.parent, ROC_CURVE_CSV_PATH.parent, METRICS_JSON_PATH.parent, CONF_MAT_CSV_PATH.parent, DROPPED_COLS_JSON.parent, CONFIG_JSON_PATH.parent, HISTORY_CSV_PATH.parent, PROGRESS_PNG_PATH.parent]:
        prepare_source_output_directory(p)
    meta = {'csv_used': str(CSV_USED), 'csv_path': str(CSV_PATH), 'round_tag': ROUND_TAG, 'model_name': MODEL_NAME, 'xgboost_version': xgb.__version__, 'params_source': param_source, 'cv_best_pr_auc': None if cv_best_pr is None else float(cv_best_pr), 'used_trees': int(used_trees), 'tile_pr_auc': float(tile_pr_auc), 'tile_roc_auc': float(tile_roc_auc), 'img_pr_auc': None if np.isnan(img_pr_auc) else float(img_pr_auc), 'img_roc_auc': None if np.isnan(img_roc_auc) else float(img_roc_auc), 'threshold': float(thr_use), 'threshold_stats': thr_stats, 'train_n': int(len(y_train)), 'test_n': int(len(y_test)), 'pos_rate_train': float(np.mean(y_train == POS_LABEL)), 'pos_rate_test': float(np.mean(y_test == POS_LABEL)) if len(y_test) else None, 'weights_mean_train': float(np.mean(review_weights_train)), 'best_params': best_params if isinstance(best_params, dict) else {}}
    joblib.dump({'pipeline': best_model, 'features': list(num_cols), 'threshold': float(thr_use), 'meta': meta}, MODEL_PATH)
    THRESH_JSON_PATH.write_text(json.dumps({'threshold': float(thr_use)}, indent=2), encoding='utf-8')
    pr_df.to_csv(PR_CURVE_CSV_PATH, index=False)
    roc_df.to_csv(ROC_CURVE_CSV_PATH, index=False)
    if tile_preds_df is not None:
        tile_preds_df.to_csv(TILE_PREDS_CSV_PATH, index=False)
    if img_preds_df is not None:
        img_preds_df.to_csv(IMG_PREDS_CSV_PATH, index=False)
    pd.DataFrame(cm, index=['true_0', 'true_1'], columns=['pred_0', 'pred_1']).to_csv(CONF_MAT_CSV_PATH)
    if fi_df is not None:
        fi_df.to_csv(FI_CSV_PATH, index=False)
    DROPPED_COLS_JSON.write_text(json.dumps({'dropped_cols': dropped_cols}, indent=2), encoding='utf-8')
    CFG_KEYS = ['TEST_SIZE', 'RANDOM_STATE', 'N_SPLITS_CV', 'POS_LABEL', 'BORDER_PX', 'TILE_SIZE', 'TINY_MIN_WH', 'TINY_MAX_AREA', 'BORDER_WEIGHT', 'UNCERT_MARGIN', 'UNCERT_MULT', 'DROP_TINY_IN_TRAIN', 'DROP_BORDER_FROM_TEST', 'EARLY_STOP_ROUNDS', 'EARLY_STOP_VAL_FRAC', 'N_ESTIMATORS_BIG', 'USE_REVIEW_WEIGHTING', 'WEIGHT_CORRECT', 'WEIGHT_WRONG', 'AUTO_DETECT_PREV_ROUND', 'USE_REVIEW_TAG_WEIGHTS', 'WEIGHT_MAP', 'DROP_SKIP_ROWS', 'AUG_CFG']

    def _configuration_bindings() -> Mapping[str, object]:
        return MappingProxyType(dict((('TEST_SIZE', TEST_SIZE), ('RANDOM_STATE', RANDOM_STATE), ('N_SPLITS_CV', N_SPLITS_CV), ('POS_LABEL', POS_LABEL), ('BORDER_PX', BORDER_PX), ('TILE_SIZE', TILE_SIZE), ('TINY_MIN_WH', TINY_MIN_WH), ('TINY_MAX_AREA', TINY_MAX_AREA), ('BORDER_WEIGHT', BORDER_WEIGHT), ('UNCERT_MARGIN', UNCERT_MARGIN), ('UNCERT_MULT', UNCERT_MULT), ('DROP_TINY_IN_TRAIN', DROP_TINY_IN_TRAIN), ('DROP_BORDER_FROM_TEST', DROP_BORDER_FROM_TEST), ('EARLY_STOP_ROUNDS', EARLY_STOP_ROUNDS), ('EARLY_STOP_VAL_FRAC', EARLY_STOP_VAL_FRAC), ('N_ESTIMATORS_BIG', N_ESTIMATORS_BIG), ('USE_REVIEW_WEIGHTING', USE_REVIEW_WEIGHTING), ('WEIGHT_CORRECT', WEIGHT_CORRECT), ('WEIGHT_WRONG', WEIGHT_WRONG), ('AUTO_DETECT_PREV_ROUND', AUTO_DETECT_PREV_ROUND), ('USE_REVIEW_TAG_WEIGHTS', USE_REVIEW_TAG_WEIGHTS), ('WEIGHT_MAP', WEIGHT_MAP), ('DROP_SKIP_ROWS', DROP_SKIP_ROWS), ('AUG_CFG', AUG_CFG))))

    cfg_out = {key: _configuration_bindings().get(key, None) for key in CFG_KEYS}
    cfg_out.update({'xgb_kwargs': xgb_kwargs, 'best_params': best_params, 'param_source': param_source, 'csv_used': str(CSV_USED), 'n_features': int(len(num_cols)), 'split_mode': 'SealedManifest(GroupPure)', 'split_manifest': str(state.split_manifest.path), 'split_manifest_sha256': state.split_manifest.sha256, 'previous_tile_predictions': str(state.previous_tile_predictions.path), 'previous_tile_predictions_sha256': state.previous_tile_predictions.sha256, 'previous_threshold': str(state.previous_threshold.path), 'previous_threshold_sha256': state.previous_threshold.sha256})
    CONFIG_JSON_PATH.write_text(json.dumps(cfg_out, indent=2, default=str), encoding='utf-8')
    metrics_out = {'tile': {'split': eval_split, 'pr_auc': float(tile_pr_auc), 'roc_auc': float(tile_roc_auc), 'threshold': float(thr_use), 'confusion_matrix': cm.tolist(), 'threshold_stats': thr_stats}, 'image': {'available': bool(img_preds_df is not None), 'pr_auc': None if np.isnan(img_pr_auc) else float(img_pr_auc), 'roc_auc': None if np.isnan(img_roc_auc) else float(img_roc_auc)}, 'meta': meta}
    METRICS_JSON_PATH.write_text(json.dumps(metrics_out, indent=2, default=str), encoding='utf-8')
    row = {'ts': pd.Timestamp.utcnow().isoformat(), 'round_tag': ROUND_TAG, 'model_name': MODEL_NAME, 'param_source': param_source, 'tile_pr_auc': float(tile_pr_auc), 'tile_roc_auc': float(tile_roc_auc) if np.isfinite(tile_roc_auc) else np.nan, 'img_pr_auc': np.nan if np.isnan(img_pr_auc) else float(img_pr_auc), 'img_roc_auc': np.nan if np.isnan(img_roc_auc) else float(img_roc_auc), 'threshold': float(thr_use), 'n_features': int(len(num_cols)), 'train_n': int(len(y_train)), 'test_n': int(len(y_test))}
    hist = None
    if HISTORY_CSV_PATH.exists():
        try:
            hist = pd.read_csv(HISTORY_CSV_PATH)
        except Exception:
            hist = None
    if hist is None:
        hist = pd.DataFrame(columns=list(row.keys()))
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(HISTORY_CSV_PATH, index=False)
    try:

        def _rnum(rt: str) -> float:
            import re
            m = re.search('r(\\d+)', str(rt), flags=re.IGNORECASE)
            return float(m.group(1)) if m else float('nan')
        hist['_rnum'] = hist['round_tag'].map(_rnum)
        hist2 = hist.sort_values(['_rnum', 'ts'], ascending=[True, True])
        plt.figure()
        plt.plot(hist2['_rnum'].to_numpy(), hist2['tile_pr_auc'].to_numpy(), marker='o')
        plt.xlabel('round')
        plt.ylabel('tile PR-AUC')
        plt.title('Training progress')
        plt.grid(True)
        plt.savefig(PROGRESS_PNG_PATH, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        services.report('[WARN] progress plot failed:', e)
    services.report('\n[SAVED]')
    services.report('  model:', MODEL_PATH)
    services.report('  thr  :', THRESH_JSON_PATH)
    services.report('  preds:', TILE_PREDS_CSV_PATH, '|', IMG_PREDS_CSV_PATH)
    services.report('  curves:', PR_CURVE_CSV_PATH, '|', ROC_CURVE_CSV_PATH)
    services.report('  metrics:', METRICS_JSON_PATH)
    services.report('  FI:', FI_CSV_PATH)
    services.report('  history:', HISTORY_CSV_PATH, '| progress:', PROGRESS_PNG_PATH)

    return SourceResult_0001_0006(state.artifacts, float(thr_use), str(eval_split), len(num_cols), int(used_trees))

@dataclass(frozen=True)
class SourceAlgorithmBatch:
    setup: SourceInputs_0001_0005
    previous_best_parameters: SourceInputArtifact | None = None

@dataclass(frozen=True)
class SourceAlgorithmBatchResult:
    cell_ids: tuple[str, str]
    setup: SourceResult_0001_0005
    fit: SourceResult_0001_0006

def execute_xgb_nb0001_source_algorithms(batch: SourceAlgorithmBatch, services: SourceAlgorithmServices, policy: ExecutionPolicy) -> SourceAlgorithmBatchResult:
    require_authorized(policy)
    setup = source_algorithm_0001_0005(batch.setup, services, policy)
    fit = source_algorithm_0001_0006(SourceInputs_0001_0006(setup.continuation, batch.previous_best_parameters), services, policy)
    return SourceAlgorithmBatchResult(SOURCE_ALGORITHM_ORDER, setup, fit)
