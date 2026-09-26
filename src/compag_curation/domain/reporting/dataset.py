"""Dataset accounting and Table-1 rendering from sealed local artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from json import loads
from pathlib import Path
from re import IGNORECASE, compile as compile_pattern
from typing import Any, Iterable, Mapping


class DatasetSummaryContractError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSummaryRequest:
    tiled_coco: Path
    original_coco: Path
    tiles_directory: Path
    split_directory: Path
    tile_size: int = 512


@dataclass(frozen=True)
class DatasetSummary:
    cards: int
    tiles: int
    instances: int
    cj_instances: int | None
    non_cj_instances: int | None
    non_cj_to_cj_ratio: float | None
    stride_y: int | None
    stride_x: int | None
    overlap_y: int | None
    overlap_x: int | None
    train_cards: int | None
    validation_cards: int | None
    test_cards: int | None
    latex: str


DatasetParameter = str | int


@dataclass(frozen=True)
class DatasetSummaryOperation:
    operation: str
    input_roles: tuple[str, ...]
    output_role: str
    parameters: tuple[tuple[str, DatasetParameter], ...] = ()


def build_dataset_summary_program(
    request: DatasetSummaryRequest,
) -> tuple[DatasetSummaryOperation, ...]:
    return (
        DatasetSummaryOperation(
            "read-sealed-coco-documents",
            ("original_coco", "tiled_coco"),
            "coco_documents",
        ),
        DatasetSummaryOperation(
            "count-cards-tiles-and-categories",
            ("coco_documents",),
            "dataset_counts",
        ),
        DatasetSummaryOperation(
            "derive-tile-stride-and-overlap",
            ("tiles_directory",),
            "tile_geometry",
            (("tile_size", request.tile_size),),
        ),
        DatasetSummaryOperation(
            "count-sealed-splits",
            ("split_directory",),
            "split_counts",
        ),
        DatasetSummaryOperation(
            "render-dataset-latex",
            ("dataset_counts", "tile_geometry", "split_counts"),
            "latex",
        ),
    )


def _load_json_object(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise DatasetSummaryContractError(f"COCO JSON is missing: {path}")
    value = loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DatasetSummaryContractError(f"COCO root must be an object: {path}")
    return value


def _read_split_count(directory: Path, stems: Iterable[str]) -> int | None:
    """Read split identities from text/CSV/JSON/NPY sidecars without NumPy import."""

    for stem in stems:
        for suffix in (".txt", ".csv", ".json"):
            path = directory / f"{stem}{suffix}"
            if not path.is_file():
                continue
            if suffix == ".json":
                value = loads(path.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    return len(value)
                if isinstance(value, dict):
                    for key in ("ids", "cards", "images", "values"):
                        if isinstance(value.get(key), list):
                            return len(value[key])
            else:
                rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
                rows = [line for line in rows if line and not line.startswith("#")]
                if suffix == ".csv" and rows and rows[0].lower() in {"id", "card_id", "image"}:
                    rows = rows[1:]
                return len(rows)
    return None


def _latex_value(value: Any) -> str:
    return "NA" if value is None else str(value)


# SOURCE_CELL: NB-LIVE-0005-C0003
# SOURCE_STATEMENT_MAP: coco-count-category-ratio-stride-overlap-splits-latex -> build_dataset_summary
def build_dataset_summary(request: DatasetSummaryRequest) -> DatasetSummary:
    """Reproduce the source dataset summary using explicit artifact paths."""

    program = iter(build_dataset_summary_program(request))

    def require_operation(expected: str) -> DatasetSummaryOperation:
        try:
            operation = next(program)
        except StopIteration as exc:
            raise DatasetSummaryContractError(
                f"dataset summary program ended before: {expected}"
            ) from exc
        if operation.operation != expected:
            raise DatasetSummaryContractError(
                f"dataset summary program order changed: {operation.operation} != {expected}"
            )
        return operation

    require_operation("read-sealed-coco-documents")
    original = _load_json_object(request.original_coco)
    tiled = _load_json_object(request.tiled_coco)
    if not request.tiles_directory.is_dir():
        raise DatasetSummaryContractError(
            f"tiles directory is missing: {request.tiles_directory}"
        )
    if not request.split_directory.is_dir():
        raise DatasetSummaryContractError(
            f"split directory is missing: {request.split_directory}"
        )
    require_operation("count-cards-tiles-and-categories")
    cards = len(original.get("images", []))
    tiles = len(tiled.get("images", []))
    annotations = tiled.get("annotations", [])
    instances = len(annotations)
    category_names = {
        int(category["id"]): str(category.get("name", ""))
        for category in tiled.get("categories", [])
        if isinstance(category, dict) and "id" in category
    }
    cj_ids = {
        identity
        for identity, name in category_names.items()
        if name.strip().lower() == "cj"
    }
    non_cj_ids = {
        identity
        for identity, name in category_names.items()
        if name.strip().lower() in {"non-cj", "non_cj", "noncj", "non cj"}
    }
    cj_count = (
        sum(int(annotation.get("category_id", -1)) in cj_ids for annotation in annotations)
        if cj_ids
        else None
    )
    non_cj_count = (
        sum(
            int(annotation.get("category_id", -1)) in non_cj_ids
            for annotation in annotations
        )
        if non_cj_ids
        else None
    )
    ratio = (
        non_cj_count / float(cj_count)
        if cj_count is not None and non_cj_count is not None and cj_count > 0
        else None
    )

    geometry_operation = require_operation("derive-tile-stride-and-overlap")
    geometry_parameters = dict(geometry_operation.parameters)
    tile_size = int(geometry_parameters["tile_size"])
    tile_pattern = compile_pattern(r"_y(?P<y>\d+)x(?P<x>\d+)\.", IGNORECASE)
    ys: set[int] = set()
    xs: set[int] = set()
    for path in request.tiles_directory.iterdir():
        match = tile_pattern.search(path.name)
        if match:
            ys.add(int(match.group("y")))
            xs.add(int(match.group("x")))
    sorted_y = sorted(ys)
    sorted_x = sorted(xs)
    stride_y = min(
        (b - a for a, b in zip(sorted_y, sorted_y[1:]) if b > a), default=None
    )
    stride_x = min(
        (b - a for a, b in zip(sorted_x, sorted_x[1:]) if b > a), default=None
    )
    overlap_y = tile_size - stride_y if stride_y is not None else None
    overlap_x = tile_size - stride_x if stride_x is not None else None
    require_operation("count-sealed-splits")
    train = _read_split_count(request.split_directory, ("train_ids", "train"))
    validation = _read_split_count(
        request.split_directory, ("val_ids", "validation_ids", "val")
    )
    test = _read_split_count(request.split_directory, ("test_ids", "test"))
    require_operation("render-dataset-latex")
    latex = (
        "\\begin{tabular}{l r}\n"
        "\\toprule\n"
        f"Full-card images & {_latex_value(cards)} \\\\\n"
        f"Tiles (512$\\times$512) & {_latex_value(tiles)} \\\\\n"
        f"Instances (total) & {_latex_value(instances)} \\\\\n"
        f"CJ / non-CJ & {_latex_value(cj_count)} / {_latex_value(non_cj_count)} \\\\\n"
        f"Imbalance (non-CJ:CJ) & {'NA' if ratio is None else f'{ratio:.2f}'}:1 \\\\\n"
        f"Tile stride (px) & {_latex_value(stride_y)}$\\times${_latex_value(stride_x)} \\\\\n"
        f"Tile overlap (px) & {_latex_value(overlap_y)}$\\times${_latex_value(overlap_x)} \\\\\n"
        f"Cards in train / val / test & {_latex_value(train)} / {_latex_value(validation)} / {_latex_value(test)} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )
    try:
        extra_operation = next(program)
    except StopIteration:
        extra_operation = None
    if extra_operation is not None:
        raise DatasetSummaryContractError(
            f"dataset summary program has an extra operation: {extra_operation.operation}"
        )
    return DatasetSummary(
        cards,
        tiles,
        instances,
        cj_count,
        non_cj_count,
        ratio,
        stride_y,
        stride_x,
        overlap_y,
        overlap_x,
        train,
        validation,
        test,
        latex,
    )
