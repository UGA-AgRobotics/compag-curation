"""Bounded adapter for the source-backed canonical point evaluator."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import stat
import tempfile
import uuid
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Callable, Sequence

from compag_curation.public_config import CANONICAL_PROFILE, V2_PIPELINE_PROFILES
from compag_curation.public_io import (
    PublicIOError,
    canonical_json_bytes,
    compact_json_sha256,
    fsync_directory,
    hash_file_snapshot,
    manifest_rows,
    portable_basename,
    publish_directory_noreplace,
    require,
    stable_file,
    verified_file_path,
    write_new_bytes,
    write_new_json,
)


CANONICAL_EVALUATION_VIEWS = ("all", "kept", "xgb")
CANONICAL_XGB_THRESHOLD = 0.5
CANONICAL_NMS_IOU = 0.5
_GEOMETRY_POLICY = "polygon-preferred-bbox-fallback"
_EVALUATOR = "compag_curation.evaluation.points_coverage_base.run_eval"
_MAX_INPUT_FILES = 16
_MAX_IMAGES = 256
_MAX_XML_BYTES = 64 * 1024 * 1024
_MAX_CSV_BYTES = 256 * 1024 * 1024
_MAX_INPUT_BYTES = 512 * 1024 * 1024
_MAX_OUTPUT_BYTES = 512 * 1024 * 1024
_MAX_OUTPUT_FILES = 32
_MAX_OVERALL_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_STREAM_CHARS = 1024 * 1024
_MAX_COUNT = 10_000_000

_REQUIRED_TEMPLATES = (
    "README__{view}.md",
    "fig_recall_by_image__{view}.png",
    "overall__{view}.json",
    "points_detail__{view}.csv",
    "summary__{view}.csv",
)
_OPTIONAL_TEMPLATES = (
    "fig_min_dist_hit_vs_miss__{view}.png",
    "fig_miss_recoverability__{view}.png",
    "fig_point_cover_hist__{view}.png",
    "fig_points_per_mask_hist__{view}.png",
)
_PROBABILITY_FIELDS = (
    "micro_recall_cov",
    "micro_proposal_cov_recall",
    "micro_al_cov_recall",
    "micro_miss_within_margin_rate",
    "micro_miss_in_al_topk_rate",
    "micro_mask_hit_rate",
    "micro_fp_rate_masks",
    "micro_merge_rate_masks_ge2pts",
    "micro_multi_cover_rate",
)


def _load_run_eval() -> Callable[[argparse.Namespace], None]:
    # Importing the source evaluator loads the numerical stack, so execution is
    # deliberately deferred until after all inert request validation succeeds.
    from compag_curation.evaluation.points_coverage_base import run_eval

    return run_eval


def _input_file(path: Path, role: str, suffix: str, limit: int) -> dict[str, object]:
    require(isinstance(path, Path) and path.is_absolute(), f"{role} path must be absolute")
    require(path.suffix.lower() == suffix, f"{role} must use the {suffix} suffix")
    try:
        info = path.lstat()
    except OSError as exc:
        raise PublicIOError(f"{role} is unavailable: {path.name}") from exc
    require(
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_nlink == 1,
        f"{role} must be a single-link regular file",
    )
    require(path == path.resolve(strict=True), f"{role} path contains a symlink component")
    require(0 < info.st_size <= limit, f"{role} size is outside its supported bound")
    digest, snapshot = hash_file_snapshot(path, require_single_link=True)
    require(snapshot.st_size <= limit, f"{role} grew beyond its supported bound")
    return {
        "path": path,
        "role": role,
        "sha256": digest,
        "size_bytes": snapshot.st_size,
        "limit": limit,
    }


def _validated_inputs(
    point_annotations: Path,
    prediction_csvs: Sequence[Path],
    tile_index_csvs: Sequence[Path],
    al_candidate_csvs: Sequence[Path],
) -> tuple[dict[str, object], tuple[dict[str, object], ...], tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    predictions = tuple(prediction_csvs)
    indexes = tuple(tile_index_csvs)
    candidates = tuple(al_candidate_csvs)
    require(1 <= len(predictions) <= _MAX_INPUT_FILES, "prediction CSV count is outside its supported bound")
    require(len(indexes) <= _MAX_INPUT_FILES, "tile-index CSV count is outside its supported bound")
    require(len(candidates) <= _MAX_INPUT_FILES, "AL-candidate CSV count is outside its supported bound")
    suffix = point_annotations.suffix.lower()
    require(suffix in {".xml", ".json"}, "point annotations must use the .xml or .json suffix")
    annotation = _input_file(
        point_annotations,
        "CVAT points XML" if suffix == ".xml" else "canonical point JSON",
        suffix,
        _MAX_XML_BYTES,
    )
    pred = tuple(
        _input_file(path, f"prediction CSV {index + 1}", ".csv", _MAX_CSV_BYTES)
        for index, path in enumerate(predictions)
    )
    tile = tuple(
        _input_file(path, f"tile-index CSV {index + 1}", ".csv", _MAX_CSV_BYTES)
        for index, path in enumerate(indexes)
    )
    al = tuple(
        _input_file(path, f"AL-candidate CSV {index + 1}", ".csv", _MAX_CSV_BYTES)
        for index, path in enumerate(candidates)
    )
    rows = (annotation, *pred, *tile, *al)
    paths = [str(row["path"]) for row in rows]
    require(len(paths) == len(set(paths)), "canonical evaluation inputs must be distinct files")
    require(sum(int(row["size_bytes"]) for row in rows) <= _MAX_INPUT_BYTES, "canonical evaluation inputs exceed the total byte bound")
    return annotation, pred, tile, al


def _normalized_cvat_xml(rows: Sequence[dict[str, object]], label: str) -> bytes:
    """Render validated canonical point rows for the source-backed XML evaluator."""

    grouped: dict[str, tuple[int, int, list[tuple[float, float]]]] = {}
    for row in rows:
        name = str(row["image"])
        width = int(row["width"])
        height = int(row["height"])
        current = grouped.setdefault(name, (width, height, []))
        require(current[:2] == (width, height), "point annotation dimensions disagree within one image")
        current[2].append((float(row["x"]), float(row["y"])))
    require(bool(grouped), "point annotations contain no canonical rows")
    root = element_tree.Element("annotations")
    for image_index, name in enumerate(sorted(grouped)):
        width, height, points = grouped[name]
        image = element_tree.SubElement(
            root,
            "image",
            {"id": str(image_index), "name": name, "width": str(width), "height": str(height)},
        )
        for x, y in sorted(points, key=lambda value: (value[1], value[0])):
            element_tree.SubElement(
                image,
                "points",
                {"label": label, "points": f"{format(x, '.17g')},{format(y, '.17g')}"},
            )
    return element_tree.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _views(value: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(value)
    require(selected and len(selected) == len(set(selected)), "evaluation views must be nonempty and unique")
    require(set(selected) <= set(CANONICAL_EVALUATION_VIEWS), "evaluation view must be all, kept, or xgb")
    return tuple(view for view in CANONICAL_EVALUATION_VIEWS if view in selected)


def _images(value: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(value)
    require(len(selected) <= _MAX_IMAGES, "image selection exceeds its supported bound")
    require(len(selected) == len(set(selected)), "image selection contains duplicates")
    for name in selected:
        require(isinstance(name, str) and "," not in name, "image selection contains an invalid name")
        portable_basename(name, "evaluation image name")
    return selected


def _claim_staging(output: Path) -> tuple[Path, Path]:
    require(isinstance(output, Path) and output.is_absolute(), "evaluation output path must be absolute")
    absolute = output.absolute()
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise PublicIOError("evaluation output parent is unavailable") from exc
    output = parent / absolute.name
    require(output == absolute, "evaluation output contains a symlinked component")
    portable_basename(output.name, "evaluation output name")
    require(not output.exists() and not output.is_symlink(), "evaluation output already exists")
    parent_info = parent.lstat()
    require(stat.S_ISDIR(parent_info.st_mode) and not parent.is_symlink(), "evaluation output parent is unsafe")
    staging = parent / f".{output.name}.canonical-eval.{uuid.uuid4()}"
    require(len(staging.name.encode("ascii")) <= 255, "evaluation output name is too long for staging")
    staging.mkdir(mode=0o700, exist_ok=False)
    fsync_directory(parent)
    return output, staging


def _pin(stack: contextlib.ExitStack, row: dict[str, object]) -> Path:
    descriptor, _snapshot = stack.enter_context(
        verified_file_path(
            Path(row["path"]),
            str(row["sha256"]),
            max_bytes=int(row["limit"]),
            require_single_link=True,
        )
    )
    return descriptor


def _namespace(
    *,
    view: str,
    output: Path,
    cvat_xml: Path,
    predictions: Sequence[Path],
    tile_indexes: Sequence[Path],
    al_candidates: Sequence[Path],
    images: Sequence[str],
    cvat_label: str,
    max_area_fraction: float | None,
    al_margin: float,
    al_topk: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        cvat_xml=cvat_xml,
        cvat_label=cvat_label,
        pred_csv=list(predictions),
        tile_index=list(tile_indexes),
        run_dir=[],
        al_candidates_csv=list(al_candidates),
        images=",".join(images),
        out_dir=output,
        images_root="",
        stage=view,
        thr_xgb=CANONICAL_XGB_THRESHOLD,
        score_col="xgb_p",
        max_area_frac=max_area_fraction,
        nms_iou=CANONICAL_NMS_IOU,
        al_margin=al_margin,
        al_topk=al_topk,
        al_only_uncertain=True,
        det_policy="xgb",
        det_thr=CANONICAL_XGB_THRESHOLD,
        yolo_conf_thr=0.20,
        yolo_iou_thr=0.60,
        use_p_fused=False,
    )


def _json_object(path: Path, role: str) -> dict[str, object]:
    payload, _snapshot = stable_file(path, max_bytes=_MAX_OVERALL_BYTES)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            require(key not in value, f"{role} contains a duplicate JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _constant: None,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PublicIOError(f"{role} is not bounded JSON") from exc
    require(isinstance(value, dict) and len(value) <= 32, f"{role} has an invalid top-level object")
    return value


def _count(value: object, role: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{role} must be an integer")
    require(0 <= value <= _MAX_COUNT, f"{role} is outside its supported bound")
    return value


def _probability(value: object, role: str) -> float | None:
    if value is None:
        return None
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{role} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        return None
    require(0.0 <= result <= 1.0, f"{role} is outside [0,1]")
    return result


def _stage_summary(staging: Path, view: str, *, al_margin: float, al_topk: int) -> dict[str, object]:
    value = _json_object(staging / f"overall__{view}.json", f"{view} overall result")
    require(value.get("stage") == view, f"{view} overall result has the wrong stage")
    notes = value.get("notes")
    require(isinstance(notes, dict), f"{view} overall result is missing notes")
    definition = notes.get("definition")
    require(
        isinstance(definition, str)
        and len(definition) <= 1024
        and "polygon preferred; bbox fallback" in definition,
        f"{view} overall result changed the point-coverage geometry policy",
    )
    params = value.get("params")
    require(isinstance(params, dict), f"{view} overall result is missing parameters")
    require(params.get("det_policy") == "xgb", f"{view} overall result enabled a non-XGB policy")
    require(params.get("thr_xgb") == CANONICAL_XGB_THRESHOLD, f"{view} overall result changed the XGB threshold")
    require(params.get("nms_iou") == CANONICAL_NMS_IOU, f"{view} overall result changed NMS")
    require(params.get("al_margin") == al_margin and params.get("al_topk") == al_topk, f"{view} overall result changed AL parameters")
    total_gt = _count(value.get("total_gt_points"), f"{view}.total_gt_points")
    total_pred = _count(value.get("total_pred_instances"), f"{view}.total_pred_instances")
    total_covered = _count(value.get("total_covered_points"), f"{view}.total_covered_points")
    require(total_covered <= total_gt, f"{view} covered-point count exceeds ground truth")
    result: dict[str, object] = {
        "n_images": _count(value.get("n_images"), f"{view}.n_images"),
        "total_gt_points": total_gt,
        "total_pred_instances": total_pred,
        "total_covered_points": total_covered,
    }
    for field in _PROBABILITY_FIELDS:
        require(field in value, f"{view} overall result is missing {field}")
        result[field] = _probability(value.get(field), f"{view}.{field}")
    redundancy = value.get("micro_redundancy_mean")
    if redundancy is None or (isinstance(redundancy, (int, float)) and not math.isfinite(float(redundancy))):
        result["micro_redundancy_mean"] = None
    else:
        require(isinstance(redundancy, (int, float)) and not isinstance(redundancy, bool), f"{view}.micro_redundancy_mean must be numeric or null")
        rendered = float(redundancy)
        require(0.0 <= rendered <= _MAX_COUNT, f"{view}.micro_redundancy_mean is outside its bound")
        result["micro_redundancy_mean"] = rendered
    return result


def _source_output_contract(views: Sequence[str]) -> tuple[set[str], set[str], dict[str, int]]:
    required = {template.format(view=view) for view in views for template in _REQUIRED_TEMPLATES}
    optional = {template.format(view=view) for view in views for template in _OPTIONAL_TEMPLATES}
    limits: dict[str, int] = {}
    for name in required | optional:
        if name.endswith(".csv"):
            limits[name] = 256 * 1024 * 1024
        elif name.endswith(".png"):
            limits[name] = 64 * 1024 * 1024
        elif name.endswith(".json"):
            limits[name] = _MAX_OVERALL_BYTES
        else:
            limits[name] = 4 * 1024 * 1024
    return required, optional, limits


def _seal_source_outputs(staging: Path, views: Sequence[str]) -> list[dict[str, object]]:
    required, optional, limits = _source_output_contract(views)
    rows = manifest_rows(
        staging,
        allowed_files=required | optional,
        file_size_limits=limits,
        max_total_bytes=_MAX_OUTPUT_BYTES,
    )
    observed = {str(row["path"]) for row in rows}
    require(required <= observed, "canonical evaluator output closure is incomplete")
    require(len(rows) <= _MAX_OUTPUT_FILES, "canonical evaluator emitted too many files")
    for row in rows:
        path = staging / str(row["path"])
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "canonical evaluator emitted an unsafe file")
            os.fchmod(fd, 0o644)
            os.fsync(fd)
        finally:
            os.close(fd)
    fsync_directory(staging)
    sealed = manifest_rows(
        staging,
        allowed_files=required | optional,
        file_size_limits=limits,
        max_total_bytes=_MAX_OUTPUT_BYTES,
    )
    require(all(row["mode_octal"] == "0644" for row in sealed), "canonical evaluator file modes are not sealed")
    return [
        {
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
        }
        for row in sealed
    ]


def evaluate_canonical_points(
    cvat_xml: Path,
    prediction_csvs: Sequence[Path],
    output: Path,
    *,
    tile_index_csvs: Sequence[Path] = (),
    al_candidate_csvs: Sequence[Path] = (),
    images: Sequence[str] = (),
    views: Sequence[str] = CANONICAL_EVALUATION_VIEWS,
    cvat_label: str = "cj",
    max_area_fraction: float | None = None,
    al_margin: float = 0.20,
    al_topk: int = 50,
    profile: str = CANONICAL_PROFILE,
) -> dict[str, object]:
    """Run source-backed point coverage and atomically publish bounded results."""

    selected_views = _views(views)
    selected_images = _images(images)
    require(
        profile in V2_PIPELINE_PROFILES,
        "v2 point-evaluation profile is unsupported",
    )
    require(
        isinstance(cvat_label, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", cvat_label) is not None,
        "CVAT label is not portable",
    )
    require(isinstance(al_topk, int) and not isinstance(al_topk, bool) and 1 <= al_topk <= 500, "AL top-k is outside [1,500]")
    require(isinstance(al_margin, (int, float)) and not isinstance(al_margin, bool) and math.isfinite(float(al_margin)) and 0.0 < float(al_margin) <= 1.0, "AL margin is outside (0,1]")
    if max_area_fraction is not None:
        require(
            isinstance(max_area_fraction, (int, float))
            and not isinstance(max_area_fraction, bool)
            and math.isfinite(float(max_area_fraction))
            and 0.0 < float(max_area_fraction) <= 1.0,
            "maximum area fraction is outside (0,1]",
        )
        max_area_fraction = float(max_area_fraction)
    annotation, predictions, tile_indexes, al_candidates = _validated_inputs(
        cvat_xml,
        prediction_csvs,
        tile_index_csvs,
        al_candidate_csvs,
    )
    from compag_curation.point_annotations import inspect_point_annotations

    annotation_rows, annotation_inspection = inspect_point_annotations(
        Path(annotation["path"]),
        label=cvat_label,
    )
    require(
        annotation_inspection.get("source_sha256") == annotation["sha256"],
        "point annotation inspection differs from the pinned input identity",
    )
    output, staging = _claim_staging(output)
    with contextlib.ExitStack() as stack:
        annotation_fd = _pin(stack, annotation)
        prediction_fds = tuple(_pin(stack, row) for row in predictions)
        tile_index_fds = tuple(_pin(stack, row) for row in tile_indexes)
        al_candidate_fds = tuple(_pin(stack, row) for row in al_candidates)
        annotation_adapter = "DIRECT_PINNED_CVAT_XML"
        evaluator_annotations = annotation_fd
        if Path(annotation["path"]).suffix.lower() == ".json":
            temporary_name = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="compag-canonical-point-adapter-")
            )
            temporary_root = Path(temporary_name)
            temporary_root.chmod(0o700)
            normalized = temporary_root / "canonical-points.xml"
            write_new_bytes(normalized, _normalized_cvat_xml(annotation_rows, cvat_label))
            evaluator_annotations = normalized
            annotation_adapter = "CANONICAL_JSON_TO_EPHEMERAL_CVAT_XML"
        runner = _load_run_eval()
        for view in selected_views:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                runner(
                    _namespace(
                        view=view,
                        output=staging,
                        cvat_xml=evaluator_annotations,
                        predictions=prediction_fds,
                        tile_indexes=tile_index_fds,
                        al_candidates=al_candidate_fds,
                        images=selected_images,
                        cvat_label=cvat_label,
                        max_area_fraction=max_area_fraction,
                        al_margin=float(al_margin),
                        al_topk=al_topk,
                    )
                )
            require(
                len(stdout.getvalue()) <= _MAX_SOURCE_STREAM_CHARS
                and len(stderr.getvalue()) <= _MAX_SOURCE_STREAM_CHARS,
                "canonical evaluator source stream exceeds its supported bound",
            )
            require(not stderr.getvalue().strip(), "canonical evaluator emitted stderr")
    stages = {
        view: _stage_summary(staging, view, al_margin=float(al_margin), al_topk=al_topk)
        for view in selected_views
    }
    source_outputs = _seal_source_outputs(staging, selected_views)
    input_records = [
        {
            "role": str(row["role"]),
            "basename": portable_basename(Path(row["path"]).name, "evaluation input basename"),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
        }
        for row in (annotation, *predictions, *tile_indexes, *al_candidates)
    ]
    summary: dict[str, object] = {
        "schema": "compag-curation-canonical-point-evaluation/v1",
        "status": "PASS",
        "profile": profile,
        "evaluator": _EVALUATOR,
        "annotation_adapter": annotation_adapter,
        "geometry_policy": _GEOMETRY_POLICY,
        "views": list(selected_views),
        "xgb_threshold": CANONICAL_XGB_THRESHOLD,
        "nms_iou": CANONICAL_NMS_IOU,
        "inputs": {
            "files": input_records,
            "files_sha256": compact_json_sha256(input_records),
        },
        "parameters": {
            "selected_images": list(selected_images),
            "cvat_label": cvat_label,
            "max_area_fraction": max_area_fraction,
            "al_margin": float(al_margin),
            "al_topk": al_topk,
        },
        "stages": stages,
        "source_outputs": source_outputs,
        "source_stream_policy": "CAPTURED_BOUNDED_NOT_RETAINED_PRIVATE_LOCATORS",
        "result_file": "CANONICAL_EVALUATION.json",
        "output_name": output.name,
    }
    require(len(canonical_json_bytes(summary)) <= 256 * 1024, "canonical evaluation summary exceeds its bound")
    write_new_json(staging / "CANONICAL_EVALUATION.json", summary)
    required, optional, source_limits = _source_output_contract(selected_views)
    final_allowed = {str(row["path"]) for row in source_outputs} | {"CANONICAL_EVALUATION.json"}
    require(
        required <= final_allowed <= required | optional | {"CANONICAL_EVALUATION.json"},
        "canonical evaluation final file set is invalid",
    )
    final_rows = manifest_rows(
        staging,
        allowed_files=final_allowed,
        file_size_limits={**source_limits, "CANONICAL_EVALUATION.json": 256 * 1024},
        max_total_bytes=_MAX_OUTPUT_BYTES,
    )
    require(len(final_rows) == len(final_allowed), "canonical evaluation final output closure changed")
    staging.chmod(0o755)
    fsync_directory(staging)
    publish_directory_noreplace(staging, output)
    return summary


__all__ = [
    "CANONICAL_EVALUATION_VIEWS",
    "CANONICAL_NMS_IOU",
    "CANONICAL_XGB_THRESHOLD",
    "evaluate_canonical_points",
]
