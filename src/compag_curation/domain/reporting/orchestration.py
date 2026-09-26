"""Source batch-inference/merge contract used to prepare paper test-set tables."""

from __future__ import annotations

from csv import DictReader, DictWriter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from ..inference import (
    CommandRunner,
    InferenceRequest,
    ProcessResult,
    build_inference_argv,
)


class TestsetPreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TestsetBatchRequest:
    test_root: Path
    folders: tuple[str, ...]
    inference: InferenceRequest
    tiles_subdirectory: str = "tiles"


@dataclass(frozen=True)
class TestsetBatchResult:
    completed_outputs: tuple[Path, ...]
    failed_folders: tuple[str, ...]
    merged_features: Path
    merged_detections: Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return [dict(row) for row in DictReader(handle)]


def _merge_outputs(paths: Sequence[tuple[str, Path]], output: Path) -> Path:
    rows: list[dict[str, str]] = []
    fields: list[str] = ["img_folder"]
    seen = {"img_folder"}
    for folder, path in paths:
        if not path.is_file():
            continue
        for source in _read_rows(path):
            row = {"img_folder": folder, **source}
            rows.append(row)
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    if not rows:
        raise TestsetPreparationError(f"no source outputs were found for {output.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output


# SOURCE_CELL: NB-LIVE-0012-C0001
# SOURCE_STATEMENT_MAP: folder-loop-local-assets-argv-mode-logs-merge-failure -> execute_testset_batch
def execute_testset_batch(
    request: TestsetBatchRequest,
    *,
    runner: CommandRunner,
) -> TestsetBatchResult:
    """Run each explicit folder, retain logs, merge outputs, then expose failures."""

    completed: list[Path] = []
    failed: list[str] = []
    mode = request.inference.options.mode
    for folder in request.folders:
        image_root = request.test_root / folder
        images = image_root / request.tiles_subdirectory
        output = image_root / f"run_{mode}"
        output.mkdir(parents=True, exist_ok=True)
        log_directory = output / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        log_path = log_directory / f"run_{mode}.log"
        assets = replace(request.inference.assets, images=images, output=output)
        child = replace(request.inference, assets=assets)
        try:
            argv = build_inference_argv(child)
            process: ProcessResult = runner.run(
                argv,
                cwd=child.cwd,
                env={
                    **{str(key): str(value) for key, value in child.environment.items()},
                    "CJ_SAVE_ALL": "1",
                    "PYTHONUNBUFFERED": "1",
                },
                acceptable_exit_codes=frozenset({0}),
                capture_output=True,
            )
            log_path.write_text(
                process.stdout + process.stderr,
                encoding="utf-8",
                newline="\n",
            )
            for required in ("detections.csv", "features_pool.csv"):
                if not (output / required).is_file():
                    raise TestsetPreparationError(
                        f"missing {required} after inference for {folder}"
                    )
            completed.append(output)
        except (RuntimeError, TestsetPreparationError, ValueError, OSError) as exc:
            failed.append(folder)
            log_path.write_text(
                f"FAILED: {type(exc).__name__}: {exc}\n",
                encoding="utf-8",
                newline="\n",
            )

    features = _merge_outputs(
        [(folder, request.test_root / folder / f"run_{mode}" / "features_pool.csv") for folder in request.folders],
        request.test_root / f"test_features_pool__{mode}.csv",
    )
    detections = _merge_outputs(
        [(folder, request.test_root / folder / f"run_{mode}" / "detections.csv") for folder in request.folders],
        request.test_root / f"test_detections__{mode}.csv",
    )
    result = TestsetBatchResult(tuple(completed), tuple(failed), features, detections)
    if failed:
        raise TestsetPreparationError(
            f"some folders failed after merge: {', '.join(failed)}"
        )
    return result
