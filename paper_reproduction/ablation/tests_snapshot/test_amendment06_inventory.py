from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


STUDY = Path(__file__).resolve().parents[1]
CODE = STUDY / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import amendment_core as core  # noqa: E402


ACCEPTED_SMOKE = (
    STUDY
    / "results/run_20260820_104526_263432_bc48422a_amendment04_cuda_smoke"
)


def _inventory_bytes(rows: list[tuple[Path, int, str]]) -> bytes:
    body = ["path\tsize_bytes\tsha256"]
    body.extend(f"{path}\t{size}\t{digest}" for path, size, digest in rows)
    return ("\n".join(body) + "\n").encode("utf-8")


def _valid_inventory_bytes(source: Path) -> bytes:
    return _inventory_bytes([(
        source,
        source.stat().st_size,
        core.sha256_file(source),
    )])


def _write_inventory(run: Path, relative: str, data: bytes) -> Path:
    path = run / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _manifest_frames(paths: list[tuple[str, Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{
        "relative_path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": core.sha256_file(path),
    } for relative, path in paths]
    output = pd.DataFrame([{**row,
        "artifact_role": "provenance",
        "include_in_review_bundle": True,
        "include_in_models_bundle": False,
    } for row in rows], columns=[
        "relative_path", "size_bytes", "sha256", "artifact_role",
        "include_in_review_bundle", "include_in_models_bundle",
    ])
    bundle = pd.DataFrame(rows, columns=[
        "relative_path", "size_bytes", "sha256",
    ])
    return output, bundle


def _validate(run: Path, run_kind: str, paths: list[tuple[str, Path]]) -> dict:
    output, bundle = _manifest_frames(paths)
    return core.validate_reference_source_inventory(
        run,
        run_kind=run_kind,
        output_manifest=output,
        bundle_manifest=bundle,
    )


def _a06_reference_arguments() -> tuple[dict[str, str], dict[str, object]]:
    records = core.amendment06_live_code_tree(STUDY)
    allowed = {
        relative: str(record["sha256"])
        for relative, record in records.items()
    }
    summary = core.amendment06_code_tree_summary(records)
    ledger = STUDY / ".runtime/amendment06_full/authorized_change_ledger.jsonl"
    evidence: dict[str, object] = {
        "status": "PASS",
        "authorization": "AMENDMENT_06_EXTERNAL_REVIEW_ACCEPTED_ONE_FULL",
        "chain_status": "PASS",
        "baseline_tree_sha256": (
            "55992443799c7623911b2a908e79ff9c12e25e7ac9da15c2215ee1b01eac5075"
        ),
        "latest_committed_tree_sha256": summary["tree_sha256"],
        "latest_committed_file_count": summary["file_count"],
        "latest_committed_total_bytes": summary["total_bytes"],
        "allowed_after_hashes_sha256": core.sha256_bytes(
            core.canonical_json(dict(sorted(allowed.items())))
        ),
        "authorized_change_ledger_path": str(ledger),
        "authorized_change_ledger_sha256": core.sha256_file(ledger),
    }
    return allowed, evidence


def test_preflight_selects_root_inventory_with_nested_absent(tmp_path):
    run = tmp_path / "preflight"
    source = tmp_path / "source.dat"
    source.write_bytes(b"sealed source\n")
    relative = "provenance/source_input_hashes_post.tsv"
    inventory = _write_inventory(run, relative, _valid_inventory_bytes(source))

    result = _validate(run, "AMENDED_PREFLIGHT", [(relative, inventory)])

    assert result == {
        "status": "PASS",
        "selected_relative_path": relative,
        "sha256": core.sha256_file(inventory),
        "row_count": 1,
    }
    assert not (
        run / "provenance/preflight_snapshot/source_input_hashes_post.tsv"
    ).exists()


def test_smoke_selects_nested_inventory_with_root_absent(tmp_path):
    run = tmp_path / "smoke"
    source = tmp_path / "source.dat"
    source.write_bytes(b"sealed source\n")
    relative = "provenance/preflight_snapshot/source_input_hashes_post.tsv"
    inventory = _write_inventory(run, relative, _valid_inventory_bytes(source))

    result = _validate(
        run, "CUDA_SMOKE_NON_SCIENTIFIC", [(relative, inventory)],
    )

    assert result["selected_relative_path"] == relative
    assert not (run / "provenance/source_input_hashes_post.tsv").exists()


def test_smoke_ignores_manifest_bound_tampered_root_decoy(tmp_path):
    run = tmp_path / "smoke"
    source = tmp_path / "source.dat"
    source.write_bytes(b"sealed source\n")
    nested_relative = (
        "provenance/preflight_snapshot/source_input_hashes_post.tsv"
    )
    nested = _write_inventory(
        run, nested_relative, _valid_inventory_bytes(source),
    )
    root_relative = "provenance/source_input_hashes_post.tsv"
    root = _write_inventory(
        run,
        root_relative,
        _inventory_bytes([(tmp_path / "missing-source", 1, "0" * 64)]),
    )

    result = _validate(
        run,
        "CUDA_SMOKE_NON_SCIENTIFIC",
        [(nested_relative, nested), (root_relative, root)],
    )

    assert result["status"] == "PASS"
    assert result["selected_relative_path"] == nested_relative


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_smoke_rejects_selected_nested_drift_even_with_valid_root(
    tmp_path, mutation,
):
    run = tmp_path / "smoke"
    source = tmp_path / "source.dat"
    source.write_bytes(b"sealed source\n")
    root_relative = "provenance/source_input_hashes_post.tsv"
    root = _write_inventory(run, root_relative, _valid_inventory_bytes(source))
    nested_relative = (
        "provenance/preflight_snapshot/source_input_hashes_post.tsv"
    )
    nested = _write_inventory(
        run, nested_relative, _valid_inventory_bytes(source),
    )
    paths = [(root_relative, root), (nested_relative, nested)]
    if mutation == "missing":
        output, bundle = _manifest_frames(paths)
        nested.unlink()
    else:
        nested.write_bytes(_inventory_bytes([(
            source, source.stat().st_size, "0" * 64,
        )]))
        output, bundle = _manifest_frames(paths)

    with pytest.raises(core.IntegrityError, match="Selected source inventory|Source changed"):
        core.validate_reference_source_inventory(
            run,
            run_kind="CUDA_SMOKE_NON_SCIENTIFIC",
            output_manifest=output,
            bundle_manifest=bundle,
        )


def test_unsupported_run_kind_fails_before_path_access():
    class ExplodingPath:
        def __truediv__(self, _other):
            raise AssertionError("unsupported kind accessed the Run path")

    with pytest.raises(core.IntegrityError, match="Unsupported Run kind"):
        core.validate_reference_source_inventory(
            ExplodingPath(),  # type: ignore[arg-type]
            run_kind="UNSUPPORTED_KIND",
            output_manifest=None,  # type: ignore[arg-type]
            bundle_manifest=None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "defect",
    ["inventory_symlink", "source_symlink", "duplicate", "schema", "output_omission", "bundle_omission"],
)
def test_inventory_requires_regular_exact_unique_manifest_bound_tsv(
    tmp_path, defect,
):
    run = tmp_path / "full"
    source = tmp_path / "source.dat"
    source.write_bytes(b"sealed source\n")
    relative = "provenance/source_input_hashes_post.tsv"
    data = _valid_inventory_bytes(source)
    if defect == "duplicate":
        row = (source, source.stat().st_size, core.sha256_file(source))
        data = _inventory_bytes([row, row])
    elif defect == "schema":
        data = (
            f"path\tsha256\tsize_bytes\n{source}\t"
            f"{core.sha256_file(source)}\t{source.stat().st_size}\n"
        ).encode()
    elif defect == "source_symlink":
        link = tmp_path / "source-link.dat"
        link.symlink_to(source)
        data = _inventory_bytes([(
            link, source.stat().st_size, core.sha256_file(source),
        )])
    if defect == "inventory_symlink":
        real = _write_inventory(run, "real-inventory.tsv", data)
        inventory = run / relative
        inventory.parent.mkdir(parents=True, exist_ok=True)
        inventory.symlink_to(real)
    else:
        inventory = _write_inventory(run, relative, data)
    output, bundle = _manifest_frames([(relative, inventory)])
    if defect == "output_omission":
        output = output.iloc[0:0]
    elif defect == "bundle_omission":
        bundle = bundle.iloc[0:0]

    with pytest.raises(core.IntegrityError):
        core.validate_reference_source_inventory(
            run,
            run_kind="FULL_SCIENTIFIC_ABLATION",
            output_manifest=output,
            bundle_manifest=bundle,
        )


def test_exact_a05_smoke_passes_real_validator_and_final_reopen():
    identity = json.loads(
        (ACCEPTED_SMOKE / "config/run_identity.lock.json").read_text()
    )
    allowed, ledger_evidence = _a06_reference_arguments()

    validated = core.validate_amended_run_reference(
        ACCEPTED_SMOKE,
        expected_kind="CUDA_SMOKE_NON_SCIENTIFIC",
        allowed_states={"SMOKE_COMPLETE"},
        amendment06_allowed_after_hashes=allowed,
        amendment06_change_ledger_evidence=ledger_evidence,
    )

    reference = validated["reference_validation"]
    assert reference["source_inventory_selected_relative_path"] == (
        "provenance/preflight_snapshot/source_input_hashes_post.tsv"
    )
    assert reference["source_inventory_sha256"] == (
        "795279d24a3b7e07a65ed0f3a95d87969e19feafd58db182a5d26c7d2b3f762d"
    )
    assert reference["source_inventory_rows_verified"] == 15
    assert reference["smoke_semantic_completeness"][
        "amendment06_reference_code_chain"
    ]["status"] == "PASS"

    review = Path(str(ACCEPTED_SMOKE) + "_review_bundle.zip")
    adjacent = ACCEPTED_SMOKE.parent / (
        ACCEPTED_SMOKE.name + "_package_verification.json"
    )
    reopened = core.verify_published_review_bundle(
        ACCEPTED_SMOKE,
        review,
        json.loads(adjacent.read_text()),
    )
    assert reopened["status"] == "PASS"
    assert reopened["live_member_byte_equality"] == "PASS"


@pytest.mark.parametrize("defect", ["arbitrary", "missing", "extra"])
def test_amendment06_reference_code_map_is_exact(tmp_path, defect):
    allowed, evidence = _a06_reference_arguments()
    broken = dict(allowed)
    if defect == "arbitrary":
        broken[next(iter(broken))] = "0" * 64
    elif defect == "missing":
        broken.pop(next(iter(broken)))
    else:
        broken["tests/uncommitted_extra.py"] = "0" * 64

    with pytest.raises(core.IntegrityError, match="complete live code/test tree"):
        core.validate_amendment06_reference_code_chain(
            allowed_after_hashes=broken,
            ledger_evidence=evidence,
            study_root=STUDY,
        )


def test_full_selects_root_inventory(tmp_path):
    run = tmp_path / "full"
    source = tmp_path / "source.dat"
    source.write_bytes(b"sealed source\n")
    relative = "provenance/source_input_hashes_post.tsv"
    inventory = _write_inventory(run, relative, _valid_inventory_bytes(source))

    result = _validate(
        run, "FULL_SCIENTIFIC_ABLATION", [(relative, inventory)],
    )

    assert result["status"] == "PASS"
    assert result["selected_relative_path"] == relative
