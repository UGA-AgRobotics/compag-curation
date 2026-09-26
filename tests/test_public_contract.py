from __future__ import annotations

import ast
import contextlib
import copy
import csv
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HEAVY_MODULES = {
    "cv2",
    "cupy",
    "dash",
    "hydra",
    "imblearn",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "plotly",
    "pycocotools",
    "scipy",
    "sklearn",
    "torch",
    "torchvision",
    "ultralytics",
    "xgboost",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _artifact(name: str, kind: str, command: str, variant: str) -> dict[str, str]:
    suffix = name if kind == "directory" else f"{name}.dat"
    return {
        "kind": kind,
        "path": f"synthetic/{command}/{variant}/{suffix}",
        "role": f"synthetic-{name}",
    }


def _workflow(command: str, variant: str) -> dict[str, object]:
    from compag_curation.config import VARIANT_ARTIFACT_KINDS, VARIANT_FIELDS

    required, _optional = VARIANT_FIELDS[(command, variant)]
    kinds = VARIANT_ARTIFACT_KINDS[(command, variant)]
    values: dict[str, object] = {"variant": variant}
    scalars: dict[str, object] = {
        "current_image": "IMG_1",
        "device": "cpu",
        "embed_backbone": "resnet50",
        "image_id": "synthetic-coverage-card-a",
        "image_name": "IMG_1",
        "mode": "xgb_recall",
        "review_mode": "xgb_recall",
    }
    for field in sorted(required - {"variant"}):
        if field in kinds:
            values[field] = _artifact(field, kinds[field], command, variant)
        elif field in scalars:
            values[field] = scalars[field]
        else:
            raise AssertionError(f"unhandled required scalar: {command}/{variant}/{field}")
    return values


def _fixture(selected: tuple[str, str] | None = None) -> dict[str, object]:
    from compag_curation.config import COMMANDS, VARIANTS_BY_COMMAND

    workflows = {
        command: _workflow(command, VARIANTS_BY_COMMAND[command][0]) for command in COMMANDS
    }
    if selected is not None:
        workflows[selected[0]] = _workflow(*selected)
    return {
        "schema": "compag-curation-domain-config/v2",
        "fixture_classification": "SYNTHETIC_NON_SCIENTIFIC_TEST_FIXTURE",
        "execution": {
            "authorized": False,
            "allow_download": False,
            "output_collision": "fail",
        },
        "notifications": {"enabled": False},
        "workflows": workflows,
    }


class PublicContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(os.environ["COMPAG_PUBLIC_ROOT"]).resolve(strict=True)

    def _module_file(self, module: str) -> Path:
        base = self.root / "src" / Path(module.replace(".", "/"))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate
        raise AssertionError(f"no source file for public module {module}")

    def _module_level_imports(self, module: str) -> set[str]:
        """Modules imported when ``module`` itself is executed (module level only)."""

        path = self._module_file(module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: set[str] = set()
        package = module.rsplit(".", 1)[0] if not path.name == "__init__.py" else module
        for node in tree.body:                       # module level only: a function body is not executed
            stack = [node]
            while stack:                             # descend through if/try/with blocks
                item = stack.pop()
                if isinstance(item, ast.Import):
                    found.update(alias.name for alias in item.names)
                elif isinstance(item, ast.ImportFrom):
                    if item.level:
                        parts = package.split(".")
                        base = ".".join(parts[: len(parts) - item.level + 1])
                        found.add(f"{base}.{item.module}" if item.module else base)
                    elif item.module:
                        found.add(item.module)
                elif isinstance(item, (ast.If, ast.Try, ast.With)):
                    stack.extend(item.body)
                    stack.extend(getattr(item, "orelse", []))
                    stack.extend(getattr(item, "finalbody", []))
                    for handler in getattr(item, "handlers", []):
                        stack.extend(handler.body)
        return found

    def test_public_module_imports_are_inert(self) -> None:
        """Importing the public modules must not pull in a heavy scientific module.

        Two independent checks, neither of which depends on what other tests have imported:

        * static - the module-level import graph of the required modules never reaches a heavy
          module (a regression is caught even if that module is already in ``sys.modules``);
        * dynamic - importing them here adds no heavy module to ``sys.modules`` that was not
          already there.  Modules preloaded by unrelated tests are reported, never purged.
        """

        rows = _read_csv(self.root / "manifests/public_module_inventory.csv")
        required = [row["module"] for row in rows if row["import_test_required"] == "YES"]
        self.assertGreater(len(required), 0)

        seen: set[str] = set()
        violations: list[str] = []
        queue = list(required)
        while queue:
            module = queue.pop()
            if module in seen:
                continue
            seen.add(module)
            for imported in sorted(self._module_level_imports(module)):
                root_name = imported.split(".", 1)[0]
                if root_name in HEAVY_MODULES:
                    violations.append(f"{module} imports {imported} at module level")
                elif root_name == "compag_curation":
                    try:
                        self._module_file(imported)
                    except AssertionError:
                        continue
                    queue.append(imported)
        self.assertEqual(violations, [], f"module-level heavy imports: {violations}")

        preloaded = {name.split(".", 1)[0] for name in sys.modules} & HEAVY_MODULES
        before = set(sys.modules)
        for module in required:
            importlib.import_module(module)
        newly_loaded = {name.split(".", 1)[0] for name in set(sys.modules) - before} & HEAVY_MODULES
        self.assertEqual(newly_loaded, set(),
                         f"importing the public modules loaded heavy modules: {sorted(newly_loaded)}"
                         f" (already loaded by earlier tests: {sorted(preloaded)})")

    def test_cli_help_every_subcommand(self) -> None:
        from compag_curation.cli import main
        from compag_curation.config import COMMANDS

        commands = (
            *COMMANDS,
            "doctor", "init-project", "inspect-data", "validate", "plan", "assets",
            "demo", "run", "convert-annotations", "verify-artifacts", "migrate-feature-csv",
            "active-learning", "review-ui",
        )
        argv_cases = (
            ["--help"],
            *([command, "--help"] for command in commands),
            ["assets", "fetch", "--help"],
            ["assets", "verify", "--help"],
            ["active-learning", "initial-export", "--help"],
            ["active-learning", "initial-resume", "--help"],
            ["active-learning", "begin-round", "--help"],
            ["active-learning", "resume-round", "--help"],
            ["active-learning", "begin-image-round", "--help"],
            ["active-learning", "resume-image-round", "--help"],
            ["active-learning", "import-r92-transfer-baseline", "--help"],
            ["active-learning", "infer-r92-image", "--help"],
            ["active-learning", "begin-transfer-image-round", "--help"],
            ["active-learning", "resume-transfer-image-round", "--help"],
            ["review-ui", "web", "--help"],
            ["review-ui", "desktop", "--help"],
            ["review-ui", "status", "--help"],
        )
        for argv in argv_cases:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(argv)
            self.assertEqual(caught.exception.code, 0)

    def test_all_variants_validate_and_plan(self) -> None:
        from compag_curation.cli import main
        from compag_curation.config import VARIANT_FIELDS, validate_mapping
        from compag_curation.plan import build_plan
        from compag_curation.registry import resolve

        self.assertEqual(len(VARIANT_FIELDS), 21)
        for command, variant in VARIANT_FIELDS:
            mapping = _fixture((command, variant))
            config = validate_mapping(copy.deepcopy(mapping), fixture=True, environment={})
            plan = build_plan(command, config, variant)
            handler_target = resolve(command, variant).handler_target
            handler_module, handler_name = handler_target.split(":", 1)
            handler = getattr(importlib.import_module(handler_module), handler_name)
            self.assertTrue(inspect.isroutine(handler) or callable(handler))
            payload = plan.to_dict()
            self.assertEqual(payload["variant"], variant)
            self.assertFalse(payload["scientific_execution"])
            self.assertEqual(payload["runtime_behavioral_parity"], "NOT_RETESTED_BY_DESIGN")
            self.assertEqual(payload["scientific_result_parity"], "NOT_RETESTED_BY_DESIGN")
            self.assertEqual(payload["policy"]["interpreter"], None)
            json.dumps(payload, sort_keys=True)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(mapping), encoding="utf-8")
                for mode in ("--validate", "--plan"):
                    capture = io.StringIO()
                    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                        rc = main(
                            [
                                command,
                                "--config",
                                str(path),
                                "--variant",
                                variant,
                                mode,
                            ]
                        )
                    self.assertEqual(rc, 0)
                    json.loads(capture.getvalue())

    def test_legacy_validate_and_plan_never_dispatch_execution_handlers(self) -> None:
        from compag_curation import public_pipeline
        from compag_curation.cli import main
        from compag_curation.config import VARIANT_FIELDS
        from compag_curation.registry import resolve

        handler_calls: list[str] = []

        def forbidden_handler(*_args: object, **_kwargs: object) -> object:
            handler_calls.append("called")
            raise AssertionError("legacy execution handler was dispatched")

        with contextlib.ExitStack() as stack:
            for command, variant in VARIANT_FIELDS:
                target = resolve(command, variant).handler_target
                module_name, symbol = target.split(":", 1)
                module = importlib.import_module(module_name)
                stack.enter_context(
                    mock.patch.object(module, symbol, side_effect=forbidden_handler)
                )
            infer = stack.enter_context(
                mock.patch.object(
                    public_pipeline,
                    "infer_bundle",
                    side_effect=forbidden_handler,
                )
            )
            for command, variant in VARIANT_FIELDS:
                mapping = _fixture((command, variant))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(mapping), encoding="utf-8")
                    for mode in ("--validate", "--plan"):
                        with (
                            contextlib.redirect_stdout(io.StringIO()),
                            contextlib.redirect_stderr(io.StringIO()),
                        ):
                            self.assertEqual(
                                main(
                                    [
                                        command,
                                        "--config",
                                        str(path),
                                        "--variant",
                                        variant,
                                        mode,
                                    ]
                                ),
                                0,
                            )
        self.assertEqual(handler_calls, [])
        infer.assert_not_called()

    def test_coverage_image_identity_is_explicit_and_portable(self) -> None:
        from compag_curation.config import ConfigurationError, validate_mapping
        from compag_curation.plan import build_plan

        variants = (
            "coverage-basic",
            "coverage-points-overlay",
            "coverage-predictions-overlay",
            "coverage-reviewability",
        )
        identities = ("synthetic-card-alpha.jpg", "synthetic-card-beta.jpg")
        for variant in variants:
            observed: list[str] = []
            for identity in identities:
                mapping = _fixture(("evaluate", variant))
                mapping["workflows"]["evaluate"]["image_id"] = identity  # type: ignore[index]
                config = validate_mapping(mapping, fixture=True, environment={})
                observed.append(str(config.workflow("evaluate")["image_id"]))
                plan = build_plan("evaluate", config, variant).to_dict()
                self.assertEqual(plan["steps"][1]["parameters"]["image_id"], identity)
            self.assertEqual(tuple(observed), identities)

            missing = _fixture(("evaluate", variant))
            del missing["workflows"]["evaluate"]["image_id"]  # type: ignore[index]
            with self.assertRaises(ConfigurationError):
                validate_mapping(missing, fixture=True, environment={})

        for invalid in ("", " synthetic-card", "synthetic-card ", ".", "..", "a/b", r"a\b", "a,b", "a\nb"):
            with self.subTest(invalid=invalid):
                mapping = _fixture(("evaluate", "coverage-basic"))
                mapping["workflows"]["evaluate"]["image_id"] = invalid  # type: ignore[index]
                with self.assertRaises(ConfigurationError):
                    validate_mapping(mapping, fixture=True, environment={})

    def test_operational_traceability(self) -> None:
        import re

        rows = _read_csv(self.root / "manifests/OPERATIONAL_CELL_TRACEABILITY.csv")
        self.assertEqual(len(rows), 84)
        self.assertEqual(len({row["cell_id"] for row in rows}), 84)
        symbols = {row["implementation_symbol"] for row in rows}
        nodes = {row["test_node_id"] for row in rows}
        self.assertEqual(len(symbols), 15)
        self.assertEqual(len(nodes), 15)
        for row in rows:
            self.assertEqual(row["public_release_disposition"], "PUBLIC_REQUIRED_IMPLEMENTED")
            module_name, symbol_name = row["implementation_symbol"].split(":", 1)
            symbol = getattr(importlib.import_module(module_name), symbol_name)
            self.assertTrue(inspect.isroutine(symbol) or callable(symbol))
            node_module, node_class, node_method = row["test_node_id"].rsplit(".", 2)
            case = getattr(importlib.import_module(node_module), node_class)
            self.assertTrue(issubclass(case, unittest.TestCase))
            self.assertTrue(callable(getattr(case, node_method)))
            self.assertIn(symbol_name, node_method)
        source_markers: set[str] = set()
        for path in (self.root / "src/compag_curation").rglob("*.py"):
            source_markers.update(
                re.findall(r"^# SOURCE_CELL: (NB-LIVE-[0-9]{4}-C[0-9]{4})$", path.read_text(encoding="utf-8"), re.MULTILINE)
            )
        self.assertEqual(source_markers, {row["cell_id"] for row in rows})

    def test_python_interpreter_resolution_and_consumer_guards(self) -> None:
        from compag_curation.contracts import ContractError, resolve_python_executable
        from compag_curation.domain.evaluation.coverage import (
            CoverageContractError,
            CoverageInputArtifact,
            CoverageRequest,
            build_coverage_argv,
        )
        from compag_curation.domain.inference import (
            InferenceAssets,
            InferenceContractError,
            InferenceOptions,
            InferenceRequest,
            build_inference_argv,
        )

        self.assertEqual(resolve_python_executable(), str(Path(sys.executable).resolve(strict=True)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable = root / "python-test"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            self.assertEqual(resolve_python_executable(str(executable)), str(executable))
            with mock.patch.dict(os.environ, {"PATH": str(root)}, clear=False):
                self.assertEqual(resolve_python_executable(executable.name), str(executable))

            missing = root / "missing-python"
            non_executable = root / "not-executable"
            non_executable.write_text("not executable\n", encoding="utf-8")
            non_executable.chmod(0o644)
            for value in (
                str(missing),
                "missing-python",
                "relative/python",
                "relative\\python",
                str(root),
                str(non_executable),
            ):
                with self.subTest(interpreter=value), mock.patch.dict(
                    os.environ, {"PATH": str(root)}, clear=False
                ):
                    with self.assertRaises(ContractError):
                        resolve_python_executable(value)

            digest = "0" * 64
            coverage = CoverageRequest(
                workflow="coverage-basic",
                image_id="IMG_SYNTHETIC",
                predictions=root / "predictions.csv",
                annotations=root / "annotations.xml",
                tile_index=CoverageInputArtifact(root / "tile-index.csv", digest),
                output=root / "coverage-output",
                script=CoverageInputArtifact(root / "evaluator.py", digest),
                python_executable=non_executable,
            )
            with self.assertRaises(CoverageContractError):
                build_coverage_argv(coverage)

            assets = InferenceAssets(
                images=root / "images",
                output=root / "inference-output",
                sam2_package_root=root / "sam2",
                sam2_package_root_sha256=digest,
                sam2_checkpoint=root / "sam2.ckpt",
                sam2_checkpoint_sha256=digest,
                sam2_config=root / "sam2.yaml",
                sam2_config_sha256=digest,
                prototype=root / "prototype.csv",
                prototype_sha256=digest,
                padded_prototype=root / "prototype-padded.csv",
                padded_prototype_sha256=digest,
                pca=root / "pca.npz",
                pca_sha256=digest,
                xgb_model=root / "model.ubj",
                xgb_model_sha256=digest,
                embed_backbone="resnet50",
                embed_backbone_weights=root / "resnet50.pth",
                embed_backbone_weights_sha256=digest,
            )
            with (
                mock.patch("compag_curation.domain.inference._require_file"),
                mock.patch("compag_curation.domain.inference._require_directory"),
                mock.patch("compag_curation.domain.inference._require_sha256"),
                self.assertRaises(InferenceContractError),
            ):
                build_inference_argv(
                    InferenceRequest(
                        assets,
                        InferenceOptions(python_executable=non_executable),
                    )
                )

    def test_subprocess_scanner_aliases_and_definition_filtering(self) -> None:
        from tools.public_qa import scan_subprocess_calls_source

        fixtures = {
            "module": "import subprocess\ndef boundary():\n    subprocess.run(['x'])\n",
            "module-alias": "import subprocess as sp\ndef boundary():\n    sp.run(['x'])\n",
            "direct": "from subprocess import run\ndef boundary():\n    run(['x'])\n",
            "direct-alias": "from subprocess import run as launch\ndef boundary():\n    launch(['x'])\n",
            "popen-module": "import subprocess\ndef boundary():\n    subprocess.Popen(['x'])\n",
            "popen-direct": "from subprocess import Popen\ndef boundary():\n    Popen(['x'])\n",
        }
        for label, source in fixtures.items():
            with self.subTest(label=label):
                records = scan_subprocess_calls_source(source)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["symbol"], "boundary")
        ignored = """
class ProtocolLike:
    def run(self, argv): ...
def run(argv):
    return argv
def use(app, obj):
    run([])
    app.run()
    obj.run()
"""
        self.assertEqual(scan_subprocess_calls_source(ignored), [])

    def test_public_qa_base_and_science_module_partition_is_closed(self) -> None:
        from tools.public_qa import (
            BASE_TEST_MODULES,
            SCIENCE_TEST_MODULES,
            test_module_partition,
        )

        record = test_module_partition(self.root)
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(tuple(record["base_modules"]), BASE_TEST_MODULES)
        self.assertEqual(tuple(record["science_modules"]), SCIENCE_TEST_MODULES)
        self.assertEqual(len(BASE_TEST_MODULES), 12)
        self.assertEqual(len(SCIENCE_TEST_MODULES), 31)
        self.assertFalse(set(BASE_TEST_MODULES) & set(SCIENCE_TEST_MODULES))
        self.assertEqual(
            record["science_disposition"],
            "DEFERRED_TO_LOCKED_SCIENCE_ACCEPTANCE_NOT_SKIPPED",
        )

    def test_notebook_inventory_partition(self) -> None:
        rows = _read_csv(self.root / "manifests/NOTEBOOK_MIGRATION_INVENTORY.csv")
        self.assertEqual(len(rows), 236)
        operational = [row for row in rows if row["classification"] == "OPERATIONAL"]
        non_operational = [row for row in rows if row["classification"] == "NON_OPERATIONAL"]
        self.assertEqual((len(operational), len(non_operational)), (84, 152))
        for row in operational:
            self.assertNotIn("NOT_APPLICABLE", (row["implementation_symbol"], row["cli_config_route"], row["test_node_id"]))
        for row in non_operational:
            self.assertEqual(row["implementation_symbol"], "NOT_APPLICABLE")
            self.assertEqual(row["cli_config_route"], "NOT_APPLICABLE")
            self.assertEqual(row["test_node_id"], "NOT_APPLICABLE")

    def test_backup_preservation_for_both_legacy_columns(self) -> None:
        from compag_curation.cli import main
        from compag_curation.contracts import ContractError
        from compag_curation.features.extraction import migrate_legacy_feature_csv

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            initial = b"file_name,ann_id,reviewed,class_id\nimage.jpg,1,yes,2\n"
            path.write_bytes(initial)
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(["migrate-feature-csv", "--path", str(path), "--column", "scale"]),
                    0,
                )
            self.assertEqual(json.loads(capture.getvalue())["backup_policy"], "PRESERVE_BACKUPS")
            scale_backup = path.with_suffix(".pre_scale.bak.csv")
            self.assertEqual(scale_backup.read_bytes(), initial)
            after_scale = path.read_bytes()
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(["migrate-feature-csv", "--path", str(path), "--column", "review_tag"]),
                    0,
                )
            self.assertTrue(json.loads(capture.getvalue())["changed"])
            review_backup = path.with_suffix(".pre_reviewtag.bak.csv")
            self.assertEqual(review_backup.read_bytes(), after_scale)
            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(
                header.split(","),
                ["file_name", "ann_id", "scale", "reviewed", "review_tag", "class_id"],
            )
            completed = path.read_bytes()
            with self.assertRaises(ContractError):
                migrate_legacy_feature_csv(path, "scale")
            with self.assertRaises(ContractError):
                migrate_legacy_feature_csv(path, "review_tag")
            self.assertEqual(path.read_bytes(), completed)
            self.assertEqual(scale_backup.read_bytes(), initial)
            self.assertEqual(review_backup.read_bytes(), after_scale)

            already_migrated = Path(directory) / "already-migrated.csv"
            already_migrated.write_bytes(completed)
            self.assertFalse(migrate_legacy_feature_csv(already_migrated, "scale"))
            self.assertFalse(migrate_legacy_feature_csv(already_migrated, "review_tag"))
            self.assertEqual(already_migrated.read_bytes(), completed)

            collision_source = Path(directory) / "collision.csv"
            collision_source.write_bytes(initial)
            collision_backup = collision_source.with_suffix(".pre_scale.bak.csv")
            collision_backup.write_bytes(b"pre-existing backup\n")
            with self.assertRaises(ContractError):
                migrate_legacy_feature_csv(collision_source, "scale")
            self.assertEqual(collision_source.read_bytes(), initial)
            self.assertEqual(collision_backup.read_bytes(), b"pre-existing backup\n")
            with self.assertRaises(ContractError):
                migrate_legacy_feature_csv(path, "unknown")

    def test_missing_inputs_fail_before_model_or_data_loading(self) -> None:
        from compag_curation.config import validate_mapping
        from compag_curation.contracts import ContractError, validate_artifact_preconditions
        from compag_curation.plan import build_plan

        with tempfile.TemporaryDirectory() as directory:
            mapping = _fixture(("infer", "sam2-xgb"))
            workflow = mapping["workflows"]["infer"]
            for name, value in workflow.items():
                if isinstance(value, dict) and "path" in value:
                    value["path"] = str(Path(directory) / name)
            config = validate_mapping(mapping, fixture=True, environment={})
            plan = build_plan("infer", config, "sam2-xgb")
            before = set(sys.modules)
            with self.assertRaisesRegex(ContractError, "required .* is unavailable"):
                validate_artifact_preconditions(plan)
            introduced = {name.split(".", 1)[0] for name in set(sys.modules) - before}
            self.assertFalse(introduced & HEAVY_MODULES)

    def test_deterministic_output_and_collision_contract(self) -> None:
        from compag_curation.contracts import ContractError
        from compag_curation.training.source_paths import prepare_source_output_directory

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "model-output"
            self.assertEqual(prepare_source_output_directory(output), output)
            self.assertEqual(prepare_source_output_directory(output), output)
            collision = Path(directory) / "collision"
            collision.write_text("occupied", encoding="utf-8")
            with self.assertRaises(ContractError):
                prepare_source_output_directory(collision)

    def test_manual_review_handoffs_are_real_and_inert(self) -> None:
        from compag_curation.domain.review import run_al_review, run_full_image_review
        from compag_curation.review.adapters import FullImageReviewConfig
        from compag_curation.review.workflow import ReviewAdvanceRequest

        for value in (run_al_review, run_full_image_review, FullImageReviewConfig, ReviewAdvanceRequest):
            self.assertTrue(callable(value))

    def test_missing_execution_authorization_blocks_before_handler_resolution(self) -> None:
        from compag_curation.config import validate_mapping
        from compag_curation.contracts import ExecutionNotAuthorized
        from compag_curation.execution import resolve_handler

        config = validate_mapping(_fixture(("infer", "sam2-xgb")), fixture=True, environment={})
        before = set(sys.modules)
        with self.assertRaises(ExecutionNotAuthorized):
            resolve_handler("infer", config, "sam2-xgb")
        after = set(sys.modules)
        introduced = {name.split(".", 1)[0] for name in after - before}
        self.assertFalse(introduced & HEAVY_MODULES)

    def test_production_ast_has_no_notebook_runtime_or_stubs(self) -> None:
        forbidden_tokens = (".ipynb", "get_ipython", "nbformat", "nbconvert", "IPython", "Jupyter")
        for path in sorted((self.root / "src/compag_curation").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for token in forbidden_tokens:
                self.assertNotIn(token, source, f"{token} in {path}")
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"eval", "exec"})
                if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                    if isinstance(node.exc.func, ast.Name):
                        self.assertNotEqual(node.exc.func.id, "NotImplementedError")
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                stub = len(node.body) == 1 and (
                    isinstance(node.body[0], ast.Pass)
                    or (
                        isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and node.body[0].value.value is Ellipsis
                    )
                )
                if not stub:
                    continue
                parent = parents.get(node)
                is_protocol = isinstance(parent, ast.ClassDef) and any(
                    (isinstance(base, ast.Name) and base.id == "Protocol")
                    or (isinstance(base, ast.Attribute) and base.attr == "Protocol")
                    for base in parent.bases
                )
                self.assertTrue(is_protocol, f"production stub at {path}:{node.lineno}")

    def test_package_inventory_and_license_scope(self) -> None:
        inventory = _read_csv(self.root / "manifests/public_file_inventory.csv")
        scope = _read_csv(self.root / "manifests/license_scope.csv")
        allowlist = json.loads(
            (self.root / "manifests/public_release_allowlist.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(allowlist),
            {"schema", "policy", "release_version", "path_encoding", "path_count", "paths_sha256", "paths"},
        )
        paths = allowlist["paths"]
        self.assertIs(type(paths), list)
        self.assertEqual(paths, sorted(set(paths)))
        self.assertEqual(allowlist["path_count"], len(paths))
        self.assertEqual(
            allowlist["paths_sha256"],
            hashlib.sha256(("\n".join(paths) + "\n").encode("ascii")).hexdigest(),
        )
        actual = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and ".git" not in path.relative_to(self.root).parts
        }
        inventory_name = "manifests/public_file_inventory.csv"
        self.assertEqual(actual, set(paths))
        self.assertEqual({row["relative_path"] for row in inventory}, set(paths) - {inventory_name})
        self.assertEqual({row["relative_path"] for row in scope}, set(paths))
        self.assertEqual(len({row["relative_path"] for row in inventory}), len(inventory))
        self.assertEqual(len({row["relative_path"] for row in scope}), len(scope))
        scope_by_path = {row["relative_path"]: row["license_class"] for row in scope}
        allowed = {
            "MIT",
            "CC-BY-4.0",
            "LEGAL_NOTICE",
            "MIT_LICENSE_TEXT",
            "CC_BY_4_0_NOTICE",
        }
        self.assertTrue(set(scope_by_path.values()) <= allowed)
        for row in inventory:
            self.assertEqual(row["license_class"], scope_by_path[row["relative_path"]])

    def test_bootstrap_lock_closes_wheel_runtime_dependency(self) -> None:
        lock = (
            self.root / "requirements/constraints-bootstrap-cp312-linux-x86_64.txt"
        ).read_text(encoding="utf-8")
        rows = [line for line in lock.splitlines() if line and not line.startswith("#")]
        distributions = {
            re.fullmatch(r"([A-Za-z0-9_.-]+) @ .+", line).group(1).lower()
            for line in rows
        }
        self.assertEqual(distributions, {"packaging", "setuptools", "wheel"})
        self.assertIn(
            "packaging @ https://files.pythonhosted.org/packages/63/34/"
            "ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/"
            "packaging-26.3-py3-none-any.whl#sha256="
            "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c",
            rows,
        )

    def test_public_schemas_and_examples_round_trip(self) -> None:
        import re
        import tomllib

        from compag_curation.point_annotations import inspect_point_annotations
        from compag_curation.public_config import (
            PublicConfigurationError,
            load_public_config,
            validate_public_mapping,
        )
        from compag_curation.review.exchange import validate_review_table, write_review_import
        from compag_curation.public_io import safe_relative, strict_json_bytes

        schema_names = (
            "canonical_point_annotations.schema.json",
            "public_project.schema.json",
            "review_table.schema.json",
        )
        for name in schema_names:
            with self.subTest(schema=name):
                root_payload = (self.root / "schemas" / name).read_bytes()
                packaged_payload = (self.root / "src/compag_curation/resources" / name).read_bytes()
                self.assertEqual(root_payload, packaged_payload)
                schema = strict_json_bytes(root_payload, f"packaged schema {name}")
                self.assertIsInstance(schema, dict)
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertEqual(schema["type"], "object")
                self.assertFalse(schema["additionalProperties"])

        project_schema = strict_json_bytes(
            (self.root / "schemas/public_project.schema.json").read_bytes(),
            "public project schema",
        )
        relative_pattern = project_schema["$defs"]["relativePath"]["pattern"]
        for valid in ("data/images", "assets/sam2.1_hiera_tiny.pt", "a", "A-1/b_2"):
            self.assertIsNotNone(re.fullmatch(relative_pattern, valid))
        for invalid in ("data/with space", "data/é", ".hidden/data", f"data/{'x' * 256}"):
            self.assertIsNone(re.fullmatch(relative_pattern, invalid))
        canonical_schema = strict_json_bytes(
            (self.root / "schemas/canonical_point_annotations.schema.json").read_bytes(),
            "canonical point schema",
        )
        self.assertEqual(canonical_schema["properties"]["rows"]["maxItems"], 250000)

        fixture = self.root / "examples/public_fixture"
        manifest = strict_json_bytes(
            (fixture / "FIXTURE_MANIFEST.json").read_bytes(),
            "public fixture manifest",
        )
        self.assertIsInstance(manifest, dict)
        self.assertEqual(
            set(manifest),
            {"schema", "status", "source", "research_data_derivative", "license", "members"},
        )
        self.assertEqual(manifest["status"], "PASS")
        member_paths = [row["path"] for row in manifest["members"]]
        self.assertEqual(member_paths, sorted(member_paths))
        self.assertEqual(len(member_paths), len(set(member_paths)))
        for row in manifest["members"]:
            self.assertEqual(set(row), {"path", "sha256", "size_bytes"})
            self.assertIsInstance(row["path"], str)
            self.assertIsInstance(row["sha256"], str)
            self.assertIs(type(row["size_bytes"]), int)
            self.assertEqual(safe_relative(row["path"], "fixture member").as_posix(), row["path"])
            self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]))
            path = fixture / row["path"]
            payload = path.read_bytes()
            self.assertEqual(row["size_bytes"], len(payload))
            self.assertEqual(row["sha256"], hashlib.sha256(payload).hexdigest())

        reviewed_rows, review_summary = validate_review_table(
            fixture / "review_request_example.csv",
            fixture / "reviewed_example.csv",
        )
        self.assertEqual(review_summary["rows"], 8)
        self.assertEqual(review_summary["positive_rows"], 4)
        self.assertEqual(review_summary["negative_rows"], 4)
        labels_by_group: dict[str, set[int]] = {}
        for row in reviewed_rows:
            labels_by_group.setdefault(row.group_id, set()).add(row.label)
        self.assertEqual(len(labels_by_group), 4)
        self.assertTrue(all(labels == {0, 1} for labels in labels_by_group.values()))
        canonical_rows, canonical_summary = validate_review_table(
            fixture / "canonical_review_request_example.csv",
            fixture / "canonical_reviewed_example.csv",
        )
        self.assertEqual(canonical_summary["review_contract"], "CANONICAL_ACTION_WEIGHTED_V2")
        self.assertEqual(canonical_summary["rows"], 2)
        self.assertEqual(canonical_summary["action_counts"]["accept"], 1)
        self.assertEqual(canonical_summary["action_counts"]["sus_accept"], 1)
        self.assertEqual(canonical_summary["effective_weight"], 1.4)
        self.assertEqual(
            {(row.review_action, row.review_weight) for row in canonical_rows},
            {("accept", 1.0), ("sus_accept", 0.4)},
        )
        point_rows, point_summary = inspect_point_annotations(fixture / "annotations/points.xml")
        self.assertEqual(point_summary["point_rows"], 2)
        self.assertEqual({row["image"] for row in point_rows}, {"synthetic-heldout__card.ppm"})
        config = load_public_config(fixture / "config.toml", check_local=False)
        self.assertEqual(config.profile, "public-safe-balanced-v1")
        self.assertEqual(config.device, "cpu")
        config_mapping = tomllib.loads((fixture / "config.toml").read_text(encoding="utf-8"))
        config_sha256 = hashlib.sha256((fixture / "config.toml").read_bytes()).hexdigest()
        for invalid in ("data/with space", "data/é", ".hidden/data", f"data/{'x' * 256}"):
            with self.subTest(invalid_project_path=invalid):
                candidate = copy.deepcopy(config_mapping)
                candidate["paths"]["images"] = invalid
                with self.assertRaises(PublicConfigurationError):
                    validate_public_mapping(
                        candidate,
                        config_path=fixture / "config.toml",
                        config_sha256=config_sha256,
                    )

        generator_path = self.root / "tools/generate_public_example_fixture.py"
        spec = importlib.util.spec_from_file_location("compag_public_fixture_generator", generator_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "generated"
            with mock.patch.object(sys, "argv", [str(generator_path), "--output", str(generated)]):
                self.assertEqual(generator.main(), 0)
            self.assertEqual(
                (generated / "FIXTURE_MANIFEST.json").read_bytes(),
                (fixture / "FIXTURE_MANIFEST.json").read_bytes(),
            )
            generated_files = {
                path.relative_to(generated).as_posix()
                for path in generated.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            self.assertEqual(generated_files, {"FIXTURE_MANIFEST.json", *member_paths})
            for relative in member_paths:
                self.assertEqual((generated / relative).read_bytes(), (fixture / relative).read_bytes())
            before_retry = {
                relative: (generated / relative).read_bytes()
                for relative in generated_files
            }
            with mock.patch.object(sys, "argv", [str(generator_path), "--output", str(generated)]):
                with self.assertRaises(SystemExit):
                    generator.main()
            self.assertEqual(
                {
                    relative: (generated / relative).read_bytes()
                    for relative in generated_files
                },
                before_retry,
            )
            imported = Path(directory) / "reviewed-roundtrip.csv"
            write_review_import(reviewed_rows, imported)
            self.assertEqual(imported.read_bytes(), (fixture / "reviewed_example.csv").read_bytes())

    def test_sdist_normalizer_is_deterministic_and_payload_preserving(self) -> None:
        import gzip
        import tarfile

        from tools.normalize_sdist import SdistNormalizationError, normalize_sdist

        def raw_sdist(path: Path, *, reverse: bool, mtime: int) -> None:
            rows = [("compag_curation-1.0.0/", True), ("compag_curation-1.0.0/value.txt", False)]
            if reverse:
                rows.reverse()
            with path.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=mtime) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w") as archive:
                        for name, is_directory in rows:
                            info = tarfile.TarInfo(name)
                            info.type = tarfile.DIRTYPE if is_directory else tarfile.REGTYPE
                            info.mode = 0o700 if is_directory else 0o600
                            info.uid = mtime
                            info.gid = mtime
                            info.mtime = mtime
                            payload = b"" if is_directory else b"payload\n"
                            info.size = len(payload)
                            archive.addfile(info, None if is_directory else io.BytesIO(payload))

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first = workspace / "first.tar.gz"
            second = workspace / "second.tar.gz"
            raw_sdist(first, reverse=False, mtime=1)
            raw_sdist(second, reverse=True, mtime=2)
            first_output = workspace / "first-normalized.tar.gz"
            second_output = workspace / "second-normalized.tar.gz"
            first_record = normalize_sdist(first, first_output, 1787788800)
            second_record = normalize_sdist(second, second_output, 1787788800)
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            self.assertEqual(first_record["members"], 2)
            self.assertEqual(second_record["members"], 2)
            with tarfile.open(first_output, "r:gz") as archive:
                members = archive.getmembers()
                self.assertEqual([member.name for member in members], [
                    "compag_curation-1.0.0",
                    "compag_curation-1.0.0/value.txt",
                ])
                self.assertTrue(all(member.mtime == 1787788800 for member in members))
                self.assertEqual([member.mode for member in members], [0o755, 0o644])
                self.assertEqual(archive.extractfile(members[1]).read(), b"payload\n")

            protected = workspace / "protected.tar.gz"
            protected.write_bytes(b"foreign\n")
            with self.assertRaises(SdistNormalizationError):
                normalize_sdist(first, protected, 1787788800)
            self.assertEqual(protected.read_bytes(), b"foreign\n")

            input_link = workspace / "input-link.tar.gz"
            input_link.symlink_to(first.name)
            with self.assertRaises(SdistNormalizationError):
                normalize_sdist(input_link, workspace / "linked-input-out.tar.gz", 1787788800)
            output_target = workspace / "output-target.tar.gz"
            output_target.write_bytes(b"target\n")
            output_link = workspace / "output-link.tar.gz"
            output_link.symlink_to(output_target.name)
            with self.assertRaises(SdistNormalizationError):
                normalize_sdist(first, output_link, 1787788800)
            self.assertEqual(output_target.read_bytes(), b"target\n")

            def archive_with(path: Path, members: list[tuple[str, bytes, bytes]]) -> None:
                with tarfile.open(path, "w:gz") as archive:
                    for name, kind, payload in members:
                        info = tarfile.TarInfo(name)
                        info.type = kind
                        info.size = len(payload) if kind == tarfile.REGTYPE else 0
                        archive.addfile(info, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)

            invalid_archives = {
                "traversal": [("root/../escape", tarfile.REGTYPE, b"x")],
                "symlink": [("root/link", tarfile.SYMTYPE, b"")],
                "duplicate": [
                    ("root/value", tarfile.REGTYPE, b"a"),
                    ("root/value", tarfile.REGTYPE, b"b"),
                ],
                "casefold": [
                    ("root/A", tarfile.REGTYPE, b"a"),
                    ("root/a", tarfile.REGTYPE, b"b"),
                ],
                "nfc": [
                    ("root/\u00e9", tarfile.REGTYPE, b"a"),
                    ("root/e\u0301", tarfile.REGTYPE, b"b"),
                ],
                "multi-root": [
                    ("one/value", tarfile.REGTYPE, b"a"),
                    ("two/value", tarfile.REGTYPE, b"b"),
                ],
            }
            for label, members in invalid_archives.items():
                with self.subTest(invalid_sdist=label):
                    candidate = workspace / f"{label}.tar.gz"
                    archive_with(candidate, members)
                    with self.assertRaises(SdistNormalizationError):
                        normalize_sdist(candidate, workspace / f"{label}-out.tar.gz", 1787788800)
            malformed = workspace / "malformed.tar.gz"
            malformed.write_bytes(b"not a tar archive\n")
            with self.assertRaises(SdistNormalizationError):
                normalize_sdist(malformed, workspace / "malformed-out.tar.gz", 1787788800)

    def test_public_paths_and_privacy(self) -> None:
        import ast

        from tools.privacy_scan import RULE_CLASSES, scan_tree

        result = scan_tree(self.root)
        self.assertEqual(result["status"], "PASS", result["findings"])
        self.assertEqual(result["effective_findings"], 0)

        observed: set[str] = set()
        text_fixtures = {
            "PRIVATE_KEY": b"-----BEGIN " + b"PRIVATE KEY-----\n",
            "AWS_ACCESS_KEY": b"AKIA" + b"A" * 16,
            "GITHUB_TOKEN": b"ghp_" + b"a" * 36,
            "SLACK_TOKEN": b"xoxb-" + b"a" * 24,
            "GOOGLE_API_KEY": b"AIza" + b"a" * 35,
            "POSIX_HOME_LOCATOR": b"/ho" + b"me/private_user/data",
            "MAC_HOME_LOCATOR": b"/Us" + b"ers/private_user/data",
            "WSL_MOUNT_LOCATOR": b"/mn" + b"t/c/Us" + b"ers/private_user/data",
            "WINDOWS_HOME_LOCATOR": b"C:" + b"\\Users\\private_user\\data",
            "WSL_UNC_LOCATOR": b"\\\\ws" + b"l.localhost\\Ubuntu\\home",
            "FILE_URI": b"fi" + b"le:///private/data",
            "INTERNAL_CONTROL_MARKER": b"EXPECTED" + b"_PROMPT_fixture",
            "TRANSITION_MARKER": b"pending " + b"upload",
            "GITLAB_TOKEN": b"glpat-" + b"a" * 24,
            "PYPI_TOKEN": b"pypi-" + b"a" * 48,
            "HUGGINGFACE_TOKEN": b"hf_" + b"a" * 32,
            "JWT_TOKEN": b"eyJ" + b"a" * 20 + b"." + b"b" * 20 + b"." + b"c" * 20,
            "BEARER_TOKEN": b"Authorization: Bearer " + b"aB3_" * 8,
            "URL_USERINFO": b"https" + b"://user:private-value@example.invalid/path",
            "GENERIC_SECRET_ASSIGNMENT": b"password=\"A7f/3zQp9Lm2Vx8Nc4Rt6Yw1\"",
            "HIGH_ENTROPY_SECRET": b"password=\"A7f/3zQp9Lm2Vx8Nc4Rt6Yw1\"",
            "PRIVATE_DEPENDENCY_REFERENCE": b"--extra-" + b"index-url https" + b"://local" + b"host/simple",
        }
        for expected, payload in text_fixtures.items():
            with self.subTest(privacy_rule=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "fixture.txt").write_bytes(payload)
                rules = {item["rule"] for item in scan_tree(root)["findings"]}
                self.assertIn(expected, rules)
                observed.update(rules)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture.txt").write_bytes(b"sk-" + b"a" * 32)
            result = scan_tree(root)
            self.assertGreater(result["effective_findings"], 0, "Synthetic service token must be detected")
            observed.update(item["rule"] for item in result["findings"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "local-ui.txt").write_text(
                "Open the private local application at http://127.0.0.1:43123/\n",
                encoding="utf-8",
            )
            rules = {item["rule"] for item in scan_tree(root)["findings"]}
            self.assertNotIn("PRIVATE_DEPENDENCY_REFERENCE", rules)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "__pycache__").mkdir()
            (root / "model.ckpt").write_bytes(b"fixture")
            prompt_name = "REVIEWER_COMMENTS_fixture.txt"
            (root / prompt_name).write_text("fixture", encoding="utf-8")
            (root / "disguised.dat").write_bytes(b"PK" + b"\x03\x04fixture")
            (root / "target.txt").write_text("fixture", encoding="utf-8")
            (root / "link.txt").symlink_to("target.txt")
            (root / "Case.txt").write_text("a", encoding="utf-8")
            (root / "case.txt").write_text("b", encoding="utf-8")
            (root / "\u00e9.txt").write_text("a", encoding="utf-8")
            (root / "e\u0301.txt").write_text("b", encoding="utf-8")
            rules = {item["rule"] for item in scan_tree(root)["findings"]}
            observed.update(rules)
        expected_structural = {
            "CACHE_OR_PRIVATE_COMPONENT",
            "DATA_MODEL_OR_BINARY_MEMBER",
            "PROMPT_OR_MANUSCRIPT_MEMBER",
            "PROHIBITED_BINARY_MAGIC",
            "SYMLINK_OR_SPECIAL",
            "CASEFOLD_COLLISION",
            "NFC_COLLISION",
        }
        self.assertTrue(expected_structural <= observed)
        self.assertTrue(set(text_fixtures) <= observed)
        self.assertTrue(observed <= set(RULE_CLASSES))

        scanner_source = (self.root / "tools/privacy_scan.py").read_text(encoding="utf-8")

        def folded_literals(source: str) -> list[bytes]:
            def fold(node: ast.AST) -> bytes | str | None:
                if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, str)):
                    return node.value
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    left = fold(node.left)
                    right = fold(node.right)
                    if type(left) is type(right) and isinstance(left, (bytes, str)):
                        return left + right
                return None

            values: list[bytes] = []
            for node in ast.walk(ast.parse(source)):
                value = fold(node)
                if isinstance(value, str):
                    values.append(value.encode("utf-8"))
                elif isinstance(value, bytes):
                    values.append(value)
            return values

        forbidden_home = re.compile(rb"/home/[A-Za-z0-9._-]+(?:/|$)")
        self.assertFalse(
            any(forbidden_home.search(value) for value in folded_literals(scanner_source)),
            "privacy scanner source must not reconstruct a personal home locator",
        )
        synthetic_source = 'VALUE = b"/ho" + b"me/example-user/private-data"\n'
        self.assertTrue(
            any(forbidden_home.search(value) for value in folded_literals(synthetic_source))
        )

        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "ordinary.txt"
            subject.write_text("ordinary\n", encoding="ascii")
            invalid = scan_tree(subject)
            self.assertEqual(invalid["status"], "FAIL")
            self.assertIn("INVALID_SUBJECT", {item["rule"] for item in invalid["findings"]})

    def test_reviewed_r92_binary_privacy_exception_is_exact_and_tamper_closed(self) -> None:
        from tools.privacy_scan import (
            REVIEWED_SAFE_BINARY_SHA256,
            _scan_payload,
        )

        for relative, expected_sha256 in REVIEWED_SAFE_BINARY_SHA256.items():
            with self.subTest(relative=relative):
                payload = (self.root / relative).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha256)
                findings: list[object] = []
                _scan_payload(payload, relative, findings)
                self.assertEqual(findings, [])

                wheel_relative = relative.removeprefix("src/")
                wheel_findings: list[object] = []
                _scan_payload(
                    payload,
                    f"fixture.whl!{wheel_relative}",
                    wheel_findings,
                )
                self.assertEqual(wheel_findings, [])

                tampered: list[object] = []
                _scan_payload(payload + b"\x00", relative, tampered)
                self.assertTrue(
                    {item.rule for item in tampered}
                    & {"DATA_MODEL_OR_BINARY_MEMBER", "PROHIBITED_BINARY_MAGIC"}
                )

    def test_privacy_archive_scan_opens_and_bounds_members(self) -> None:
        import tarfile
        import zipfile

        from tools.privacy_scan import scan_archive

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.whl"
            with zipfile.ZipFile(valid, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("package/module.py", "VALUE = 1\n")
            result = scan_archive(valid)
            self.assertEqual(result["status"], "PASS", result["findings"])
            self.assertEqual(result["scanned_files"], 1)

            leaking = root / "leaking.zip"
            with zipfile.ZipFile(leaking, "w") as archive:
                archive.writestr("package/value.txt", b"pending " + b"upload")
            rules = {item["rule"] for item in scan_archive(leaking)["findings"]}
            self.assertIn("TRANSITION_MARKER", rules)

            traversal = root / "traversal.zip"
            with zipfile.ZipFile(traversal, "w") as archive:
                archive.writestr("../escape.txt", "x")
            rules = {item["rule"] for item in scan_archive(traversal)["findings"]}
            self.assertIn("UNSAFE_PATH", rules)

            inner_buffer = io.BytesIO()
            with zipfile.ZipFile(inner_buffer, "w") as archive:
                archive.writestr("inner.txt", "ordinary\n")
            nested = root / "nested.zip"
            with zipfile.ZipFile(nested, "w") as archive:
                archive.writestr("nested.bin", inner_buffer.getvalue())
            rules = {item["rule"] for item in scan_archive(nested)["findings"]}
            self.assertIn("NESTED_ARCHIVE", rules)

            tar_path = root / "traversal.tar.gz"
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo("root/../escape.txt")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            rules = {item["rule"] for item in scan_archive(tar_path)["findings"]}
            self.assertIn("UNSAFE_PATH", rules)

    def test_metadata_ci_citation_and_internal_links(self) -> None:
        pyproject = (self.root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('version = "1.9.3"', pyproject)
        self.assertIn('requires-python = ">=3.12,<3.13"', pyproject)
        self.assertIn('compag-curation = "compag_curation.cli:main"', pyproject)
        cff = (self.root / "CITATION.cff").read_text(encoding="utf-8")
        for marker in ("cff-version: 1.2.0", "type: software", "version: 1.9.3"):
            self.assertIn(marker, cff)
        for forbidden in ("date-released:", "doi:", "url:", "orcid:", "preferred-citation:", "COMPAG-D-26-02120"):
            self.assertNotIn(forbidden, cff)
        workflow = json.loads((self.root / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        matrix = workflow["jobs"]["qa"]["strategy"]["matrix"]["python-version"]
        self.assertEqual(matrix, ["3.12.7"])
        serialized = json.dumps(workflow, sort_keys=True).lower()
        for forbidden in ("pull_request_target", '"secrets"', '"write"', "publish", "upload", "deploy"):
            self.assertNotIn(forbidden, serialized)
        uses = [
            step["uses"]
            for step in workflow["jobs"]["qa"]["steps"]
            if "uses" in step
        ]
        self.assertEqual(
            uses,
            [
                "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
                "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
            ],
        )
        self.assertTrue(all(__import__("re").fullmatch(r"[^@]+@[0-9a-f]{40}", value) for value in uses))
        self.assertFalse(workflow["jobs"]["qa"]["steps"][0]["with"]["persist-credentials"])
        metadata_step = next(step["run"] for step in workflow["jobs"]["qa"]["steps"] if step.get("name") == "Reviewed metadata and source privacy")
        self.assertIn("build_public_metadata.py", metadata_step)
        self.assertIn("--check", metadata_step)
        self.assertIn("privacy_scan.py", metadata_step)
        self.assertIn('--root "$RUNNER_TEMP/source"', metadata_step)
        source_zip_step = next(
            step["run"]
            for step in workflow["jobs"]["qa"]["steps"]
            if step.get("name") == "Deterministic allowlist-closed public source ZIP"
        )
        self.assertEqual(source_zip_step.count("build_public_zip.py"), 2)
        self.assertEqual(source_zip_step.count("--source-date-epoch"), 2)
        self.assertIn("cmp -s --", source_zip_step)
        self.assertIn("privacy_scan.py", source_zip_step)
        self.assertIn("--archive", source_zip_step)
        self.assertIn("-m zipfile -e", source_zip_step)
        self.assertIn("build_public_metadata.py", source_zip_step)
        self.assertIn("--check", source_zip_step)
        self.assertIn(
            '--root "$RUNNER_TEMP/public-zip-extracted/COMPAG_D_26_02120_PUBLIC_RUNNABLE_GITHUB_v1.9.3"',
            source_zip_step,
        )
        build_step = next(step["run"] for step in workflow["jobs"]["qa"]["steps"] if step.get("name") == "Build and check distributions")
        self.assertIn("constraints-bootstrap-cp312-linux-x86_64.txt", build_step)
        self.assertIn("constraints-dev-cp312-linux-x86_64.txt", build_step)
        self.assertEqual(build_step.count("--require-hashes --no-deps"), 2)
        self.assertIn("build --no-isolation", build_step)
        self.assertEqual(build_step.count("privacy_scan.py --archive"), 2)
        self.assertIn('privacy_scan.py --archive "$RUNNER_TEMP/dist/compag_curation-1.9.3-py3-none-any.whl"', build_step)
        self.assertIn('privacy_scan.py --archive "$RUNNER_TEMP/dist/compag_curation-1.9.3.tar.gz"', build_step)
        base_step = next(step["run"] for step in workflow["jobs"]["qa"]["steps"] if step.get("name") == "Clean base installation")
        self.assertIn("--no-deps", base_step)
        self.assertIn("--no-compile", base_step)
        self.assertIn(' -I -B -c "from compag_curation.runtime_lock import package_implementation_identity', base_step)
        self.assertIn("SYS_PREFIX_DISTRIBUTION_NON_EDITABLE_ISOLATED_NO_BYTECODE", base_step)
        self.assertIn("raise SystemExit", base_step)
        self.assertNotIn(" assert ", base_step)
        quick_step = next(step["run"] for step in workflow["jobs"]["qa"]["steps"] if step.get("name") == "Installed quick demo")
        self.assertIn("constraints-quick-demo-cp312-linux-x86_64.txt", quick_step)
        self.assertIn("--require-hashes --no-deps", quick_step)
        self.assertNotIn("[quick-demo]", quick_step)
        self.assertGreaterEqual(quick_step.count("--no-deps"), 2)
        self.assertIn("doctor --profile quick", quick_step)
        self.assertIn("demo --profile quick", quick_step)
        self.assertIn("manifest_rows", quick_step)
        self.assertIn("members_sha256", quick_step)
        self.assertIn("m['members']==rows", quick_step)
        self.assertIn("m['members_sha256']==compact_json_sha256(rows)", quick_step)
        self.assertIn("raise SystemExit", quick_step)
        self.assertNotIn(" assert ", quick_step)
        gpu_step = next(
            step["run"]
            for step in workflow["jobs"]["qa"]["steps"]
            if step.get("name") == "Mocked GPU contracts (no hosted accelerator claim)"
        )
        for module in (
            "tests.test_gpu_runtime_profile",
            "tests.test_canonical_gpu_kernels",
            "tests.test_canonical_service_gpu_profile",
            "tests.test_canonical_active_learning_gpu_profile",
            "tests.test_full_image_backend",
            "tests.test_full_image_features",
            "tests.test_full_image_review_session",
            "tests.test_full_image_service_integration",
            "tests.test_model_bundle_gpu_profile",
        ):
            self.assertIn(module, gpu_step)
        self.assertNotIn("doctor --profile science-gpu", gpu_step)
        self.assertNotIn("nvidia-smi", gpu_step)
        for markdown in self.root.rglob("*.md"):
            text = markdown.read_text(encoding="utf-8")
            for target in __import__("re").findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                local = target.split("#", 1)[0]
                if not local:
                    continue
                self.assertTrue((markdown.parent / local).resolve().exists(), f"broken link {target} in {markdown}")

    def test_documented_command_matrix_closure(self) -> None:
        matrix = _read_csv(self.root / "manifests/DOCUMENTED_COMMAND_EXECUTION_MATRIX.csv")
        runner_path = self.root / "tools/run_documented_commands.py"
        spec = importlib.util.spec_from_file_location("compag_documented_matrix_contract", runner_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        documented = runner.documented_commands(self.root)

        matrix_path, validated_matrix = runner.read_validated_matrix(self.root)
        self.assertEqual(matrix_path, self.root / "manifests/DOCUMENTED_COMMAND_EXECUTION_MATRIX.csv")
        self.assertEqual(validated_matrix, matrix)
        self.assertGreater(len(documented), 0)
        self.assertGreaterEqual(sum(len(value) for value in documented.values()), len(documented))
        self.assertEqual(len(matrix), len(documented) + 83)
        self.assertEqual(len({row["command_id"] for row in matrix}), len(matrix))
        self.assertEqual(len({row["command"] for row in matrix}), len(matrix))
        documented_rows = [row for row in matrix if row["command_id"].startswith("CMD-DOC-")]
        self.assertEqual(
            [row["command_id"] for row in documented_rows],
            [f"CMD-DOC-{index:03d}" for index in range(1, len(documented) + 1)],
        )
        self.assertEqual([row["command"] for row in documented_rows], list(documented))
        for row in documented_rows:
            self.assertEqual(row["command_source_location"], ";".join(documented[row["command"]]))
        self.assertEqual(runner.regenerated_matrix_rows(self.root, matrix), matrix)

        expected_safe_documented = {
            "python -I -S -B tools/public_qa.py --root .",
        }
        self.assertEqual(set(runner.SAFE_DOCUMENTED_COMMANDS), expected_safe_documented)
        self.assertEqual(
            {row["command"] for row in documented_rows if row["safe_to_execute"] == "YES"},
            expected_safe_documented,
        )
        self.assertTrue(all(row["execution_context"] == "BOTH" for row in documented_rows if row["safe_to_execute"] == "YES"))
        self.assertTrue(all(row["expected_return_code"] == "NOT_EXECUTED" for row in documented_rows if row["safe_to_execute"] == "NO"))
        self.assertEqual(
            {row["execution_context"] for row in documented_rows if row["safe_to_execute"] == "NO"},
            {
                "BUILD_ORCHESTRATION",
                "DEPENDENCY_PROFILE",
                "DOCUMENTATION_REGRESSION",
                "EXTERNAL_ASSET_DEPENDENCY",
                "INSTALL_ORCHESTRATION",
                "NETWORKED_ASSET_ACQUISITION",
                "PUBLICATION_PREFLIGHT",
                "SCIENTIFIC_EXECUTION",
                "SCIENTIFIC_PREFLIGHT_DEPENDENCY",
                "SHELL_STATE_CHANGE",
                "STATEFUL_PREFLIGHT",
                "SYNTHETIC_QUICK_EXECUTION",
                "SYNTAX_TEMPLATE",
                "USER_DATA_MUTATION",
                "WORKFLOW_PREFLIGHT",
            },
        )

        from compag_curation.config import VARIANT_FIELDS

        legacy_rows = [row for row in matrix if row["command_id"].startswith("CMD-V") and row["command_id"] != "CMD-VERSION-001"]
        expected_legacy = {
            f"compag-curation {command} --config {{CONFIG}} --variant {variant} {mode}"
            for command, variant in VARIANT_FIELDS
            for mode in ("--validate", "--plan")
        }
        self.assertEqual(len(legacy_rows), 42)
        self.assertEqual({row["command"] for row in legacy_rows}, expected_legacy)

        top_commands = (
            "propose", "extract-features", "review", "train", "infer", "evaluate", "transfer", "report",
            "doctor", "init-project", "inspect-data", "validate", "plan", "assets", "demo", "run",
            "convert-annotations", "verify-artifacts", "migrate-feature-csv", "active-learning",
            "review-ui",
        )
        expected_help = {
            "compag-curation --help",
            *(f"compag-curation {command} --help" for command in top_commands),
            "compag-curation assets fetch --help",
            "compag-curation assets verify --help",
            "compag-curation active-learning initial-export --help",
            "compag-curation active-learning initial-resume --help",
            "compag-curation active-learning begin-round --help",
            "compag-curation active-learning resume-round --help",
            "compag-curation active-learning begin-image-round --help",
            "compag-curation active-learning resume-image-round --help",
            "compag-curation active-learning import-r92-transfer-baseline --help",
            "compag-curation active-learning infer-r92-image --help",
            "compag-curation active-learning begin-transfer-image-round --help",
            "compag-curation active-learning resume-transfer-image-round --help",
            "compag-curation review-ui web --help",
            "compag-curation review-ui desktop --help",
            "compag-curation review-ui status --help",
        }
        help_rows = [row for row in matrix if row["command_id"].startswith("CMD-HELP-")]
        self.assertEqual(len(help_rows), 37)
        self.assertEqual({row["command"] for row in help_rows}, expected_help)

        self.assertEqual(
            {row["command"] for row in matrix if row["command_id"].startswith("CMD-MUTATION-")},
            {
                "compag-curation migrate-feature-csv --path {FEATURE_CSV} --column scale",
                "compag-curation migrate-feature-csv --path {FEATURE_CSV} --column review_tag",
            },
        )
        self.assertEqual(
            {row["command"] for row in matrix if row["command_id"] == "CMD-MANIFEST-001"},
            {"compag-curation verify-artifacts --root . --manifest manifests/public_file_inventory.csv"},
        )
        safe = [row for row in matrix if row["safe_to_execute"] == "YES"]
        self.assertTrue(safe)
        self.assertTrue(all(row["execution_context"] == "BOTH" for row in safe))

        runner_source = (self.root / "tools/run_documented_commands.py").read_text(encoding="utf-8")
        for marker in (
            "subject_root_identity",
            "public_manifest_sha256",
            '"wheel"',
            '"interpreter"',
            '"console"',
            '"execution_root_identity_before"',
            '"execution_root_identity_after"',
            '"execution_tree_matches_subject"',
            '"package_origin"',
            '"source_binding"',
            '"requested_path"',
            '"resolved_target"',
            "SAFE_DOCUMENTED_COMMANDS",
            "regenerated_matrix_rows",
        ):
            self.assertIn(marker, runner_source)

    def test_reviewed_new_documented_command_classification_is_exact(self) -> None:
        runner_path = self.root / "tools/run_documented_commands.py"
        spec = importlib.util.spec_from_file_location(
            "compag_reviewed_new_documented_command_contract",
            runner_path,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)

        reviewed = runner.REVIEWED_V140_NEW_COMMAND_CONTEXTS
        self.assertEqual(len(reviewed), 28)
        self.assertEqual(
            {context: list(reviewed.values()).count(context) for context in set(reviewed.values())},
            {
                "BUILD_ORCHESTRATION": 8,
                "INSTALL_ORCHESTRATION": 1,
                "PUBLICATION_PREFLIGHT": 8,
                "SHELL_STATE_CHANGE": 5,
                "STATEFUL_PREFLIGHT": 6,
            },
        )
        for command, context in reviewed.items():
            with self.subTest(command=command):
                metadata = runner.reviewed_new_documented_metadata(command)
                self.assertEqual(metadata["execution_context"], context)
                self.assertEqual(metadata["safe_to_execute"], "NO")
                self.assertEqual(metadata["expected_return_code"], "NOT_EXECUTED")
                self.assertEqual(metadata["expected_output"], "NOT_EXECUTED")
                with self.assertRaisesRegex(RuntimeError, "lacks reviewed metadata"):
                    runner.reviewed_new_documented_metadata(
                        f"{command} --unreviewed-near-miss"
                    )

        exact_non_v140_groups = (
            runner.REVIEWED_GPU_INSTALL_ORCHESTRATION_COMMANDS,
            runner.REVIEWED_GPU_PROFILE_INSTALLER_COMMANDS,
            runner.REVIEWED_GPU_PREFLIGHT_COMMANDS,
            runner.REVIEWED_O_EXCL_COPY_COMMANDS,
            runner.REVIEWED_REVIEW_UI_COMMANDS,
        )
        for command in set().union(*exact_non_v140_groups):
            with self.subTest(non_v140_exact=command):
                runner.reviewed_new_documented_metadata(command)
                with self.assertRaisesRegex(RuntimeError, "lacks reviewed metadata"):
                    runner.reviewed_new_documented_metadata(
                        f"{command} --unreviewed-near-miss"
                    )
        with self.assertRaisesRegex(RuntimeError, "lacks reviewed metadata"):
            runner.reviewed_new_documented_metadata(
                "python tools/normalize_sdist.py --input arbitrary.tar.gz --output arbitrary-normalized.tar.gz"
            )

    def test_documented_command_fixture_registry_covers_every_required_field(self) -> None:
        runner_path = self.root / "tools/run_documented_commands.py"
        spec = importlib.util.spec_from_file_location(
            "compag_documented_command_fixture_registry", runner_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)

        from compag_curation.config import VARIANT_FIELDS, validate_mapping
        from compag_curation.plan import build_plan

        for (command, variant), (required, _optional) in VARIANT_FIELDS.items():
            with self.subTest(command=command, variant=variant):
                workflow = runner._workflow(command, variant)
                self.assertEqual(set(workflow), required)
                self.assertEqual(workflow["variant"], variant)

                fixture = runner._fixture(command, variant)
                self.assertEqual(
                    fixture["fixture_classification"],
                    "SYNTHETIC_NON_SCIENTIFIC_TEST_FIXTURE",
                )
                self.assertEqual(
                    fixture["execution"],
                    {
                        "authorized": False,
                        "allow_download": False,
                        "output_collision": "fail",
                    },
                )
                config = validate_mapping(fixture, fixture=True, environment={})
                self.assertFalse(
                    build_plan(command, config, variant).to_dict()["scientific_execution"]
                )

    def test_documented_command_parser_accepts_only_exact_fail_fast_prologue(self) -> None:
        runner_path = self.root / "tools/run_documented_commands.py"
        spec = importlib.util.spec_from_file_location(
            "compag_documented_command_prologue_contract", runner_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)

        documentation_metadata = runner.reviewed_new_documented_metadata(
            runner.DOCUMENTATION_SHELL_REGRESSION_COMMAND
        )
        self.assertEqual(
            documentation_metadata["execution_context"], "DOCUMENTATION_REGRESSION"
        )
        self.assertEqual(documentation_metadata["safe_to_execute"], "NO")
        self.assertEqual(documentation_metadata["expected_return_code"], "NOT_EXECUTED")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = root / "README.md"
            markdown.write_text(
                "# Fixture\n```bash\n# Fail closed\nset -euo pipefail\n"
                "python -m example --help\n```\n",
                encoding="utf-8",
            )
            self.assertEqual(
                runner.documented_commands(root),
                {"python -m example --help": ("README.md:5",)},
            )

            rejected_bodies = (
                "set -eu\npython -m example --help",
                "set -o pipefail\npython -m example --help",
                "set -euo pipefail && true\npython -m example --help",
                "set -euo \\\npipefail\npython -m example --help",
                "python -m example --help\nset -euo pipefail",
                "set -euo pipefail\nset -euo pipefail\npython -m example --help",
            )
            for body in rejected_bodies:
                with self.subTest(body=body):
                    markdown.write_text(f"```bash\n{body}\n```\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        RuntimeError, "unsupported Bash set command"
                    ):
                        runner.documented_commands(root)

    def test_documented_command_runner_preserves_requested_python_and_binds_clean_source(self) -> None:
        runner_path = self.root / "tools/run_documented_commands.py"
        spec = importlib.util.spec_from_file_location("compag_documented_command_runner_test", runner_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            executable = temporary / "python-target"
            executable.write_bytes(Path(sys.executable).read_bytes())
            executable.chmod(0o755)
            requested = temporary / "python-requested"
            requested.symlink_to(executable.name)
            selected, target, record = module._requested_executable(requested, "python")
            self.assertEqual(selected, requested.absolute())
            self.assertEqual(target, executable.resolve())
            self.assertTrue(record["requested_is_symlink"])
            self.assertEqual(record["requested_symlink_target"], executable.name)

            source_root = temporary / "source"
            package = source_root / "src/compag_curation"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("SOURCE_BINDING_PROBE = True\n", encoding="utf-8")
            expected_module = (package / "__init__.py").resolve()
            probe_stdout = json.dumps(
                {
                    "module": str(expected_module),
                    "prefix": str(temporary / "venv"),
                    "base_prefix": str(temporary / "base"),
                    "executable": str(requested),
                },
                sort_keys=True,
            ).encode("utf-8")
            completed = subprocess.CompletedProcess(
                [str(requested), "-B", "-c", "<ORIGIN_PROBE>"],
                0,
                stdout=probe_stdout,
                stderr=b"",
            )
            with mock.patch.object(module, "_run", return_value=completed) as runner:
                origin = module._package_origin(Path(sys.executable), source_root, source_root=source_root)
            self.assertEqual(runner.call_args.kwargs["source_root"], source_root)
            self.assertEqual(Path(origin["module"]).resolve(), (package / "__init__.py").resolve())

            documented = ["compag-curation", "coco-features", "--config", "{CONFIG}", "--variant", "coco-features"]
            self.assertEqual(module._documented_subcommand(documented), "coco-features")
            rewritten = [str(requested), "-B", "-m", "compag_curation", *documented[1:]]
            self.assertEqual(module._documented_subcommand(documented), rewritten[4])
            with self.assertRaisesRegex(RuntimeError, "documented compag-curation subcommand"):
                module._documented_subcommand(rewritten)

            installed = module.rewrite_executable(
                [".venv-base/bin/compag-curation", "--version"],
                context="INSTALLED_WHEEL",
                python=requested,
                console=executable,
            )
            self.assertEqual(installed, [str(executable), "--version"])
            source = module.rewrite_executable(
                ["compag-curation", "doctor", "--profile", "base"],
                context="CLEAN_EXTRACTED_REPOSITORY",
                python=requested,
                console=executable,
            )
            self.assertEqual(
                source,
                [str(requested), "-B", "-m", "compag_curation", "doctor", "--profile", "base"],
            )
            python_command = module.rewrite_executable(
                ["python3.12", "-I", "-S", "-B", "tools/public_qa.py"],
                context="INSTALLED_WHEEL",
                python=requested,
                console=executable,
            )
            self.assertEqual(python_command[0], str(requested))
            self.assertEqual(
                module.rewrite_executable(
                    ["python3.12", "--version"],
                    context="INSTALLED_WHEEL",
                    python=requested,
                    console=executable,
                ),
                [str(requested), "--version"],
            )
            workspace = temporary / "workspace"
            self.assertEqual(
                module.execution_cwd(
                    {
                        "command_id": "CMD-DOC-999",
                        "command": "python -I -S -B tools/public_qa.py --root .",
                    },
                    source_root,
                    workspace,
                ),
                source_root,
            )
            self.assertEqual(
                module.execution_cwd(
                    {"command_id": "CMD-DOC-004", "command": "python3.12 --version"},
                    source_root,
                    workspace,
                ),
                workspace,
            )
            with self.assertRaisesRegex(RuntimeError, "does not support executable"):
                module.rewrite_executable(
                    ["pip", "install", "value"],
                    context="INSTALLED_WHEEL",
                    python=requested,
                    console=executable,
                )


if __name__ == "__main__":
    unittest.main()
