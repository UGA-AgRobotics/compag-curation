# Contributing

Contributions should keep the package notebook-independent, configuration
driven, and safe to inspect without scientific assets.

## Development Environment

Use CPython 3.12. Install the reviewed bootstrap and development archive locks
before installing the project editable; do not ask pip to resolve the `dev`
extra independently:

```bash
set -euo pipefail
umask 0022
test ! -e .venv
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes --no-deps -r requirements/constraints-bootstrap-cp312-linux-x86_64.txt
python -m pip install --require-hashes --no-deps -r requirements/constraints-dev-cp312-linux-x86_64.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
```

Do not add a dependency when a standard-library or existing project facility
is sufficient. Never vendor a third-party repository, wheel, model, weight,
checkpoint, dataset, or private evidence file.

## Required QA for this candidate

Run the bounded JR09 public-safe acceptance from the repository root with
CPython 3.12 and new output paths:

```bash
set -euo pipefail
python3.12 -I -B tools/public_safe_acceptance.py \
  --root . \
  --work-dir /absolute/absent/compag-public-acceptance-work \
  --receipt /absolute/existing-parent/compag-public-acceptance.json
```

The selected base suite uses mocks and small synthetic temporary fixtures and
must not train, infer, score, load a model, initialize CUDA, use a network, or
discover local project data. Every active test must execute with zero skips.
The exact active and excluded inventory is `PUBLIC_TEST_INVENTORY.json`;
excluded tests receive no pass credit. Runtime/scientific changes are outside
this candidate and require a separate authorized process.

## Pull Requests

Keep changes scoped and explain:

- the user-visible behavior;
- the affected workflow and configuration fields;
- tests added or changed;
- optional dependency implications;
- documentation and license-scope effects.

Do not alter sealed scientific values or claim that a source change reproduced
study results. Changes that require real data, trained models, scientific
execution, or a new metric belong in a separately authorized scientific
process, not an ordinary software pull request.

Use relative paths in examples. Do not submit credentials, personal
information, raw images, annotations, predictions, scores, review logs, model
files, private absolute paths, or notebook files.

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
Contributions are accepted under the path-specific terms described in
[LICENSES.md](LICENSES.md).
