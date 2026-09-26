# Data And Artifact Contract

## Typed Artifacts

Workflow inputs and outputs are represented by immutable `ArtifactRef`
records. Each record has a semantic name, role, path, file/directory kind,
requiredness, and optional SHA-256 identity. The selected registry variant
defines the contract; caller-provided metadata cannot weaken it.

A plan contains only artifact descriptions. Plan construction performs no
artifact read, hash, model load, directory traversal, or output creation.

## Controlled Preflight

A separately controlled Python integration validates artifacts before a
handler can use them. Depending on the variant, preflight requires:

- required files to be regular and non-symlink;
- required directories to be real directories with no symlink members;
- sealed file hashes and directory-tree identities to match configuration;
- input and output locators to be distinct and safely separated;
- future output paths to be absent;
- local model/config assets to be explicit;
- implicit downloads to remain disabled.

Directory identities are deterministic over relative paths, modes, sizes, and
file hashes. Input identities are rechecked around sensitive reads and output
installation.

## Output And Collision Policy

Normal workflow outputs use `output_collision: "fail"`. An existing output is
an error; it is not silently overwritten, resumed, or renamed. Output files are
written to a same-directory temporary file and installed without clobbering an
existing destination. Output directories are claimed explicitly.

Workflow plans use named artifacts. A step cannot consume an unavailable
artifact, overwrite an input, or produce an undeclared terminal output.

Some workflow implementations allocate deterministic names inside a declared
output root. The declared root remains the ownership boundary, and name
collisions fail rather than selecting an ambient backend default.

## Legacy Feature CSV Exception

`migrate-feature-csv` is the one documented in-place maintenance operation.
It implements `PRESERVE_BACKUPS` for two historical schema additions:

| Column | Placement | Backup suffix |
|---|---|---|
| `scale` | after `ann_id`, otherwise at zero-based index 2 | `.pre_scale.bak.csv` |
| `review_tag` | after `reviewed`, otherwise before `class_id` or at the end | `.pre_reviewtag.bak.csv` |

New cells in the added column are empty. Before replacing the CSV, the command
links the exact immediate source bytes to the fixed same-directory backup.
Successful installation retains that backup. It is never deleted or
overwritten.

Safety rules:

- the source must be a regular non-symlink file;
- source identity is bound before and during the read;
- the backup must not already exist;
- the destination and backup share a real parent directory;
- a column already present produces no change and no backup;
- interruption recovery does not intentionally discard the original bytes.

For `workspace/features.csv`, the two possible backup names are
`workspace/features.pre_scale.bak.csv` and
`workspace/features.pre_reviewtag.bak.csv`.

## User Assets

Raw project data, row-level records, predictions, scores, models, weights,
checkpoints, and fitted transforms are not package artifacts. Users provide
authorized local copies and bind them through configuration. The repository
does not grant rights to or publish acquisition links for those assets.

See [DATA_MODEL_AVAILABILITY.md](DATA_MODEL_AVAILABILITY.md).

## Package Manifests

The package verifier accepts a root and a self-excluding CSV manifest:

```bash
compag-curation verify-artifacts --root ROOT --manifest CSV
```

The verifier requires exact regular-file closure. Public release construction
also checks safe relative archive paths, unique normalized names, deterministic
modes and timestamps, and byte-identical independent builds.
