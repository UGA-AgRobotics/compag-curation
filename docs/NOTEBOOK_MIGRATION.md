# Notebook Migration

## Inventory

The authoritative static inventory reconciles:

- 22 notebooks;
- 236 code-cell occurrences;
- 84 operational cells;
- 152 non-operational cells.

Notebooks were treated as static source authority during migration. They are
not included in the public source tree and are not opened, parsed, imported, or
executed by production runtime.

## Operational Traceability

Every operational cell is bound to:

- a current implemented Python symbol or real non-notebook manual entry point;
- a workflow capability and typed input/output contract;
- a CLI/configuration route where applicable;
- a public test node;
- evidence supporting its disposition.

Many-to-one and one-to-many mappings are valid. Duplicate or superseded cells
still resolve to the implemented symbol that owns their behavior. Manual review
cells resolve to explicit review launcher interfaces rather than simulated
automation.

Current acceptance is based on capability, callability, configuration, and
test traceability. Historical internal diagnostic gates are not runtime
components and are excluded from this public release.

## Runtime Result

The package carries no generated `cell_nb_live_*` dispatch layer. State moves
through configuration, dataclasses, named artifacts, and injected services.
Production paths do not depend on interactive globals, cell order, shell
magics, Jupyter, IPython, or notebook conversion libraries.

The public CLI supports configuration validation and inert planning for all
workflow variants. Human-review workflows have documented non-notebook
handoffs. The two observable legacy feature-CSV migrations are exposed through
`migrate-feature-csv` with `PRESERVE_BACKUPS`.

## Claim Boundary

Migration completeness is a software traceability statement. It is not a claim
that scientific or behavioral parity was rerun:

```text
SCIENTIFIC_PARITY_STATUS=NOT_RETESTED_BY_DESIGN
BEHAVIORAL_PARITY_STATUS=NOT_RETESTED_BY_DESIGN
```
