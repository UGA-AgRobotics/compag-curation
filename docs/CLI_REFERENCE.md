# CLI reference - accepted surface of the JR09 model-free candidate

The installed console name remains `compag-curation`, application version
1.9.3. This source package has the distinct packaging revision recorded in
`PUBLIC_PACKAGING_REVISION.json`.

The current accepted commands are:

```text
compag-curation --help
compag-curation --version
compag-curation doctor --profile base
```

The bounded acceptance also creates a temporary synthetic project and runs an
inert `plan`; the returned record must state `scientific_execution=false`.
Exact command help for retained interfaces is checked without dispatching a
scientific handler.

Other subcommands remain visible as application code evidence but are not
accepted execution routes in this package. In particular, GPU/science doctor,
asset download, model-assisted review, training, inference, resume, and
active-learning require dependencies or controlled assets that are not
included. They must fail closed and must not be described as runnable public
features here.

Run the complete supported route with the single command in
[Development](DEVELOPMENT.md); see `PUBLIC_TEST_INVENTORY.json` for its exact
test selection.
