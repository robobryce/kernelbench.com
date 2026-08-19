# Reward-hacking reports

Reward-hacking audits have two different kinds of output:

- reusable validation and rendering tooling, which belongs on `main`; and
- generated Markdown, HTML, manifests, and checksums, which belong on a
  dedicated report branch.

Keeping generated snapshots off `main` preserves their history without making
the application repository carry every report revision. A report branch should
start from the relevant `main` revision so it also contains the reusable
tooling used to validate the snapshot.

## Audit the evidence

Follow the `autocuda:report-reward-hacking` workflow. There is no automated
data-build step for this report: read the optimization logs, source changes,
validation surface, and retained artifacts directly. The report must use this
name:

```text
<output>-report-reward-hacking.md
```

Keep machine-specific evidence roots out of published links. Prefer stable
public URLs; when the raw corpus is not published, use a documented symbolic
root such as `BREVAL_RUNS/path/to/file.py:123`.

## Render

The repository wrapper validates the Markdown schema before asking autocuda to
render the adjacent self-contained HTML file:

```bash
scripts/reward_hacking_report.sh render \
  path/to/<output>-report-reward-hacking.md
```

This requires the `autocuda` CLI and Pandoc. It does not recalculate or revise
the audit findings.

## Package and verify

A durable report branch should contain:

- the reviewed Markdown source;
- the generated self-contained HTML;
- a manifest recording the audit cutoff, corpus counts, tool versions, and
  source provenance;
- `SHA256SUMS` covering the generated files and manifest; and
- a short README explaining how to locate any external raw evidence.

Verify the schema, HTML presence, and recorded hashes before committing:

```bash
scripts/reward_hacking_report.sh verify \
  path/to/<output>-report-reward-hacking.md
```

The verifier reads `SHA256SUMS` beside the Markdown by default. Pass a checksum
file as a second path when the package uses a different layout.
