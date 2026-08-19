# KernelBench Reward-Hacking Audit

This directory preserves the cross-run reward-hacking audit completed on
2026-08-07. The audit reviewed every path-distinct completed KernelBench result
available at the cutoff: 120 exported result directories across four run
directories. At the cutoff, a second wave in the fourth run was still active
and was excluded. The audit identified 28 paths requiring action, including
semantic or replay defects, incomplete artifacts, and one network-dependent
export.

## Reports

- [Markdown report](kernelbench-all-report-reward-hacking.md)
- [Self-contained HTML report](kernelbench-all-report-reward-hacking.html)

The HTML is rendered from the Markdown and contains its styling inline. Both
files are checked in so the historical result does not depend on the original
reporting machine or a particular renderer version.

## Raw evidence and provenance

The report was generated from a Breval archive rooted at `BREVAL_RUNS`. Full
run exports and native session logs are not duplicated here because they are
several gigabytes. Evidence citations are archive-relative paths with line
numbers, so they remain usable wherever that raw archive is restored.

The analysis followed the `autocuda:report-reward-hacking` workflow. That
workflow has no lossy data-build step: the author reviewed result metadata,
submitted code, sessions, grader inputs, and retained artifacts directly, then
wrote the report. The checked-in snapshot was validated with autocuda 0.5.0
and rendered with Pandoc 2.9.2.1. `MANIFEST.json` records the original generated
hashes, publication normalization, tool versions, and corpus counts;
`SHA256SUMS` records the exact published contents.

The four contributing run-directory suffixes and their completed exports at
the cutoff were:

| Run suffix | Completed exports |
| --- | ---: |
| `20260804-000405` | 36 |
| `20260804-214205` | 36 |
| `20260806-132917` | 36 |
| `20260807-013130` | 12 |

Treat the Markdown as the reviewed audit source. Revise it only when the
underlying evidence has been reviewed again, and always regenerate the HTML
rather than editing the generated file directly.

## Verify

With the reusable reporting tooling and `autocuda` CLI available:

```bash
scripts/reward_hacking_report.sh verify \
  analysis/reward-hacking/kernelbench-all-report-reward-hacking.md
```

This checks the Markdown against the built-in reward-hacking report schema and
verifies both generated files and the manifest against their recorded SHA-256
digests.

## Render the HTML

With `autocuda` and Pandoc installed:

```bash
scripts/reward_hacking_report.sh render \
  analysis/reward-hacking/kernelbench-all-report-reward-hacking.md
```

The script validates the Markdown before replacing the adjacent HTML file. A
new audit requires restoring the raw Breval corpus and repeating the direct
evidence review; rendering alone does not recalculate any findings.
