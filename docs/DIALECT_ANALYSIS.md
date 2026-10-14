# CUDA dialect analysis

The CUDA-dialect performance studies and their reporting tools are maintained
in [robobryce/cuda-dialect-agent-analysis](https://github.com/robobryce/cuda-dialect-agent-analysis)
(private repository; access is required).

That repository keeps GPU MODE and KernelBench-Hard as peer studies, with
shared reporting infrastructure. KernelBench's study includes the four
performance/token-efficiency chart generators, the reviewed combined dataset,
historical charts, regrade controller, and the August 7 reward-hacking audit.
The audit wrapper, workflow documentation, and tests live there too.

The move is recorded in
[consolidation PR #1](https://github.com/robobryce/cuda-dialect-agent-analysis/pull/1).
Merge that PR before removing the old reporting code here. Its
`docs/CONSOLIDATION.md` records the source commits and preservation checks.

In the analysis repository, run:

```bash
./scripts/verify_reports.sh kernelbench-hard
./scripts/build_reports.sh kernelbench-hard
```

The old `analysis/` chart scripts and `scripts/reward_hacking_report.sh` are no
longer maintained here. The historical
`analysis/reward-hacking-report-2026-08-07` branch and Git history remain intact
as provenance; use the consolidated repository for future analysis changes.

This move does not change KernelBench's website, benchmark definitions,
grading, leaderboard, or operational `kb audit`/`kb lint` commands.
