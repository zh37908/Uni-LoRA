#!/bin/bash
# Collect Slurm accounting stats (elapsed time, state, allocated resources) for
# all new-experiment jobs (p0_* math/commonsense, p1_* vision, p2_* eval, e1_/e2_ theory).
# Usage: bash collect_job_stats.sh [start_date, default 2026-08-27] > job_stats.txt

START_DATE="${1:-2026-08-27}"

sacct \
  --starttime "$START_DATE" \
  --user "$USER" \
  --format="JobID%18,JobName%42,Partition%12,AllocTRES%42,Submit,Start,Elapsed,State%12" \
  | awk 'NR<=2 || $2 ~ /^(p0_|p1_|p2_|e1_|e2_)/'
