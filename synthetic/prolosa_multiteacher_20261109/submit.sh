#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_job="$here/gpu_job.sh"
analysis_job="$here/analysis_job.sh"
logs="$here/logs"
record="$here/job_ids.tsv"
pending="$here/submission_pending"
release_done="$here/smoke_released"
shards=5
parallel=5

# A whole-filesystem sync on shared NFS can report another writer's ENOSPC.
# Flush only the ledger file whose durability this submission needs.
fsync_path() {
    python -c 'import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.fsync(fd); os.close(fd)' "$1"
}

mkdir -p "$logs"
[[ -f "$gpu_job" && -f "$analysis_job" ]] || {
    echo "Missing GPU or analysis job script" >&2; exit 1;
}

# Do not let concurrent invocations submit duplicate stages.
exec 9>"$here/submit.lock"
flock -n 9 || { echo "Another submit.sh invocation is active" >&2; exit 1; }

declare -A ids=()
expected_stages=(smoke main analysis)
stage_index=0
if [[ -f "$record" ]]; then
    while IFS=$'\t' read -r timestamp stage id dependency extra || [[ -n "${timestamp:-}${stage:-}${id:-}${dependency:-}${extra:-}" ]]; do
        [[ -n "$timestamp" && "$id" =~ ^[0-9]+$ && -z "${extra:-}" ]] || {
            echo "Malformed job record in $record" >&2; exit 1;
        }
        [[ "$stage_index" -lt "${#expected_stages[@]}" && "$stage" == "${expected_stages[$stage_index]}" ]] || {
            echo "Unexpected or duplicate stage in $record: $stage" >&2; exit 1;
        }
        expected_dependency=-
        case "$stage" in
            main) expected_dependency="afterok:${ids[smoke]}" ;;
            analysis) expected_dependency="afterok:${ids[main]}" ;;
        esac
        [[ "$dependency" == "$expected_dependency" ]] || {
            echo "Invalid dependency for $stage in $record" >&2; exit 1;
        }
        ids[$stage]=$id
        ((stage_index += 1))
    done < "$record"
fi

if [[ -e "$pending" ]]; then
    pending_stage=$(cat "$pending")
    [[ "$pending_stage" == smoke || "$pending_stage" == main || "$pending_stage" == analysis ]] || {
        echo "Invalid submission intent in $pending" >&2; exit 1;
    }
    if [[ -n "${ids[$pending_stage]:-}" ]]; then
        # ID was recorded, but the process stopped before clearing its intent.
        rm -- "$pending"
    else
        echo "Submission of '$pending_stage' may have reached Slurm without a recorded ID." >&2
        echo "Inspect squeue/sacct and reconcile $pending before rerunning; no jobs were submitted now." >&2
        exit 1
    fi
fi

submit_stage() {
    local stage=$1 dependency=$2 raw id
    [[ -z "${ids[$stage]:-}" ]] || return 0

    # Persist intent before contacting Slurm; an ambiguous failure blocks
    # automatic retry until the possible submission has been reconciled.
    printf '%s\n' "$stage" > "$pending"
    fsync_path "$pending"
    case "$stage" in
        smoke)
            raw=$(sbatch --parsable --hold \
                --job-name=prolosa-mt-smoke --time=01:00:00 \
                --output="$logs/smoke_%j.out" --error="$logs/smoke_%j.err" \
                "$gpu_job" smoke)
            ;;
        main)
            raw=$(sbatch --parsable \
                --job-name=prolosa-mt-main --time=04:00:00 \
                --array="0-$((shards-1))%$parallel" \
                --dependency="$dependency" \
                --output="$logs/main_%A_%a.out" --error="$logs/main_%A_%a.err" \
                "$gpu_job" run "$shards")
            ;;
        analysis)
            raw=$(sbatch --parsable \
                --job-name=prolosa-mt-analysis --dependency="$dependency" \
                --output="$logs/analysis_%j.out" --error="$logs/analysis_%j.err" \
                "$analysis_job")
            ;;
        *) echo "Unknown stage: $stage" >&2; exit 2 ;;
    esac
    id=${raw%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || {
        echo "Unexpected $stage sbatch response: $raw" >&2; exit 1;
    }
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$stage" "$id" "$dependency" >> "$record"
    fsync_path "$record"
    ids[$stage]=$id
    rm -- "$pending"
}

submit_stage smoke -
submit_stage main "afterok:${ids[smoke]}"
submit_stage analysis "afterok:${ids[main]}"

# Release smoke only after the full dependency chain is recorded. A restart
# after release but before the marker verifies Slurm's current state.
if [[ -f "$release_done" ]]; then
    IFS=$'\t' read -r released_at released_id extra < "$release_done"
    [[ -n "$released_at" && "$released_id" == "${ids[smoke]}" && -z "${extra:-}" ]] || {
        echo "Release marker does not match smoke job ${ids[smoke]}" >&2; exit 1;
    }
fi
if [[ ! -f "$release_done" ]]; then
    if ! scontrol release "${ids[smoke]}"; then
        queue=$(squeue -h -j "${ids[smoke]}" -o '%T|%r')
        if [[ -n "$queue" ]]; then
            [[ "$queue" != *JobHeld* ]] || {
                echo "Smoke job ${ids[smoke]} is still held" >&2; exit 1;
            }
        else
            state=$(sacct -n -X -j "${ids[smoke]}" -o State -P | head -n 1)
            [[ -n "$state" && "$state" != PENDING* ]] || {
                echo "Cannot verify release of smoke job ${ids[smoke]}" >&2; exit 1;
            }
        fi
    fi
    printf '%s\t%s\n' "$(date -u +%FT%TZ)" "${ids[smoke]}" > "$release_done"
    fsync_path "$release_done"
fi

printf 'Smoke job: %s\nMain array: %s (afterok:%s; 0-%d%%%d)\nAnalysis job: %s (afterok:%s)\nJob IDs: %s\n' \
    "${ids[smoke]}" "${ids[main]}" "${ids[smoke]}" "$((shards-1))" "$parallel" \
    "${ids[analysis]}" "${ids[main]}" "$record"
