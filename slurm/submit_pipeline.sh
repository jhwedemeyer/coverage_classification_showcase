#!/bin/bash
# Submit the whole coverage classification pipeline as one chain of
# dependent slurm jobs.
#
#   1 classify_impact_type    ->  2 classify_locations (GPU, array over types)
#   2                         ->  3 extract_locations  ->  5 map_location_to_country
#   2                         ->  4 rank_locations (array over types)
#   4 + 5                     ->  6 gen_coverage_days
#
# 4 and 5 are independent of each other and run side by side; every other edge
# is a real data dependency. Each stage starts only if the one it needs
# succeeded (afterok), so a failure stops the rest of the chain instead of
# feeding it half-written files.
#
# Usage
#   ./slurm/submit_pipeline.sh                 # everything
#   ./slurm/submit_pipeline.sh --from 4        # stage 4 onwards, no dependencies
#   ./slurm/submit_pipeline.sh --only 2        # a single stage
#   ./slurm/submit_pipeline.sh --dry-run       # print the sbatch calls only
#
# Knobs (environment variables)
#   ACCOUNT, QOS          slurm account and qos            (no default, must be set)
#   GPU_QOS               qos of stage 2                   (gpushort)
#   GPU_PARTITION         partition of stage 2             (gpu)
#   GPU_GRES              gres of stage 2                  (gpu:1)
#   PIPELINE_ENV          conda environment of all stages  (coverage)
#   PIPELINE_ENV_ROOT     where that environment lives     (/p/tmp/$USER/envs)
#   CONDA_ENV             environment of the CPU stages    (= PIPELINE_ENV)
#   CONDA_ENV_GPU         environment of stage 2           (= PIPELINE_ENV)
#   GEOCODE_CHAIN         geocoding jobs run back to back  (2)
#   SKIP_GEOCODE=1        leave stage 5 out of the chain
#   DTYPES_ARRAY          array range of stages 2 and 4    (0-4)

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COVERAGE_DIR="$(cd "$SLURM_DIR/.." && pwd)"
export COVERAGE_DIR

ACCOUNT="${ACCOUNT:-}"      # slurm account to bill -- set to your project's account
QOS="${QOS:-}"              # slurm qos -- set to one available on your cluster
GPU_QOS="${GPU_QOS:-gpushort}"
GPU_PARTITION="${GPU_PARTITION:-gpu}"
GPU_GRES="${GPU_GRES:-gpu:1}"
GEOCODE_CHAIN="${GEOCODE_CHAIN:-2}"
SKIP_GEOCODE="${SKIP_GEOCODE:-0}"
DTYPES_ARRAY="${DTYPES_ARRAY:-0-4}"

FROM=1
ONLY=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --only) ONLY="$2"; FROM="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$ACCOUNT" || -z "$QOS" ]]; then
    echo "ACCOUNT and QOS must be set, e.g. ACCOUNT=myproject QOS=normal ./slurm/submit_pipeline.sh" >&2
    exit 1
fi

# The job scripts write their logs to slurm/logs and address their data relative
# to the repository root, which is where slurm starts them from.
mkdir -p "$SLURM_DIR/logs"
cd "$COVERAGE_DIR"

# Submits one stage and returns its job id. Every argument before the script is
# passed to sbatch, which is how the qos, the dependencies and the GPU request
# get in.
submit() {
    local script="$1"; shift
    local args=(--account="$ACCOUNT" --export=ALL,COVERAGE_DIR "$@" "slurm/$script")
    if (( DRY_RUN )); then
        echo "sbatch ${args[*]}" >&2
        echo "DRYRUN_$RANDOM"
        return
    fi
    local id
    id="$(sbatch --parsable "${args[@]}")"
    # --parsable prints "jobid" or, with a cluster, "jobid;cluster"
    id="${id%%;*}"
    printf '%-32s %s\n' "$script" "$id" >&2
    echo "$id"
}

# A dependency is only expressed for a stage whose predecessor this run
# submitted itself; with --from/--only the earlier output is taken as given.
started() { local stage="$1"; [[ -z "$ONLY" || "$ONLY" == "$stage" ]] && (( stage >= FROM )); }
dep() { local id="${1:-}"; [[ -n "$id" ]] && echo "--dependency=afterok:$id" || echo ""; }

id1="" id2="" id3="" id4="" id5="" id6=""

if started 1; then
    id1="$(submit 01_classify_impact_type.sb --qos="$QOS")"
fi

if started 2; then
    id2="$(submit 02_classify_locations.sb \
        --qos="$GPU_QOS" --array="$DTYPES_ARRAY" \
        --partition="$GPU_PARTITION" --gres="$GPU_GRES" \
        $(dep "$id1"))"
fi

if started 3; then
    id3="$(submit 03_extract_locations.sb --qos="$QOS" $(dep "$id2"))"
fi

if started 4; then
    id4="$(submit 04_rank_locations.sb --qos="$QOS" --array="$DTYPES_ARRAY" $(dep "$id2"))"
fi

if started 5 && [[ "$SKIP_GEOCODE" != "1" ]]; then
    # Nominatim answers one request per second, so the ~110k locations do not
    # fit into a single wall time. The script picks up where the previous run
    # left off, so the follow-ups depend on the predecessor having ended
    # (afterany), not on it having finished the list (afterok).
    prev="$(dep "$id3")"
    for (( i = 0; i < GEOCODE_CHAIN; i++ )); do
        id5="$(submit 05_map_location_to_country.sb --qos="$QOS" $prev)"
        prev="--dependency=afterany:$id5"
    done
fi

if started 6; then
    deps=()
    [[ -n "$id4" ]] && deps+=("$id4")
    [[ -n "$id5" ]] && deps+=("$id5")
    dependency=""
    if (( ${#deps[@]} )); then
        dependency="--dependency=afterok:$(IFS=:; echo "${deps[*]}")"
    fi
    id6="$(submit 06_gen_coverage_days.sb --qos="$QOS" $dependency)"
fi

echo >&2
echo "logs: $SLURM_DIR/logs, queue: squeue -u \$USER -o '%.10i %.20j %.8T %.10M %R'" >&2
