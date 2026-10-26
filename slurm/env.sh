#!/bin/bash
# Shared environment for every stage of the coverage classification pipeline.
# Sourced by the 0*.sb job scripts, never submitted on its own.
#
# Usage inside a job script:
#   source "${COVERAGE_DIR:-$SLURM_SUBMIT_DIR}/slurm/env.sh"        # CPU stages
#   source "${COVERAGE_DIR:-$SLURM_SUBMIT_DIR}/slurm/env.sh" gpu    # GPU stages

# ~/.bashrc is read before `set -u` so that an unset variable in the login
# files does not abort the job.
source ~/.bashrc

set -euo pipefail

# The scripts of the pipeline address their data relative to the repository
# root ("data/eventcoverage/...", "lib/"), so every stage runs from there.
COVERAGE_DIR="${COVERAGE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$COVERAGE_DIR"

# Make lib/ importable for every stage.
export PYTHONPATH="$COVERAGE_DIR:$COVERAGE_DIR/..:${PYTHONPATH:-}"

# Ignore packages in ~/.local, which would otherwise shadow the environment.
export PYTHONNOUSERSITE=1

# The impact types the pipeline is run for. Their order defines the array task
# ids of the per-type stages (02 and 04).
DTYPES=(cold flood heat storm wildfire)

# The environment slurm/setup_env.sh creates, shared by all stages. It lives
# outside the home directory to stay within the home quota.
#   PIPELINE_ENV_ROOT=...   somewhere else than /p/tmp/$USER/envs
#   PIPELINE_ENV=...        a differently named one, for all stages
#   CONDA_ENV=... CONDA_ENV_GPU=...   separate environments for CPU / GPU stages
PIPELINE_ENV_ROOT="${PIPELINE_ENV_ROOT:-/p/tmp/$USER/envs}"
PIPELINE_ENV="${PIPELINE_ENV:-coverage}"
CONDA_ENV="${CONDA_ENV:-$PIPELINE_ENV}"
CONDA_ENV_GPU="${CONDA_ENV_GPU:-$PIPELINE_ENV}"

module purge
module load anaconda

# A bare name is looked up under PIPELINE_ENV_ROOT first; anything else is
# passed to conda as it is.
activate_env() {
    local env="$1"
    if [[ "$env" != */* && -d "$PIPELINE_ENV_ROOT/$env" ]]; then
        env="$PIPELINE_ENV_ROOT/$env"
    fi
    source activate "$env"
}

if [[ "${1:-cpu}" == "gpu" ]]; then
    activate_env "$CONDA_ENV_GPU"
else
    activate_env "$CONDA_ENV"
fi

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_NAME:-?} ${SLURM_JOB_ID:-?}${SLURM_ARRAY_TASK_ID:+ (task $SLURM_ARRAY_TASK_ID)}"
echo "directory : $PWD"
echo "python    : $(which python)"
echo "started   : $(date --iso-8601=seconds)"

# An outbound connection for the stages that need one: flair downloading its
# model, geopy talking to Nominatim. The compute nodes have no route of their
# own, so the job tunnels a SOCKS proxy back to the submit host and puts an HTTP
# proxy in front of it, which is what the environment variables point at.
#
# Called by the stages that need it; the tunnel and the proxy die with the job.
# The variables are only exported once a test request through the proxy has
# succeeded, otherwise it returns non-zero.
setup_http_proxy() {
    local socks_port http_port pproxy_pid waited log

    if [[ -z "${SLURM_SUBMIT_HOST:-}" ]]; then
        echo "proxy     : no SLURM_SUBMIT_HOST, assuming a direct connection"
        return 0
    fi
    if ! command -v pproxy > /dev/null; then
        echo "proxy     : pproxy not in $CONDA_DEFAULT_ENV, no proxy set up" >&2
        return 1
    fi

    # A random pair of ports, so that two jobs on the same node do not collide.
    socks_port="$(shuf -i 10000-60000 -n 1)"
    http_port="$(shuf -i 10000-60000 -n 1)"
    log="${TMPDIR:-/tmp}/pproxy-$http_port.log"

    # ExitOnForwardFailure so that a port already in use is an error here rather
    # than a tunnel that stands but forwards nothing; BatchMode so that a
    # missing key fails instead of asking a job for a password.
    if ! ssh -o BatchMode=yes -o ExitOnForwardFailure=yes \
             -D "$socks_port" "$SLURM_SUBMIT_HOST" -N -f; then
        echo "proxy     : ssh -D $socks_port to $SLURM_SUBMIT_HOST failed" >&2
        return 1
    fi

    # Backgrounded rather than --daemon, which would need python-daemon.
    pproxy -l "http://:$http_port" -r "socks5://127.0.0.1:$socks_port" > "$log" 2>&1 &
    pproxy_pid=$!

    # Neither the tunnel nor the proxy is a child of this shell -- the tunnel is
    # not, and pproxy is only until the stage's python replaces it -- so they are
    # matched by the ports they were started on.
    trap "pkill -f 'ssh -D $socks_port' || true; pkill -f 'pproxy -l http://:$http_port' || true" EXIT

    # Wait for pproxy to bind, which also catches it dying on startup.
    for (( waited = 0; waited < 20; waited++ )); do
        if timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$http_port" 2> /dev/null; then
            break
        fi
        if ! kill -0 "$pproxy_pid" 2> /dev/null; then
            break
        fi
        sleep 0.5
    done

    # Listening is not the same as routing: the tunnel behind it can be up and
    # still refuse to carry traffic, so the check is a real request.
    if ! python - "$http_port" <<'PY'
import sys, urllib.request

proxy = f"http://127.0.0.1:{sys.argv[1]}"
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": proxy, "https": proxy})
)
try:
    opener.open("https://huggingface.co/api/models/flair/ner-german-large", timeout=30).read(1)
except Exception as e:
    print(f"            {e.__class__.__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PY
    then
        echo "proxy     : no route through http://127.0.0.1:$http_port, pproxy said:" >&2
        sed 's/^/            /' "$log" >&2 || true
        pkill -f "ssh -D $socks_port" || true
        kill "$pproxy_pid" 2> /dev/null || true
        trap - EXIT
        return 1
    fi

    export http_proxy="127.0.0.1:$http_port"
    export https_proxy="127.0.0.1:$http_port"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$https_proxy"

    echo "proxy     : http://127.0.0.1:$http_port -> socks5 via $SLURM_SUBMIT_HOST"
}

# The task id of an array job, mapped onto the impact type it stands for.
dtype_of_task() {
    if [[ -n "${DTYPE:-}" ]]; then
        echo "$DTYPE"
    elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "${DTYPES[$SLURM_ARRAY_TASK_ID]}"
    else
        echo "no DTYPE set and not an array job" >&2
        return 1
    fi
}
