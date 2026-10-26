#!/bin/bash
# Create the conda environment the coverage classification pipeline runs in.
#
# Run this once on a login node -- it needs an outbound connection, which the
# compute nodes only have through the proxy, and it downloads a few GB. The
# environment it creates is the one every stage of the pipeline activates
# (PIPELINE_ENV in env.sh), for the CPU stages and the GPU stage alike: one
# torch installation with CUDA support serves both.
#
#   ./slurm/setup_env.sh                    # create it
#   ./slurm/setup_env.sh --force            # replace an existing one
#   ./slurm/setup_env.sh --name coverage2   # under a different name
#   ./slurm/setup_env.sh --root /p/tmp/...  # somewhere else than /p/tmp/$USER/envs
#   ./slurm/setup_env.sh --cuda cpu         # torch without CUDA
#   ./slurm/setup_env.sh --no-flair-model   # do not pre-fetch ner-german-large
#   ./slurm/setup_env.sh --check            # only verify what is there
#
# The environment is created as a directory under /p/tmp/$USER/envs, not by
# name in the home directory: torch, transformers and their dependencies come to
# several gigabytes, and the package caches conda and pip fill on the way are
# put next to it for the same reason.
#
# What it installs beyond environment.yml, and why it cannot live in there:
#   torch              the wheel index depends on the cluster's CUDA version
#   de_core_news_sm    spaCy models are downloaded, not packaged
#   ner-german-large   flair's model, pre-fetched into ~/.flair so that the job
#                      does not depend on the proxy working

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_ROOT="${PIPELINE_ENV_ROOT:-/p/tmp/$USER/envs}"
ENV_NAME="${PIPELINE_ENV:-coverage}"
# cu121 runs against any driver from 530 on, which covers the GPU nodes; "cpu"
# installs the CPU-only build, "cu118" the older one. `nvidia-smi` on a GPU node
# prints the driver's CUDA version.
CUDA="${CUDA:-cu121}"
FORCE=0
FLAIR_MODEL=1
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) ENV_NAME="$2"; shift 2 ;;
        --root) ENV_ROOT="$2"; shift 2 ;;
        --cuda) CUDA="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --no-flair-model) FLAIR_MODEL=0; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# `set -u` is lifted for the login files, which reference variables of their own
# that they do not always set.
set +u
source ~/.bashrc
set -u

# `module` is the cluster; on a laptop conda is simply on the PATH already, and
# the script then works there too.
if command -v module > /dev/null; then
    module purge
    module load anaconda
fi

if ! command -v conda > /dev/null; then
    echo "conda is not on the PATH -- on the cluster, 'module load anaconda' first" >&2
    exit 1
fi

# mamba resolves this environment in a fraction of the time conda needs, so it
# is used where it is there.
if command -v mamba > /dev/null; then
    CONDA=mamba
else
    CONDA=conda
fi

ENV_PREFIX="$ENV_ROOT/$ENV_NAME"

# The package caches belong next to the environment, not into the home
# directory: conda's alone grows to several gigabytes while resolving this.
CACHE_ROOT="$(dirname "$ENV_ROOT")"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_ROOT/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip_cache}"

# Ignore packages in ~/.local: they would shadow the environment, and pip would
# skip installing them into it. env.sh does the same for the jobs.
export PYTHONNOUSERSITE=1
export PIP_USER=0

env_exists() {
    [[ -x "$ENV_PREFIX/bin/python" ]]
}

verify() {
    echo
    echo "== checking $ENV_NAME"
    python - <<'PY'
import importlib, sys

# module name -> the stage that needs it
MODULES = {
    "pandas": "1, 3, 4, 6", "numpy": "4, 6", "pyarrow": "1-4", "yaml": "1, 4",
    "tqdm": "2, 4", "matplotlib": "1", "spacy": "4", "geopy": "5",
    "countrynames": "6", "torch": "2", "flair": "2",
}

missing = []
outside = []
for module, stages in sorted(MODULES.items()):
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "?")
        # An import that resolves outside the environment comes from
        # ~/.local/lib/python3.11/site-packages and shadows what is installed
        # here, which is what PYTHONNOUSERSITE keeps out.
        path = getattr(m, "__file__", "") or ""
        where = ""
        if path and not path.startswith(sys.prefix):
            outside.append(module)
            where = f"  OUTSIDE THE ENVIRONMENT: {path}"
        print(f"  {module:<14} {version:<12} stage {stages}{where}")
    except Exception as e:
        missing.append(module)
        print(f"  {module:<14} MISSING      stage {stages}  ({e.__class__.__name__})")

try:
    spacy = importlib.import_module("spacy")
    spacy.load("de_core_news_sm")
    print("  de_core_news_sm  installed    stage 4")
except Exception:
    missing.append("de_core_news_sm")
    print("  de_core_news_sm  MISSING      stage 4")

import shutil
print(f"  pproxy           {'found' if shutil.which('pproxy') else 'MISSING'}        proxy for stages 2, 5")

# Stage 1 searches an FTS5 index; an sqlite without the module reports the
# index as a missing table, which says nothing about the real cause.
import sqlite3
con = sqlite3.connect(":memory:")
try:
    con.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
    print(f"  sqlite           {sqlite3.sqlite_version:<12} fts5 ok, stage 1")
except sqlite3.OperationalError:
    missing.append("sqlite fts5")
    print(f"  sqlite           {sqlite3.sqlite_version:<12} NO FTS5 MODULE, stage 1 cannot search")

try:
    import torch
    print(f"  torch build      {torch.version.cuda or 'cpu-only'}")
    print(f"  cuda visible     {torch.cuda.is_available()}  (False on a login node is expected)")
except Exception:
    pass

# Stage 2 runs with HF_HUB_OFFLINE=1 when the model is in this cache, and needs
# a working proxy on the compute node when it is not, so which of the two it is
# belongs in the report.
import os, pathlib
flair_root = pathlib.Path(os.environ.get("FLAIR_CACHE_ROOT") or pathlib.Path.home() / ".flair")
snapshots = flair_root / "models/ner-german-large/models--flair--ner-german-large/snapshots"
if any(snapshots.glob("*/pytorch_model.bin")):
    print(f"  ner-german-large cached       stage 2 runs offline ({flair_root})")
else:
    print(f"  ner-german-large NOT CACHED   stage 2 downloads it through the proxy ({flair_root})")

if outside:
    print(
        "\n  " + ", ".join(outside) + " are imported from outside the environment."
        "\n  Run the pipeline with PYTHONNOUSERSITE=1 (env.sh does) or clean up"
        "\n  ~/.local/lib/python3.11/site-packages."
    )

sys.exit(1 if missing or outside else 0)
PY
}

if (( CHECK_ONLY )); then
    if ! env_exists; then
        echo "no environment at $ENV_PREFIX" >&2
        exit 1
    fi
    source activate "$ENV_PREFIX"
    verify
    exit
fi

if env_exists; then
    if (( FORCE )); then
        echo "== removing the existing $ENV_PREFIX"
        conda env remove -p "$ENV_PREFIX" -y
        rm -rf "$ENV_PREFIX"
    else
        echo "an environment exists already at $ENV_PREFIX." >&2
        echo "re-run with --force to replace it, or with --check to verify it." >&2
        exit 1
    fi
fi

mkdir -p "$ENV_ROOT"

echo "== creating $ENV_PREFIX from slurm/environment.yml (with $CONDA)"
# -p, not -n: the environment is a directory on /p/tmp. The name in
# environment.yml would collide with that and is therefore dropped.
$CONDA env create -p "$ENV_PREFIX" -f "$SLURM_DIR/environment.yml"

source activate "$ENV_PREFIX"

# torch first, flair second: installed the other way round, flair brings a torch
# of its own from PyPI and the wheel from the CUDA index is then skipped as
# already satisfied. `python -m pip`, not `pip`, so that it is the environment's
# interpreter doing the installing whatever else the PATH may offer.
echo
echo "== installing torch ($CUDA)"
if [[ "$CUDA" == "cpu" ]]; then
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    python -m pip install torch --index-url "https://download.pytorch.org/whl/$CUDA"
fi

echo
echo "== installing flair (stage 2)"
python -m pip install "flair>=0.14"

echo
echo "== installing the German spaCy model (stage 4)"
python -m spacy download de_core_news_sm

if (( FLAIR_MODEL )); then
    echo
    echo "== pre-fetching flair's ner-german-large into ${FLAIR_CACHE_ROOT:-$HOME/.flair} (stage 2)"
    echo "   about 2.1G; skip it with --no-flair-model and let the job download it"
    python -c "from flair.models import SequenceTagger; SequenceTagger.load('flair/ner-german-large')"
fi

verify

cat <<EOF

$ENV_PREFIX

The pipeline finds it there on its own as long as it is called "coverage" and
sits under /p/tmp/$USER/envs; otherwise submit with

    PIPELINE_ENV=$ENV_NAME PIPELINE_ENV_ROOT=$ENV_ROOT ./slurm/submit_pipeline.sh

Interactively, use it with

    module load anaconda && source activate $ENV_PREFIX

/p/tmp is scratch and is swept of files that have not been touched for a while,
so if a stage starts failing on a missing interpreter, run this script again.
EOF
