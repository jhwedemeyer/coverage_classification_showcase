# Running the pipeline on a Slurm cluster

The scripts in this folder run the classification pipeline (see the
[main README](../README.md)) on an HPC cluster as a chain of dependent Slurm
jobs. The per-impact-type steps run as array jobs, the NER step gets a GPU,
and location ranking runs at the same time as geocoding.

```bash
./slurm/setup_env.sh        # once, on a login node: build the conda environment
./slurm/submit_pipeline.sh  # from the repository root: submit the job chain
```

```
1 classify_impact_type ─> 2 classify_locations ─┬─> 3 extract_locations ─> 5 map_location_to_country ─┐
                            (GPU, 5 tasks)       └─> 4 rank_locations (5 tasks) ─────────────────────────┴─> 6 gen_coverage_days
```

Every dependency is `afterok`, so a failed step stops everything downstream
instead of passing on half-written output.

| Stage | Script | Resources | Notes |
| --- | --- | --- | --- |
| 1 | `classify_impact_type.py` | 4 CPUs, 64 GB, 6 h | SQLite full-text search, all five impact types in one job |
| 2 | `classify_locations.py` | GPU, 16 CPUs, 60 GB | Array job (one task per impact type), flair NER — the most expensive step |
| 3 | `extract_locations.py` | 2 CPUs, 32 GB, 1 h | Collects the unique location strings |
| 4 | `ranking_locations.py` | 8 CPUs, 48 GB, 12 h | Array job, spaCy sentence segmentation |
| 5 | `map_location_to_country.py` | 1 CPU, 16 GB, 24 h per job | Nominatim geocoding, one request per second |
| 6 | `gen_coverage_days.py` | 4 CPUs, 96 GB, 4 h | Builds the coverage-days panel |

Resource requests are estimates. Check them with
`sacct -j <id> --format=JobID,MaxRSS,Elapsed,State` after the first run.

## Options

```bash
./slurm/submit_pipeline.sh --from 4   # start at stage 4, reuse earlier output
./slurm/submit_pipeline.sh --only 2   # run a single stage
./slurm/submit_pipeline.sh --dry-run  # print the sbatch calls without submitting
```

Configuration is set with environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `ACCOUNT`, `QOS` | Slurm account and QOS | none, required |
| `GPU_QOS`, `GPU_PARTITION`, `GPU_GRES` | GPU request for stage 2 | `gpushort`, `gpu`, `gpu:1` |
| `PIPELINE_ENV` | Conda environment for all stages | `coverage` |
| `DTYPES_ARRAY` | Array range for stages 2 and 4, e.g. `1` to rerun one impact type | `0-4` |
| `GEOCODE_CHAIN` | Number of 24 h geocoding jobs queued back to back | `2` |
| `SKIP_GEOCODE=1` | Leave stage 5 out (e.g. to run it locally) | off |

Logs go to `slurm/logs/<jobname>-<jobid>.{out,err}`.

## Environment

`setup_env.sh` builds a single conda environment from
[environment.yml](environment.yml) for both the CPU and GPU stages. It is
stored under `/p/tmp/$USER/envs/coverage`, outside the home quota. On top of
the package list, the script installs:

- **torch** with the CUDA build that matches the cluster (`--cuda`, default `cu121`)
- the spaCy model **de_core_news_sm**
- flair's **ner-german-large** model, downloaded in advance so stage 2 needs no network access

Run `./slurm/setup_env.sh --check` to verify the installation, and `--force`
to rebuild it. `/p/tmp` is scratch space that gets cleaned up, so the script
is safe to rerun at any time.

## Network access from compute nodes

Compute nodes have no outbound internet access. For the stages that need it,
`env.sh` opens an SSH SOCKS tunnel to the login node, puts an HTTP proxy
(`pproxy`) in front of it, and closes both when the job ends. The proxy
variables are only exported once a test request has succeeded.

- **Stage 5** always uses the proxy to reach Nominatim.
- **Stage 2** uses it only if the NER model is not cached yet. Otherwise it
  runs fully offline.

This requires `pproxy` in the environment and passwordless SSH back to the
login node.

## Before the first run

- **Corpus:** `python src/check_news_db.py` must pass. It checks that
  `data/news021024.db` (83 GB, not in the repo) is present and has its
  full-text index, and can rebuild the index with `--build-fts`.
- **Environment:** `./slurm/setup_env.sh --check` must report a CUDA build of
  torch, not `cpu-only`. Otherwise stage 2 runs on the CPU.
- **GPU time limit:** stage 2 saves its results only at the end, so a task
  that hits the QOS time limit loses all its work. If the larger impact types
  (flood, storm) do not fit, switch to a longer QOS with `GPU_QOS=`.
- **Geocoding:** stage 5 geocodes about 110k locations, which takes well over
  a day. It saves progress regularly and resumes where it stopped, so each of
  the `GEOCODE_CHAIN` jobs continues the previous one.
