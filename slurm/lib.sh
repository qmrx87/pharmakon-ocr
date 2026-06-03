# shellcheck shell=bash
# =============================================================================
# slurm/lib.sh  —  shared runtime library for every VignOCR sbatch job.
#
# This file is *sourced* (never executed) by each slurm/*.sbatch script after
# the #SBATCH directive block. It centralizes the cross-cutting concerns so the
# individual job scripts stay short and the contract is enforced in ONE place:
#
#   • env-var contract           : VIGNOCR_ACCOUNT / VIGNOCR_PI must be set
#   • Narval module stack         : StdEnv + python + cuda (live names overridable)
#   • storage tiers               : /home (code), ~/projects/def-$PI (data),
#                                   ~/scratch (checkpoints + logs + run dirs)
#   • per-run reproducibility      : run dir = <timestamp>-<gitSHA>; snapshot the
#                                   git SHA, the full configs/ tree, the seed,
#                                   the resolved SBATCH env, and pip freeze
#   • resumability                 : stable "latest" symlink per stage so a job
#                                   can be resubmitted and pick up its checkpoint
#
# NOTHING here imports Python or hardcodes a class name / threshold — those live
# in configs/ and are read by the vignocr package. This library only wires the
# HPC plumbing around those config-driven entrypoints.
#
# ---------------------------------------------------------------------------
# REQUIRED ENV (set once in your shell or ~/.bashrc on the Narval login node):
#     export VIGNOCR_ACCOUNT=def-<PI>      # your Slurm allocation, e.g. def-smith
#     export VIGNOCR_PI=<PI>               # the PI shortname, e.g. smith
# The sbatch scripts pass --account=$VIGNOCR_ACCOUNT; this file re-validates it
# at runtime and derives the shared project path ~/projects/def-$VIGNOCR_PI.
# =============================================================================

set -Eeuo pipefail

# --------------------------------------------------------------------------- #
# 0. Module stack — REAL Narval (Digital Research Alliance of Canada) names.
#    These are the modules present on Narval as of 2026; ALWAYS confirm against
#    the live system with `module spider python` / `module spider cuda` and
#    override via env if the cluster has rolled forward (no code edit needed):
#       export VIGNOCR_MODULE_STDENV=StdEnv/2023
#       export VIGNOCR_MODULE_PYTHON=python/3.11
#       export VIGNOCR_MODULE_CUDA=cuda/12.6
# --------------------------------------------------------------------------- #
VIGNOCR_MODULE_STDENV="${VIGNOCR_MODULE_STDENV:-StdEnv/2023}"
VIGNOCR_MODULE_PYTHON="${VIGNOCR_MODULE_PYTHON:-python/3.11}"
VIGNOCR_MODULE_CUDA="${VIGNOCR_MODULE_CUDA:-cuda/12.6}"

# --------------------------------------------------------------------------- #
# Logging helpers (stderr, timestamped). Keep parsing-friendly + greppable.
# --------------------------------------------------------------------------- #
vlog()  { printf '[vignocr %s] %s\n' "$(date +'%Y-%m-%dT%H:%M:%S%z')" "$*" >&2; }
vwarn() { printf '[vignocr %s] WARN: %s\n' "$(date +'%Y-%m-%dT%H:%M:%S%z')" "$*" >&2; }
vdie()  { printf '[vignocr %s] FATAL: %s\n' "$(date +'%Y-%m-%dT%H:%M:%S%z')" "$*" >&2; exit 1; }

# Trap uncaught errors with the failing line number for a friendlier message.
# IMPORTANT: a manually-installed ERR trap fires even under `set +e`, so we make
# it a NO-OP whenever errexit is OFF — this lets stages temporarily `set +e`
# around a tee'd pipeline to capture PIPESTATUS and emit their OWN actionable
# error, without this trap pre-empting them. ($- contains 'e' iff errexit is on.)
_vignocr_err_trap() {
  local rc=$?
  case "$-" in
    *e*) vdie "unexpected error at ${BASH_SOURCE[1]:-?}:${BASH_LINENO[0]:-?} (exit $rc)" ;;
    *)   return 0 ;;   # errexit disabled on purpose -> caller handles the failure
  esac
}
trap '_vignocr_err_trap' ERR

# --------------------------------------------------------------------------- #
# 1. Contract: required env vars. Account + PI are mandatory; everything else
#    has a sane default that can be overridden from the environment.
# --------------------------------------------------------------------------- #
vignocr_require_env() {
  : "${VIGNOCR_ACCOUNT:?set VIGNOCR_ACCOUNT=def-<PI> (your Slurm allocation) before submitting}"
  : "${VIGNOCR_PI:?set VIGNOCR_PI=<PI> (PI shortname, used for ~/projects/def-<PI>) before submitting}"
  if [[ "$VIGNOCR_ACCOUNT" == "def-<PI>" || "$VIGNOCR_PI" == "<PI>" ]]; then
    vdie "VIGNOCR_ACCOUNT / VIGNOCR_PI still hold the placeholder. Set them to your real allocation, e.g. export VIGNOCR_ACCOUNT=def-smith VIGNOCR_PI=smith"
  fi
}

# --------------------------------------------------------------------------- #
# 2. Storage tiers (Narval). Override any of these via env if your layout
#    differs. Defaults follow the Alliance convention.
#       /home          : small, backed-up  -> CODE (this repo lives here)
#       ~/projects/...  : shared, backed-up  -> DATASET + nomenclature
#       ~/scratch       : large, purgeable   -> checkpoints, logs, run dirs, venv
# --------------------------------------------------------------------------- #
vignocr_paths() {
  vignocr_require_env

  # Repo root = the directory that contains configs/classes.yaml. We resolve it
  # relative to this lib file (slurm/lib.sh -> repo root is its parent's parent).
  local _lib_dir
  _lib_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
  export VIGNOCR_REPO_ROOT="${VIGNOCR_REPO_ROOT:-$(cd -- "$_lib_dir/.." >/dev/null 2>&1 && pwd)}"
  [[ -f "$VIGNOCR_REPO_ROOT/configs/classes.yaml" ]] \
    || vdie "VIGNOCR_REPO_ROOT=$VIGNOCR_REPO_ROOT does not contain configs/classes.yaml — point it at the pharmakon-ocr repo root"

  # Shared project space for the dataset (backed up, shared with the PI group).
  export VIGNOCR_PROJECT_DIR="${VIGNOCR_PROJECT_DIR:-$HOME/projects/def-$VIGNOCR_PI/vignocr}"
  # Scratch for everything large + ephemeral (checkpoints, logs, run dirs, venv).
  export VIGNOCR_SCRATCH_DIR="${VIGNOCR_SCRATCH_DIR:-$HOME/scratch/vignocr}"
  export VIGNOCR_VENV_DIR="${VIGNOCR_VENV_DIR:-$VIGNOCR_SCRATCH_DIR/venv}"
  export VIGNOCR_RUNS_DIR="${VIGNOCR_RUNS_DIR:-$VIGNOCR_SCRATCH_DIR/runs}"
  # The detection trainer (src/vignocr/detection/train.py) reads $VIGNOCR_SCRATCH
  # to place RF-DETR checkpoints on the fast/large tier. Point it at scratch.
  export VIGNOCR_SCRATCH="${VIGNOCR_SCRATCH:-$VIGNOCR_SCRATCH_DIR}"
  # Dataset root consumed by the vignocr package. IMPORTANT: the package resolves
  # the dataset root from configs/data.yaml RELATIVE TO THE REPO ROOT (there is no
  # env override for the root path itself). On Narval the real export is far too
  # large for /home, so we keep it in project space and SYMLINK the config-resolved
  # in-repo path to it (see vignocr_link_dataset). VIGNOCR_DATA_ROOT names the real
  # location; eval.run(root=...) is also passed this explicitly.
  export VIGNOCR_DATA_ROOT="${VIGNOCR_DATA_ROOT:-$VIGNOCR_PROJECT_DIR/data}"

  mkdir -p "$VIGNOCR_SCRATCH_DIR" "$VIGNOCR_RUNS_DIR" "$VIGNOCR_PROJECT_DIR"
}

# vignocr_dataset_link_target: echo the in-repo path the package resolves for the
# ACTIVE dataset (configs/data.yaml: datasets.<active>.root, repo-relative). This
# is the path we symlink to the project-space data so training/eval — which read
# the root from config, not argv — see the Narval dataset with no config edit.
vignocr_dataset_link_target() {
  VIGNOCR_REPO_ROOT="$VIGNOCR_REPO_ROOT" \
  VIGNOCR_DATA_ACTIVE="${VIGNOCR_DATA_ACTIVE:-}" \
  python - <<'PY' 2>/dev/null || true
import os
try:
    # Use the package's own resolver so we match it EXACTLY.
    from vignocr.common import get_active_dataset
    print(get_active_dataset()["root"])
except Exception:
    # Fallback before the venv/package is importable: parse data.yaml minimally.
    import re
    root = os.environ["VIGNOCR_REPO_ROOT"]
    active = os.environ.get("VIGNOCR_DATA_ACTIVE") or "synthetic"
    p = os.path.join(root, "configs", "data.yaml")
    rel = "fixtures/synthetic" if active == "synthetic" else "data"
    try:
        import yaml
        with open(p, encoding="utf-8") as fh:
            rel = yaml.safe_load(fh)["datasets"][active]["root"]
    except Exception:
        pass
    print(rel if os.path.isabs(rel) else os.path.join(root, rel))
PY
}

# vignocr_resolved_data_root: echo the dataset root the PACKAGE will actually use
# (configs/data.yaml resolved through get_active_dataset). For synthetic this is
# the repo-relative fixtures/synthetic; for real it is <repo>/data (which we have
# symlinked to project space). Stages that pass an explicit `root` to the library
# API MUST use THIS — not $VIGNOCR_DATA_ROOT — so they agree with config-driven
# stages (training reads the root from config, not argv).
vignocr_resolved_data_root() {
  local r; r="$(vignocr_dataset_link_target)"
  echo "${r:-$VIGNOCR_DATA_ROOT}"
}

# vignocr_link_dataset: ensure the package's config-resolved dataset root points
# at the project-space data ($VIGNOCR_DATA_ROOT). Idempotent. Only creates a
# symlink when the in-repo target does not already exist as a real directory
# (never clobbers a real checkout / fixture).
vignocr_link_dataset() {
  local target src
  target="$(vignocr_dataset_link_target)"
  [[ -n "$target" ]] || { vwarn "could not resolve dataset link target; skipping"; return 0; }
  src="$VIGNOCR_DATA_ROOT"

  if [[ -L "$target" ]]; then
    vlog "dataset symlink already present: $target -> $(readlink -f "$target")"
    return 0
  fi
  if [[ -d "$target" && -n "$(ls -A "$target" 2>/dev/null || true)" ]]; then
    vlog "in-repo dataset dir is real + non-empty ($target) — leaving it as-is"
    return 0
  fi
  if [[ -d "$src" && -n "$(ls -A "$src" 2>/dev/null || true)" ]]; then
    mkdir -p "$(dirname "$target")"
    [[ -d "$target" ]] && rmdir "$target" 2>/dev/null || true   # remove empty placeholder
    ln -sfn "$src" "$target"
    vlog "linked dataset: $target -> $src"
  else
    vwarn "project-space dataset $src is empty — stage data there or generate the fixture (see setup_narval.sh)"
  fi
}

# --------------------------------------------------------------------------- #
# 3. Module stack + venv activation. On compute nodes there is NO internet, so
#    we only ACTIVATE the venv here; creation/install happens on the login node
#    via scripts/setup_narval.sh (documented offline pattern).
# --------------------------------------------------------------------------- #
vignocr_load_modules() {
  if ! command -v module >/dev/null 2>&1; then
    vwarn "no 'module' command found (not on a Narval compute/login node?) — skipping module load"
    return 0
  fi
  vlog "module load $VIGNOCR_MODULE_STDENV $VIGNOCR_MODULE_PYTHON $VIGNOCR_MODULE_CUDA"
  module --force purge 2>/dev/null || true
  # If a module name is stale on the live system, this is the line to adjust
  # (or override VIGNOCR_MODULE_* from your environment — no edit required).
  module load "$VIGNOCR_MODULE_STDENV" "$VIGNOCR_MODULE_PYTHON" "$VIGNOCR_MODULE_CUDA" \
    || vdie "module load failed — run 'module spider python' / 'module spider cuda' on Narval and override VIGNOCR_MODULE_{STDENV,PYTHON,CUDA}"
}

vignocr_activate_venv() {
  [[ -d "$VIGNOCR_VENV_DIR" ]] \
    || vdie "venv missing at $VIGNOCR_VENV_DIR — run scripts/setup_narval.sh on the LOGIN node first (compute nodes have no internet)"
  # shellcheck disable=SC1091
  source "$VIGNOCR_VENV_DIR/bin/activate" \
    || vdie "failed to activate venv at $VIGNOCR_VENV_DIR"
  vlog "venv active: $(command -v python) ($(python --version 2>&1))"
}

# --------------------------------------------------------------------------- #
# 4. Per-run reproducibility.
#    run dir = $VIGNOCR_RUNS_DIR/<stage>/<UTC timestamp>-<gitSHA>[-dirty]
#    Inside it we snapshot: git SHA + status, the full configs/ tree, the seed,
#    the resolved SLURM/Narval env, and (best-effort) pip freeze. A per-stage
#    "latest" symlink makes resubmission resume the most recent run.
# --------------------------------------------------------------------------- #
vignocr_git_sha() {
  ( cd "$VIGNOCR_REPO_ROOT" && git rev-parse --short=12 HEAD 2>/dev/null ) || echo "nogit"
}
vignocr_git_dirty() {
  ( cd "$VIGNOCR_REPO_ROOT" && [[ -n "$(git status --porcelain 2>/dev/null)" ]] ) && echo "-dirty" || echo ""
}

# vignocr_resolve_seed: single source of truth for the seed is configs/data.yaml
# (datasets.synthetic.seed). We read it with the stdlib only (no PyYAML import)
# so this works even before the venv is active; default 1337 if unreadable.
vignocr_resolve_seed() {
  local seed
  seed="$(VIGNOCR_REPO_ROOT="$VIGNOCR_REPO_ROOT" python - <<'PY' 2>/dev/null || true
import os, re, sys
p = os.path.join(os.environ["VIGNOCR_REPO_ROOT"], "configs", "data.yaml")
try:
    import yaml
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    print(int(cfg["datasets"]["synthetic"]["seed"]))
except Exception:
    # stdlib-only fallback: grep the first 'seed:' line
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                m = re.search(r"seed:\s*(\d+)", line)
                if m:
                    print(int(m.group(1))); break
    except Exception:
        pass
PY
)"
  echo "${seed:-${VIGNOCR_SEED:-1337}}"
}

# vignocr_new_run_dir <stage>  -> echoes the absolute run-dir path on stdout.
# Side effect: writes the reproducibility snapshot and updates the latest symlink.
vignocr_new_run_dir() {
  local stage="${1:?vignocr_new_run_dir needs a stage name}"
  local ts sha dirty run_dir
  ts="$(date -u +'%Y%m%dT%H%M%SZ')"
  sha="$(vignocr_git_sha)"
  dirty="$(vignocr_git_dirty)"
  run_dir="$VIGNOCR_RUNS_DIR/$stage/${ts}-${sha}${dirty}"
  mkdir -p "$run_dir"

  # --- snapshot: git provenance ---
  {
    echo "stage:      $stage"
    echo "timestamp:  $ts"
    echo "git_sha:    $sha"
    echo "git_dirty:  ${dirty:-clean}"
    echo "repo_root:  $VIGNOCR_REPO_ROOT"
    echo "host:       $(hostname)"
    echo "slurm_job:  ${SLURM_JOB_ID:-<none>}"
    echo "account:    $VIGNOCR_ACCOUNT"
  } > "$run_dir/RUN_INFO.txt"
  ( cd "$VIGNOCR_REPO_ROOT" && git status --porcelain 2>/dev/null ) > "$run_dir/git_status.txt" || true
  ( cd "$VIGNOCR_REPO_ROOT" && git rev-parse HEAD 2>/dev/null ) > "$run_dir/git_sha_full.txt" || true

  # --- snapshot: full configs/ tree (the exact config that produced this run) ---
  if [[ -d "$VIGNOCR_REPO_ROOT/configs" ]]; then
    cp -R "$VIGNOCR_REPO_ROOT/configs" "$run_dir/configs_snapshot"
  fi

  # --- snapshot: seed (single source = configs/data.yaml) ---
  vignocr_resolve_seed > "$run_dir/seed.txt"

  # --- snapshot: resolved SLURM + Narval environment (greppable) ---
  {
    echo "# resolved environment for $stage @ $ts"
    echo "VIGNOCR_MODULE_STDENV=$VIGNOCR_MODULE_STDENV"
    echo "VIGNOCR_MODULE_PYTHON=$VIGNOCR_MODULE_PYTHON"
    echo "VIGNOCR_MODULE_CUDA=$VIGNOCR_MODULE_CUDA"
    echo "VIGNOCR_PROJECT_DIR=$VIGNOCR_PROJECT_DIR"
    echo "VIGNOCR_SCRATCH_DIR=$VIGNOCR_SCRATCH_DIR"
    echo "VIGNOCR_VENV_DIR=$VIGNOCR_VENV_DIR"
    echo "VIGNOCR_DATA_ROOT=$VIGNOCR_DATA_ROOT"
    echo "VIGNOCR_DATA_ACTIVE=${VIGNOCR_DATA_ACTIVE:-<data.yaml default>}"
    env | grep -E '^SLURM_' | sort || true
  } > "$run_dir/env_snapshot.txt"

  # --- maintain a stable per-stage "latest" symlink for resumption ---
  ln -sfn "$run_dir" "$VIGNOCR_RUNS_DIR/$stage/latest"

  echo "$run_dir"
}

# vignocr_snapshot_freeze <run_dir>: best-effort `pip freeze` once the venv is
# active (records the EXACT installed wheel versions for this run).
vignocr_snapshot_freeze() {
  local run_dir="${1:?need run_dir}"
  python -m pip freeze > "$run_dir/pip_freeze.txt" 2>/dev/null || vwarn "pip freeze unavailable"
}

# vignocr_latest_run <stage>: echo the most recent run dir for a stage (for the
# downstream stage to find its input), or empty if none.
vignocr_latest_run() {
  local stage="${1:?need stage}"
  local link="$VIGNOCR_RUNS_DIR/$stage/latest"
  [[ -L "$link" || -d "$link" ]] && readlink -f "$link" || true
}

# vignocr_find_resume <run_dir>: echo a checkpoint to resume RF-DETR from, if a
# prior run of THIS run_dir was interrupted (idempotent resubmission). The
# detection trainer places checkpoints under $VIGNOCR_SCRATCH/<cfg dir>/<run name>
# (NOT inside run_dir), so we search there as well as the run dir itself. RF-DETR
# writes `checkpoint.pth` (latest) + `checkpoint_best_total.pth` (best); we prefer
# the latest for resuming, falling back to the best, then any newest *.pth/*.pt.
vignocr_find_resume() {
  local run_dir="${1:?need run_dir}"
  local run_name; run_name="$(basename "$run_dir")"
  local search_dirs=(
    "$run_dir/checkpoints"
    "$run_dir"
    "$VIGNOCR_SCRATCH/detection/rfdetr_medium/$run_name"
    "$VIGNOCR_SCRATCH_DIR/detection/rfdetr_medium/$run_name"
  )
  local d cand
  for d in "${search_dirs[@]}"; do
    [[ -d "$d" ]] || continue
    for cand in checkpoint.pth checkpoint_best_total.pth checkpoint_best.pth last.ckpt last.pth; do
      if [[ -f "$d/$cand" ]]; then echo "$d/$cand"; return 0; fi
    done
  done
  # last resort: newest checkpoint-like file across the search dirs
  for d in "${search_dirs[@]}"; do
    [[ -d "$d" ]] || continue
    cand="$(ls -1t "$d"/*.pth "$d"/*.pt "$d"/*.ckpt 2>/dev/null | head -n1 || true)"
    [[ -n "$cand" ]] && { echo "$cand"; return 0; }
  done
  echo ""
}

# vignocr_find_best <run_dir>: echo the BEST detector checkpoint produced by a
# finished training run (for eval/export/benchmark to consume). Same search path
# as resume but prefers the best-total artifact RF-DETR writes.
vignocr_find_best() {
  local run_dir="${1:?need run_dir}"
  local run_name; run_name="$(basename "$run_dir")"
  local d cand
  for d in "$run_dir/checkpoints" "$run_dir" \
           "$VIGNOCR_SCRATCH/detection/rfdetr_medium/$run_name" \
           "$VIGNOCR_SCRATCH_DIR/detection/rfdetr_medium/$run_name"; do
    [[ -d "$d" ]] || continue
    for cand in checkpoint_best_total.pth checkpoint_best.pth checkpoint_best_ema.pth best.pth best.ckpt; do
      if [[ -f "$d/$cand" ]]; then echo "$d/$cand"; return 0; fi
    done
  done
  for d in "$run_dir/checkpoints" "$run_dir" \
           "$VIGNOCR_SCRATCH/detection/rfdetr_medium/$run_name" \
           "$VIGNOCR_SCRATCH_DIR/detection/rfdetr_medium/$run_name"; do
    [[ -d "$d" ]] || continue
    cand="$(ls -1t "$d"/*.pth "$d"/*.pt 2>/dev/null | head -n1 || true)"
    [[ -n "$cand" ]] && { echo "$cand"; return 0; }
  done
  echo ""
}

# vignocr_preamble <stage>: the standard opening every compute job runs.
# Validates env, sets up paths, loads modules, activates venv, seeds, and
# creates+returns a fresh run dir (exported as VIGNOCR_RUN_DIR). Resumption:
# if VIGNOCR_RESUME_RUN_DIR is set, reuse it instead of minting a new one.
vignocr_preamble() {
  local stage="${1:?vignocr_preamble needs a stage name}"
  vignocr_require_env
  vignocr_paths
  vignocr_load_modules
  vignocr_activate_venv
  # Point the package's config-resolved dataset root at the project-space data.
  vignocr_link_dataset
  # The actual root the package will read (synthetic: fixtures/synthetic; real:
  # <repo>/data -> project space). Stages passing an explicit `root` use this.
  export VIGNOCR_RESOLVED_DATA_ROOT
  VIGNOCR_RESOLVED_DATA_ROOT="$(vignocr_resolved_data_root)"

  if [[ -n "${VIGNOCR_RESUME_RUN_DIR:-}" ]]; then
    [[ -d "$VIGNOCR_RESUME_RUN_DIR" ]] \
      || vdie "VIGNOCR_RESUME_RUN_DIR=$VIGNOCR_RESUME_RUN_DIR does not exist"
    export VIGNOCR_RUN_DIR="$VIGNOCR_RESUME_RUN_DIR"
    ln -sfn "$VIGNOCR_RUN_DIR" "$VIGNOCR_RUNS_DIR/$stage/latest"
    vlog "RESUMING stage=$stage run_dir=$VIGNOCR_RUN_DIR"
  else
    VIGNOCR_RUN_DIR="$(vignocr_new_run_dir "$stage")"
    export VIGNOCR_RUN_DIR
    vlog "NEW run stage=$stage run_dir=$VIGNOCR_RUN_DIR"
  fi
  vignocr_snapshot_freeze "$VIGNOCR_RUN_DIR"

  export VIGNOCR_SEED
  VIGNOCR_SEED="$(cat "$VIGNOCR_RUN_DIR/seed.txt" 2>/dev/null || vignocr_resolve_seed)"
  vlog "seed=$VIGNOCR_SEED  data_active=${VIGNOCR_DATA_ACTIVE:-<data.yaml default>}"
}
