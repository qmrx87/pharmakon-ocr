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
#   • centralized SLURM logs       : <repo>/logs/slurm/<stage>/<jobid>/ holds
#                                   stdout/stderr + a copy of the train log
#                                   (so post-mortem is ONE directory, not 3)
#   • per-run reproducibility      : run dir = <timestamp>-<gitSHA>; snapshot the
#                                   git SHA, the full configs/ tree, the seed,
#                                   the resolved SBATCH env, and pip freeze
#   • offline pretrained weights   : cache RF-DETR's COCO init on the LOGIN node
#                                   (compute nodes have NO internet); the trainer
#                                   reads from $VIGNOCR_PRETRAINED_DIR
#   • resumability                 : stable "latest" symlink per stage so a job
#                                   can be resubmitted and pick up its checkpoint
#   • automatic post-mortem        : on FAILURE, copy the inline train_*.log into
#                                   logs/slurm/<stage>/<jobid>/ and tail it to
#                                   stderr so the .err file is self-contained
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
VIGNOCR_MODULE_OPENCV="${VIGNOCR_MODULE_OPENCV:-opencv}"
# hdf5: ONLY the v2 docTR path needs it. python-doctr imports `h5py` at
# `import doctr` (via its SVHN dataset loader), and the Compute Canada h5py
# wheel must link the MATCHING hdf5 module — without it you get
# `ImportError: ... libhdf5_hl.so: undefined symbol: H5T_IEEE_F16BE_g`
# (a skewed libhdf5 set). Loaded NON-FATALLY below so v1 (rfdetr/paddle, which
# never imports h5py) is never broken by a missing/renamed hdf5 module.
VIGNOCR_MODULE_HDF5="${VIGNOCR_MODULE_HDF5:-hdf5}"

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
# 1. Contract: required env vars. Account + PI are mandatory on the SUBMISSION
#    node (login). Inside a running Slurm job we don't depend on --export=ALL
#    carrying them — we DERIVE both from $SLURM_JOB_ACCOUNT (which Slurm always
#    populates inside every job, regardless of the user's shell env). This is
#    important because some Compute Canada policies strip user-facing env vars
#    that match certain patterns, so the job can't trust the submitter's env.
#
# Account format on Compute Canada is `def-<PI>` with an optional partition
# suffix Slurm adds at runtime (e.g. `def-khenni_g` on a GPU partition). We
# strip both the `def-` prefix and any trailing `_X` suffix to recover the PI
# shortname.
# --------------------------------------------------------------------------- #
vignocr_derive_account_from_slurm() {
  # Echoes (account, pi) on stdout (space-separated) if SLURM_JOB_ACCOUNT is
  # set; empty otherwise. Caller decides whether to consume the values.
  local acct="${SLURM_JOB_ACCOUNT:-}"
  [[ -z "$acct" ]] && return 0
  # Strip Slurm partition suffix (`_g`, `_c`, ...). e.g. def-khenni_g -> def-khenni.
  local clean="${acct%_*}"
  [[ "$clean" == "$acct" ]] && clean="$acct"   # no suffix? keep as-is
  # PI = the part after `def-` (also handle rrg-/ctb- allocation prefixes).
  local pi="${clean#def-}"; pi="${pi#rrg-}"; pi="${pi#ctb-}"
  echo "$clean $pi"
}

# vignocr_derive_account_from_system: discover the user's Compute Canada
# allocation on the LOGIN node, without any env vars set. Tries (in order):
#   1. sacctmgr show user $USER --format=DefaultAccount    (canonical)
#   2. `id -Gn` filtered for def-/rrg-/ctb- group memberships (fallback)
# Echoes a single account name on stdout or nothing.
vignocr_derive_account_from_system() {
  local acct=""
  if command -v sacctmgr >/dev/null 2>&1; then
    acct="$(sacctmgr show user "$USER" --format=DefaultAccount -P -n 2>/dev/null \
             | head -n1 | tr -d '[:space:]')"
  fi
  if [[ -z "$acct" ]] && command -v id >/dev/null 2>&1; then
    # First def-/rrg-/ctb- group, bare (no `_cpu`/`_gpu`/etc. suffix).
    acct="$(id -Gn 2>/dev/null | tr ' ' '\n' \
             | grep -E '^(def|rrg|ctb)-[A-Za-z0-9._-]+$' \
             | grep -v '_' \
             | head -n1)"
  fi
  echo "$acct"
}

vignocr_require_env() {
  # SOURCE OF TRUTH (in order):
  #   1. explicit user env (VIGNOCR_ACCOUNT / VIGNOCR_PI already set)
  #   2. inside-job: $SLURM_JOB_ACCOUNT (Slurm sets it; survives any policy
  #      that strips user-named env)
  #   3. login-node: sacctmgr DefaultAccount, then `id -Gn` def-/rrg-/ctb-
  # The script only needs ANY of these to work; nothing has to be exported.

  # (2) inside a Slurm job
  if [[ -z "${VIGNOCR_ACCOUNT:-}" || -z "${VIGNOCR_PI:-}" ]] && [[ -n "${SLURM_JOB_ACCOUNT:-}" ]]; then
    local derived; derived="$(vignocr_derive_account_from_slurm)"
    if [[ -n "$derived" ]]; then
      local d_acct d_pi
      d_acct="${derived% *}"; d_pi="${derived#* }"
      : "${VIGNOCR_ACCOUNT:=$d_acct}"
      : "${VIGNOCR_PI:=$d_pi}"
      export VIGNOCR_ACCOUNT VIGNOCR_PI
      vlog "derived env from SLURM_JOB_ACCOUNT=$SLURM_JOB_ACCOUNT: VIGNOCR_ACCOUNT=$VIGNOCR_ACCOUNT VIGNOCR_PI=$VIGNOCR_PI"
    fi
  fi

  # (3) login node — no SLURM_JOB_ACCOUNT, ask the system who we are.
  if [[ -z "${VIGNOCR_ACCOUNT:-}" ]]; then
    local sys_acct; sys_acct="$(vignocr_derive_account_from_system)"
    if [[ -n "$sys_acct" ]]; then
      export VIGNOCR_ACCOUNT="$sys_acct"
      vlog "derived VIGNOCR_ACCOUNT=$VIGNOCR_ACCOUNT from system (sacctmgr / id -Gn)"
    fi
  fi
  # Derive PI from whichever account we ended up with.
  if [[ -n "${VIGNOCR_ACCOUNT:-}" && -z "${VIGNOCR_PI:-}" ]]; then
    local pi="${VIGNOCR_ACCOUNT%_*}"
    pi="${pi#def-}"; pi="${pi#rrg-}"; pi="${pi#ctb-}"
    export VIGNOCR_PI="$pi"
    vlog "derived VIGNOCR_PI=$VIGNOCR_PI from VIGNOCR_ACCOUNT=$VIGNOCR_ACCOUNT"
  fi

  # If after all of that they're still empty, surface a CLEAR actionable error.
  if [[ -z "${VIGNOCR_ACCOUNT:-}" || -z "${VIGNOCR_PI:-}" ]]; then
    vdie "could not resolve VIGNOCR_ACCOUNT/VIGNOCR_PI automatically. Diagnostics:
    SLURM_JOB_ACCOUNT       = '${SLURM_JOB_ACCOUNT:-<unset>}'
    sacctmgr DefaultAccount = '$(command -v sacctmgr >/dev/null 2>&1 && sacctmgr show user $USER --format=DefaultAccount -P -n 2>/dev/null | head -n1)'
    id -Gn (filtered)       = '$(id -Gn 2>/dev/null | tr ' ' '\n' | grep -E '^(def|rrg|ctb)-' | tr '\n' ' ')'
Manual override:  export VIGNOCR_ACCOUNT=def-<PI>  VIGNOCR_PI=<PI>"
  fi
  if [[ "$VIGNOCR_ACCOUNT" == "def-<PI>" || "$VIGNOCR_PI" == "<PI>" ]]; then
    vdie "VIGNOCR_ACCOUNT / VIGNOCR_PI still hold the placeholder. Set them to your real allocation, e.g. export VIGNOCR_ACCOUNT=def-smith VIGNOCR_PI=smith"
  fi
}

# vignocr_assert_account_resolved: the .sbatch directive block carries a literal
# `--account=def-<PI>` placeholder. submit_all.sh overrides it on the command line
# via `--account=$VIGNOCR_ACCOUNT`, but if a user runs `sbatch slurm/02_...sbatch`
# DIRECTLY without that override, Slurm would have already rejected it pre-launch.
# Inside the running job, SLURM_JOB_ACCOUNT exposes the resolved value — assert
# it's not the placeholder (defense-in-depth; catches misconfigured submission).
vignocr_assert_account_resolved() {
  local acct="${SLURM_JOB_ACCOUNT:-}"
  if [[ -n "$acct" && "$acct" == "def-<PI>" ]]; then
    vdie "SLURM_JOB_ACCOUNT is the literal placeholder 'def-<PI>' — submit via submit_all.sh, or pass --account=\$VIGNOCR_ACCOUNT to sbatch directly"
  fi
  # If we ran outside Slurm (local invocation for testing), there's no
  # SLURM_JOB_ACCOUNT; vignocr_require_env already covered the env contract.
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

  # Centralized SLURM log root (one tree, per-stage, per-job). Lives in the REPO
  # so post-mortem is `cd <repo>; ls logs/slurm/<stage>/` — no scratch hunting.
  # Per-job dir is created lazily once we know the stage + job id.
  export VIGNOCR_LOGS_DIR="${VIGNOCR_LOGS_DIR:-$VIGNOCR_REPO_ROOT/logs/slurm}"

  # Pretrained-weight cache. Compute nodes have NO internet so RF-DETR (and any
  # huggingface/torchvision hub call) MUST resolve from a local file. Pre-fetched
  # on the login node by scripts/fetch_pretrained.sh into PROJECT space (backed up,
  # shared across runs). Trainer code reads from here.
  export VIGNOCR_PRETRAINED_DIR="${VIGNOCR_PRETRAINED_DIR:-$VIGNOCR_PROJECT_DIR/checkpoints/pretrained}"

  # Bake an offline-by-default stance for every Hub-like cache the wheelhouse
  # might consult — torchvision / huggingface / paddleocr all check these. This is
  # belt-and-suspenders: the trainer should never NEED the network, but if a
  # third-party dependency tries, we want a CLEAR error, not a 6-minute hang.
  if [[ -z "${HF_HUB_OFFLINE:-}" ]]; then export HF_HUB_OFFLINE=1; fi
  if [[ -z "${TRANSFORMERS_OFFLINE:-}" ]]; then export TRANSFORMERS_OFFLINE=1; fi
  if [[ -z "${HF_DATASETS_OFFLINE:-}" ]]; then export HF_DATASETS_OFFLINE=1; fi
  if [[ -z "${TORCH_HOME:-}" ]]; then export TORCH_HOME="$VIGNOCR_PRETRAINED_DIR/torch"; fi
  if [[ -z "${HF_HOME:-}" ]]; then export HF_HOME="$VIGNOCR_PRETRAINED_DIR/huggingface"; fi
  # docTR (v2b full-page OCR) downloads det/rec weights on first use — point its
  # cache into the shared pretrained dir so the login-node prefetch
  # (scripts/fetch_pretrained_v2.sh) is what offline compute nodes read.
  if [[ -z "${DOCTR_CACHE_DIR:-}" ]]; then export DOCTR_CACHE_DIR="$VIGNOCR_PRETRAINED_DIR/doctr"; fi

  mkdir -p \
    "$VIGNOCR_SCRATCH_DIR" "$VIGNOCR_RUNS_DIR" "$VIGNOCR_PROJECT_DIR" \
    "$VIGNOCR_LOGS_DIR" "$VIGNOCR_PRETRAINED_DIR" \
    "$TORCH_HOME" "$HF_HOME" "$DOCTR_CACHE_DIR"
}

# vignocr_logs_dir <stage> [jobid] -> echo (and create) the central log dir for a
# given stage + job. Pattern: <repo>/logs/slurm/<stage>/<jobid>/. When jobid is
# absent (local run), uses "local-<pid>" so a manual invocation still lands in a
# stable bucket. Called by vignocr_preamble — every stage gets one.
vignocr_logs_dir() {
  local stage="${1:?vignocr_logs_dir needs a stage name}"
  local jid="${2:-${SLURM_JOB_ID:-local-$$}}"
  local d="$VIGNOCR_LOGS_DIR/$stage/$jid"
  mkdir -p "$d"
  echo "$d"
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
  vlog "module load $VIGNOCR_MODULE_STDENV $VIGNOCR_MODULE_PYTHON $VIGNOCR_MODULE_CUDA $VIGNOCR_MODULE_OPENCV"
  module --force purge 2>/dev/null || true
  # If a module name is stale on the live system, this is the line to adjust
  # (or override VIGNOCR_MODULE_* from your environment — no edit required).
  module load "$VIGNOCR_MODULE_STDENV" "$VIGNOCR_MODULE_PYTHON" "$VIGNOCR_MODULE_CUDA" "$VIGNOCR_MODULE_OPENCV" \
    || vdie "module load failed — run 'module spider python' / 'module spider cuda' on Narval and override VIGNOCR_MODULE_{STDENV,PYTHON,CUDA,OPENCV}"
  # hdf5 for the v2 docTR path (see the VIGNOCR_MODULE_HDF5 note above). NON-FATAL:
  # v1 never imports h5py, so a missing hdf5 module must not abort training. If
  # the default doesn't match the installed h5py wheel, override the version, e.g.
  #   export VIGNOCR_MODULE_HDF5=hdf5/1.14.6
  module load "$VIGNOCR_MODULE_HDF5" 2>/dev/null \
    || vwarn "could not load hdf5 module ('$VIGNOCR_MODULE_HDF5') — only the v2 docTR path needs it; v1/v2a unaffected. If v2 doctr fails with an h5py 'undefined symbol' error, run 'module spider hdf5' and set VIGNOCR_MODULE_HDF5."

  # h5py-vs-opencv soname race. python-doctr's `doctr.io` imports cv2 (the opencv
  # module) BEFORE `doctr.models` imports h5py. opencv links an OLDER libhdf5 with
  # the SAME soname (libhdf5.so.310), so it loads first and h5py's libhdf5_hl then
  # can't resolve the float16 symbol H5T_IEEE_F16BE_g (added in HDF5 1.14.4):
  #   ImportError: .../libhdf5_hl.so.310: undefined symbol: H5T_IEEE_F16BE_g
  # Force the hdf5-MODULE's (newer) libhdf5 to win process-wide via LD_PRELOAD so
  # both cv2 and h5py share it. Guarded on the module actually loading (EBROOTHDF5
  # set by Lmod) and the files existing. Harmless to v1/v2a (nothing else needs a
  # specific hdf5; cv2's core image ops are unaffected by a newer-patch hdf5).
  if [[ -n "${EBROOTHDF5:-}" && -e "$EBROOTHDF5/lib/libhdf5.so" ]]; then
    export LD_PRELOAD="$EBROOTHDF5/lib/libhdf5.so${LD_PRELOAD:+:$LD_PRELOAD}"
    [[ -e "$EBROOTHDF5/lib/libhdf5_hl.so" ]] \
      && export LD_PRELOAD="$EBROOTHDF5/lib/libhdf5_hl.so:$LD_PRELOAD"
    vlog "LD_PRELOAD set for hdf5 ($EBROOTHDF5/lib) so docTR's h5py and opencv share one libhdf5"
  fi
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

# vignocr_warn_if_stale: REFUSE to submit when this checkout is BEHIND its git
# upstream. This is the #1 recurring failure on Narval: a stale `git pull` leaves
# the compute jobs running OLD code, so a fix already on origin "still fails"
# (e.g. job 63684084 ran 21bb809-dirty and used the pre-TrOCR doctr default that
# was fixed two commits earlier). Fetches first (login nodes have network),
# compares HEAD to @{u}: behind -> vdie unless VIGNOCR_SKIP_GIT_CHECK=1; a dirty
# tree -> warn only; no upstream / git absent / offline -> warn only. Call this
# at the TOP of every submit_* script, after vignocr_paths.
vignocr_warn_if_stale() {
  command -v git >/dev/null 2>&1 || return 0
  local root="${VIGNOCR_REPO_ROOT:-.}"
  git -C "$root" rev-parse --git-dir >/dev/null 2>&1 || return 0
  git -C "$root" fetch --quiet 2>/dev/null || vwarn "git fetch failed (offline?) — staleness check is best-effort"
  local head up behind
  head="$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo '?')"
  up="$(git -C "$root" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || echo '')"
  if [[ -n "$up" ]]; then
    behind="$(git -C "$root" rev-list --count "HEAD..@{u}" 2>/dev/null || echo 0)"
    if [[ "${behind:-0}" -gt 0 ]]; then
      vwarn "this checkout is ${behind} commit(s) BEHIND ${up} (HEAD=${head}) — you may be submitting STALE code."
      vwarn "  fix:  git -C $root pull --ff-only"
      vwarn "  if local edits block the pull:  git -C $root fetch && git -C $root reset --hard @{u}"
      [[ "${VIGNOCR_SKIP_GIT_CHECK:-0}" == "1" ]] \
        || vdie "refusing to submit stale code — pull first, or override with VIGNOCR_SKIP_GIT_CHECK=1"
    fi
  fi
  [[ -n "$(git -C "$root" status --porcelain 2>/dev/null)" ]] \
    && vwarn "working tree is DIRTY (HEAD=${head}) — jobs run the on-disk code, which may differ from what was tested/committed."
  return 0
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

# vignocr_ckpt_dir_for_config <cfg_name>: echo the cfg.train.checkpoint.dir value
# the detection trainer uses for a given config (e.g. "detection/rfdetr_medium"
# -> "scratch/detection/rfdetr_medium"). Reads the YAML with PyYAML if available
# else falls back to a stdlib grep. This is the SINGLE knob that decouples
# resume/best discovery from any specific Stage (A, B, or future Stage).
vignocr_ckpt_dir_for_config() {
  local cfg="${1:?need a detection cfg name (e.g. detection/rfdetr_medium)}"
  VIGNOCR_REPO_ROOT="$VIGNOCR_REPO_ROOT" VIGNOCR_CFG="$cfg" python - <<'PY' 2>/dev/null || true
import os, re, sys
root = os.environ["VIGNOCR_REPO_ROOT"]
cfg  = os.environ["VIGNOCR_CFG"]
p = os.path.join(root, "configs", f"{cfg}.yaml")
default = "scratch/detection/" + cfg.split("/")[-1]
try:
    import yaml
    with open(p, encoding="utf-8") as fh:
        c = yaml.safe_load(fh) or {}
    d = ((c.get("train") or {}).get("checkpoint") or {}).get("dir") or default
    print(d)
except Exception:
    # stdlib-only fallback: find the FIRST `dir:` line under `checkpoint:`.
    try:
        with open(p, encoding="utf-8") as fh:
            txt = fh.read()
        m = re.search(r"checkpoint:\s*\n(?:[ \t]+.*\n)*?[ \t]+dir:\s*([^\s#]+)", txt)
        print((m.group(1).strip("\"'") if m else default))
    except Exception:
        print(default)
PY
}

# _vignocr_search_dirs <run_dir> <cfg_name>: emit (one per line) the dirs in which
# RF-DETR may have left a checkpoint for this (run, config) pair.
_vignocr_search_dirs() {
  local run_dir="$1"; local cfg="$2"
  local run_name; run_name="$(basename "$run_dir")"
  local ckpt_rel; ckpt_rel="$(vignocr_ckpt_dir_for_config "$cfg")"
  ckpt_rel="${ckpt_rel:-scratch/detection/${cfg##*/}}"
  # Search both possible scratch roots + the run dir itself. Order matters: the
  # FIRST hit wins, and the explicit scratch tree is preferred over the run dir.
  printf '%s\n' \
    "$VIGNOCR_SCRATCH/$ckpt_rel/$run_name" \
    "$VIGNOCR_SCRATCH_DIR/$ckpt_rel/$run_name" \
    "$run_dir/checkpoints" \
    "$run_dir"
}

# vignocr_find_resume <run_dir> [cfg]: echo a checkpoint to resume RF-DETR from
# (idempotent resubmission). RF-DETR writes `checkpoint.pth` (latest) and
# `checkpoint_best_total.pth` (best) into output_dir; we prefer the latest for
# resuming, falling back to the best, then any newest *.pth/*.pt. The cfg
# argument selects the checkpoint subtree (default: detection/rfdetr_medium for
# back-compat with stage 02).
vignocr_find_resume() {
  local run_dir="${1:?need run_dir}"
  local cfg="${2:-${VIGNOCR_DETECTION_CONFIG:-detection/rfdetr_medium}}"
  local d cand
  while IFS= read -r d; do
    [[ -d "$d" ]] || continue
    for cand in checkpoint.pth checkpoint_last.pth checkpoint_best_total.pth checkpoint_best.pth last.ckpt last.pth; do
      if [[ -f "$d/$cand" ]]; then echo "$d/$cand"; return 0; fi
    done
  done < <(_vignocr_search_dirs "$run_dir" "$cfg")
  # last resort: newest checkpoint-like file across the search dirs
  while IFS= read -r d; do
    [[ -d "$d" ]] || continue
    cand="$(ls -1t "$d"/*.pth "$d"/*.pt "$d"/*.ckpt 2>/dev/null | head -n1 || true)"
    [[ -n "$cand" ]] && { echo "$cand"; return 0; }
  done < <(_vignocr_search_dirs "$run_dir" "$cfg")
  echo ""
}

# vignocr_find_best <run_dir> [cfg]: echo the BEST detector checkpoint produced by
# a finished training run. Same search path as resume but prefers the best-total
# artifact RF-DETR writes.
vignocr_find_best() {
  local run_dir="${1:?need run_dir}"
  local cfg="${2:-${VIGNOCR_DETECTION_CONFIG:-detection/rfdetr_medium}}"
  local d cand
  while IFS= read -r d; do
    [[ -d "$d" ]] || continue
    for cand in checkpoint_best_total.pth checkpoint_best.pth checkpoint_best_ema.pth best.pth best.ckpt; do
      if [[ -f "$d/$cand" ]]; then echo "$d/$cand"; return 0; fi
    done
  done < <(_vignocr_search_dirs "$run_dir" "$cfg")
  while IFS= read -r d; do
    [[ -d "$d" ]] || continue
    cand="$(ls -1t "$d"/*.pth "$d"/*.pt 2>/dev/null | head -n1 || true)"
    [[ -n "$cand" ]] && { echo "$cand"; return 0; }
  done < <(_vignocr_search_dirs "$run_dir" "$cfg")
  echo ""
}

# vignocr_pretrained_weights <cfg>: echo the cached COCO-pretrained .pth that the
# RF-DETR trainer should pass via `pretrain_weights=...`. The trainer code also
# checks this — it's exposed here so the .sbatch can log "using cached weights:
# <path>" up front. Returns empty if nothing is cached (the trainer will then
# raise the offline-init error with a clear next-step).
#
# Probe order MIRRORS train.py::_resolve_pretrained_weights:
#   1. $VIGNOCR_PRETRAINED_RFDETR (explicit override)
#   2. $VIGNOCR_PRETRAINED_DIR/{rfdetr_medium_coco.pth, rf-detr-medium.pth}
#   3. ~/.roboflow/models/rf-detr-medium.pth  (rfdetr 1.x's canonical cache)
vignocr_pretrained_weights() {
  local cfg="${1:-${VIGNOCR_DETECTION_CONFIG:-detection/rfdetr_medium}}"

  # 1) explicit env override
  if [[ -n "${VIGNOCR_PRETRAINED_RFDETR:-}" && -f "${VIGNOCR_PRETRAINED_RFDETR}" ]]; then
    echo "$VIGNOCR_PRETRAINED_RFDETR"; return
  fi
  # 2) our centralized cache
  local d="${VIGNOCR_PRETRAINED_DIR:-$VIGNOCR_PROJECT_DIR/checkpoints/pretrained}"
  for n in rfdetr_medium_coco.pth rf-detr-medium-coco.pth rf-detr-medium.pth; do
    [[ -f "$d/$n" ]] && { echo "$d/$n"; return; }
  done
  # 3) rfdetr's own cache (the canonical 1.x location)
  local rf="${ROBOFLOW_MODELS_DIR:-$HOME/.roboflow/models}"
  for n in rf-detr-medium.pth rf-detr-medium-coco.pth; do
    [[ -f "$rf/$n" ]] && { echo "$rf/$n"; return; }
  done
  echo ""
}

# vignocr_install_postmortem <stage>: register an EXIT trap that, if the job
# fails (any non-zero exit code, including SIGTERM/preempt), copies the run dir's
# train log + last 200 lines into the central logs/slurm/<stage>/<jobid>/ folder
# AND tails it to stderr. After this trap the .err file is self-contained — no
# more "see $VIGNOCR_RUN_DIR/train_*.log" chases on scratch.
#
# IMPORTANT CLOSURE NOTE
#   Bash trap handlers do NOT close over the locals of the enclosing function.
#   When `trap 'fn' EXIT` fires, the outer function's stack frame is GONE — so
#   referencing a local from there with `set -u` active crashes the trap and
#   takes the whole script with it (we hit this: a job that completed
#   successfully exited rc=1 because the trap blew up on its own `$jlog` ref,
#   then SLURM marked the job FAILED and the afterok DAG cancelled everything
#   downstream). The fix: persist EVERY value the trap needs as an EXPORTED
#   environment variable, and reference it inside the trap with `${VAR:-...}`
#   so a missing var degrades gracefully instead of unbound-erroring.
vignocr_install_postmortem() {
  local stage="${1:?need stage}"
  local jlog
  jlog="${VIGNOCR_JOB_LOG_DIR:-$(vignocr_logs_dir "$stage")}"
  # Export EVERYTHING the trap will read — see the closure note above. These
  # are read inside the trap as ${VIGNOCR_JOB_LOG_DIR:-} (defended against
  # set -u even if some future caller unsets them).
  export VIGNOCR_JOB_LOG_DIR="$jlog"
  export VIGNOCR_POSTMORTEM_STAGE="$stage"
  # Stash the resolved scratch run_dir + a back-pointer the user can grep.
  {
    echo "stage:     $stage"
    echo "job:       ${SLURM_JOB_ID:-<local>}"
    echo "node:      $(hostname)"
    echo "run_dir:   ${VIGNOCR_RUN_DIR:-<not-yet-created>}"
    echo "started:   $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  } > "$jlog/BACKPOINTERS.txt"

  trap 'vignocr_exit_trap' EXIT
}

# vignocr_exit_trap: the EXIT trap body. Defined at FILE scope (not nested
# inside vignocr_install_postmortem) so it cannot accidentally capture locals
# from any caller. Every value it needs is read from an exported global with a
# safe default. The function is hardened so that even if a sub-step here
# crashes, the trap itself returns 0 — the original exit code of the script is
# preserved automatically (bash EXIT semantics), and a misbehaving trap can
# NEVER take down a successful job (the bug that caused this rewrite).
vignocr_exit_trap() {
  # Capture the original rc IMMEDIATELY — every command we run below would
  # overwrite $?. We don't want any of those to leak into the script's exit.
  local rc=$?
  # Turn OFF strict mode inside the trap so an unset var or grep-no-match
  # cannot escalate to a script-killing error. The script's own strict mode
  # remains in effect for the original code; this is just the trap scope.
  set +eu
  local jlog="${VIGNOCR_JOB_LOG_DIR:-}"
  local rundir="${VIGNOCR_RUN_DIR:-}"
  # If the trap fired before vignocr_install_postmortem set things up, just
  # exit with the original rc — there's nothing to record.
  if [[ -z "$jlog" || ! -d "$jlog" ]]; then
    return "$rc"
  fi

  local ts
  ts="$(date -u +'%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || echo unknown)"

  if [[ "$rc" -eq 0 ]]; then
    # SUCCESS path: stamp DONE. NEVER let this crash a successful job.
    echo "ok: $ts  rc=0" > "$jlog/DONE" 2>/dev/null || true
    return 0
  fi

  # FAILURE path — best-effort capture. Every command tolerates failure.
  echo "fail: $ts  rc=$rc" > "$jlog/FAILED" 2>/dev/null || true

  if [[ -n "$rundir" && -d "$rundir" ]]; then
    # Copy every *.log in the run dir into the central log dir.
    local f
    for f in "$rundir"/*.log; do
      [[ -f "$f" ]] || continue
      cp -f "$f" "$jlog/" 2>/dev/null || true
    done
    # And tail the most-likely-cause file to stderr so .err contains the actual
    # Python traceback. Prefer the named log; fall back to the newest *.log.
    local primary=""
    for f in train_detection.log train_vignette.log train_detection_ddp.log \
             train_ocr.log autolabel.log eval_export_detection.log eval_ocr.log \
             pipeline_benchmark.log finetune.log validate.log; do
      if [[ -f "$rundir/$f" ]]; then primary="$rundir/$f"; break; fi
    done
    if [[ -z "$primary" ]]; then
      primary="$(ls -1t "$rundir"/*.log 2>/dev/null | head -n1)" || primary=""
    fi
    if [[ -n "$primary" && -f "$primary" ]]; then
      {
        echo
        echo "=============================================================="
        echo " POST-MORTEM: last 200 lines of $primary"
        echo "=============================================================="
        tail -n 200 "$primary" 2>/dev/null || echo "(could not tail)"
        echo "=============================================================="
        echo " Full log was copied to: $jlog/$(basename "$primary")"
        echo "=============================================================="
      } >&2
    fi
  fi
  return "$rc"
}

# vignocr_preamble <stage>: the standard opening every compute job runs.
# Validates env, sets up paths, loads modules, activates venv, seeds, and
# creates+returns a fresh run dir (exported as VIGNOCR_RUN_DIR). Resumption:
# if VIGNOCR_RESUME_RUN_DIR is set, reuse it instead of minting a new one.
# Also installs the post-mortem trap so a job failure leaves a SELF-CONTAINED
# logs/slurm/<stage>/<jobid>/ directory with stderr + the train log copy.
vignocr_preamble() {
  local stage="${1:?vignocr_preamble needs a stage name}"
  vignocr_require_env
  vignocr_assert_account_resolved
  vignocr_paths
  vignocr_install_postmortem "$stage"
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
  # Re-stamp the back-pointer file now that VIGNOCR_RUN_DIR is finalized.
  {
    echo "stage:     $stage"
    echo "job:       ${SLURM_JOB_ID:-<local>}"
    echo "node:      $(hostname)"
    echo "run_dir:   $VIGNOCR_RUN_DIR"
    echo "started:   $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  } > "$VIGNOCR_JOB_LOG_DIR/BACKPOINTERS.txt"
  # Symlink the run_dir <-> log_dir so squeue/sacct users can hop either way.
  ln -sfn "$VIGNOCR_JOB_LOG_DIR" "$VIGNOCR_RUN_DIR/slurm_logs"

  vignocr_snapshot_freeze "$VIGNOCR_RUN_DIR"

  export VIGNOCR_SEED
  VIGNOCR_SEED="$(cat "$VIGNOCR_RUN_DIR/seed.txt" 2>/dev/null || vignocr_resolve_seed)"
  vlog "seed=$VIGNOCR_SEED  data_active=${VIGNOCR_DATA_ACTIVE:-<data.yaml default>}"
  vlog "logs/slurm: $VIGNOCR_JOB_LOG_DIR"
}
