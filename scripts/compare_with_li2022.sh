#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

mkdir -p "${REPO_ROOT}/outputs/tables" "${REPO_ROOT}/outputs/figures" "${REPO_ROOT}/outputs/.matplotlib" "${REPO_ROOT}/outputs/.cache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/outputs/.matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/outputs/.cache}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/compare_with_li2022.py" --config "${REPO_ROOT}/configs/compare_li2022.yaml"
