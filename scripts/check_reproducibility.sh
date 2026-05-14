#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/reproduce_fig5.sh
bash scripts/reproduce_fig7.sh
bash scripts/compare_with_li2022.sh

expected_outputs=(
  outputs/fig5/fig5.png
  outputs/fig7/fig7.png
  outputs/figures/li2022_slice_comparison.png
  outputs/figures/li2022_depthwise_error.png
)

for output in "${expected_outputs[@]}"; do
  if [[ ! -s "${output}" ]]; then
    echo "Missing expected output: ${output}" >&2
    exit 1
  fi
done

echo "All expected outputs were created."
