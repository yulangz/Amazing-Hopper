#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

device=${1:-0}
repeats=${2:-10000}
steps=${3:-32}
make -s >&2
# ctas=0 asks the executable for one resident CTA per SM (cluster-aligned).
for gemm in 0 1; do
  for cluster in 1 2; do
    for drain in 0 1; do
      for invalidate in 0 1; do
        timeout 120s build/repro "$cluster" 0 "$steps" "$repeats" \
          "$drain" "$invalidate" "$device" "$gemm"
      done
    done
  done
done
