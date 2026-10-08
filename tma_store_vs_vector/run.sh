#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
device=${1:-0}
samples=${2:-40}
make -j2
results_dir=$(mktemp -d build/results.XXXXXX)
printf 'Results: %s\n' "$results_dir"

for round in 0 1; do
  if [ "$round" = 0 ]; then
    store_configs='64:1 64:2 64:4 64:8 64:16 64:32 64:64 4096:1 4096:4'
    ring_configs='lane0:32:1 elect:32:1 elect:64:1 elect:64:0 lane0:32:0'
    operations='copy gemm'
    row_counts='49152 122880'
  else
    store_configs='4096:4 4096:1 64:64 64:32 64:16 64:8 64:4 64:2 64:1'
    ring_configs='lane0:32:0 elect:64:0 elect:64:1 elect:32:1 lane0:32:1'
    operations='gemm copy'
    row_counts='122880 49152'
  fi
  for config in $store_configs; do
    build/store_repro 64 "${config%:*}" 512 "${config#*:}" "$samples" "$device" \
      >> "$results_dir/store_r${round}.log"
  done
  for config in $ring_configs; do
    IFS=: read -r issuer tile_rows fence <<< "$config"
    for op in $operations; do
      for rows in $row_counts; do
        build/ring_stage_repro "$op" "$rows" "$issuer" "$tile_rows" "$fence" \
          "$samples" "$device" >> "$results_dir/ring_r${round}.log"
      done
    done
  done
done
