#!/usr/bin/env bash
# Downloads MLPerf Training's pretokenized C4 for Llama 3.1 8B and GPT-OSS 20B
# (about 85 GB) into DIR and checks every file against MLCommons' md5 list.
#
#   scripts/mlperf_download_c4.sh DIR
#
# The files are those of MLCommons' R2 downloader for
# https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri,
# fetched with curl, which resumes a file whose transfer broke off. Rerunning
# continues where the last run stopped and checks the md5 sums again.
set -euo pipefail

dir=${1:?usage: $0 DIR}
meta=https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset
base=https://training.mlcommons-storage.org/llama3_1/datasets/c4/llama3_1_8b
sums=llama-3-1-8b-preprocessed-c4-dataset.md5

mkdir -p "$dir"
cd "$dir"
curl -sSf --retry 5 -o "$sums" "$meta.md5"
for file in $(sed 's/^[a-f0-9]\{32\}[[:space:]]*//; s,^\./,,' "$sums"); do
  for attempt in 1 2 3 4 5; do
    if curl -sSf -C - --retry 5 --retry-connrefused -o "$file" "$base/$file"; then
      break
    fi
    echo "retrying $file ($attempt)" >&2
    sleep 5
  done
  echo "$file: $(stat -c %s "$file") bytes"
done
md5sum -c "$sums"
