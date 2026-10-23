#!/usr/bin/env bash
# Download the METABRIC breast cancer dataset from Kaggle into data/.
# Requires a Kaggle API token (~/.kaggle/kaggle.json or KAGGLE_USERNAME /
# KAGGLE_KEY); see https://www.kaggle.com/docs/api#authentication
set -euo pipefail

DATASET="raghadalharbi/breast-cancer-gene-expression-profiles-metabric"
DATA_DIR="data"

mkdir -p "$DATA_DIR"
kaggle datasets download "$DATASET" -p "$DATA_DIR" --unzip

if [ ! -f "$DATA_DIR/METABRIC_RNA_Mutation.csv" ]; then
    echo "Download finished but METABRIC_RNA_Mutation.csv is missing" >&2
    exit 1
fi
