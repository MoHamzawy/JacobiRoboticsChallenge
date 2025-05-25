#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

echo "[*] Installing dependencies with Poetry..."
cd "$PROJECT_ROOT/box_segmentation"
poetry install

echo "[*] Running segmentation training script..."
poetry run python segmentation/train_segmentation.py