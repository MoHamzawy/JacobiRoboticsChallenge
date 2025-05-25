#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

echo "[*] Installing dependencies with Poetry..."
cd "$PROJECT_ROOT/box_pose_estimation"
poetry install

echo "[*] Running pose estimation Python script..."
poetry run python estimation/run_pose_estimation.py