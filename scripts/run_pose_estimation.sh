#!/bin/bash
set -e

echo "[*] Installing dependencies with Poetry..."
poetry install

echo "[*] Running pose estimation Python script..."
poetry run python box_pose_estimation/run_pose_estimation.py