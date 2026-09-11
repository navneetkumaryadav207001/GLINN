#!/usr/bin/env bash
set -e

# GLINN: Master All-in-One Execution Script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "==============================================================================================="
echo "  GLINN: General Language Interface for Neural Networks"
echo "  Running Master Reproduction Pipeline..."
echo "==============================================================================================="

python3 reproduce_all.py

echo "==============================================================================================="
echo "  GLINN Reproduction Finished Successfully!"
echo "==============================================================================================="
