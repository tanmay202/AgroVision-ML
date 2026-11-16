#!/usr/bin/env bash
# AgroVision-ML — Launch Staging Server (POSIX)
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================================="
echo "  Starting AgroVision-ML Universal Staging Environment"
echo "=========================================================="

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/member2-price-volatility/src"
export AGROVISION_ENV="staging"
export AGROVISION_ENABLE_SHADOW_EVALUATION="true"
export AGROVISION_ENABLE_RICE_SHADOW="true"
export AGROVISION_ENABLE_WHEAT_SHADOW="true"
export AGROVISION_ENABLE_POTATO_SHADOW="true"
export AGROVISION_ENABLE_TEA_SHADOW="true"

echo "Loaded staging configuration."
echo "Launching Uvicorn ASGI server on http://0.0.0.0:8000 ..."

exec python -m uvicorn api_v2:app --app-dir "${PROJECT_ROOT}/member2-price-volatility/src" --host 0.0.0.0 --port 8000 --reload
