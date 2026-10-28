#!/bin/bash
# 启动本地服务。会自动挑一个「装了 fastapi + sherpa_onnx」的 Python。
#
# 为什么需要这个脚本：`python src/app.py` 用的是系统 Python，那里没有 sherpa_onnx，
# 报的错是 `ModuleNotFoundError: No module named 'fastapi'`，看起来像项目缺依赖，
# 实际是解释器选错了。这个脚本按顺序试几个候选，跑通第一个就用它。
#
#   bash scripts/run.sh           启动
#   bash scripts/run.sh --check   只检查环境，不启动
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

pick_python() {
  local cands=()
  [ -n "${VP_PY:-}" ] && cands+=("$VP_PY")
  cands+=(
    "$ROOT/.venv/bin/python"
    "$ROOT/../.vpvenv/bin/python"
    "$HOME/.workbuddy/binaries/python/envs/voiceprint/bin/python"
  )
  command -v python3 >/dev/null 2>&1 && cands+=("$(command -v python3)")
  local p
  for p in "${cands[@]}"; do
    [ -n "$p" ] && [ -x "$p" ] || continue
    if "$p" - <<'PY' >/dev/null 2>&1
import fastapi, uvicorn, sherpa_onnx, soundfile, numpy
PY
    then
      echo "$p"
      return 0
    fi
  done
  return 1
}

PY="$(pick_python)" || {
  echo "找不到装了依赖的 Python。任选一种装法："
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt fastapi 'uvicorn[standard]'"
  echo "或者显式指定：VP_PY=/path/to/venv/bin/python bash scripts/run.sh"
  exit 1
}

if [ "${1:-}" = "--check" ]; then
  echo "Python: $PY"
  "$PY" - <<PY
import sys
sys.path.insert(0, "$ROOT/src")
import config
print("模型  :", config.MODEL_DIR)
print("模型在:", __import__("os").path.exists(config.EMB_MODEL))
print("声纹库:", config.DB_DIR)
print("录音目:", config.RECORDING_DIR)
print("状态  :", config.STATE)
try:
    import protolib
    print()
    print(protolib.status())
except Exception as e:
    print("声纹库读取失败:", e)
PY
  exit 0
fi

echo "Python: $PY"
cd "$ROOT/src" && exec "$PY" app.py
