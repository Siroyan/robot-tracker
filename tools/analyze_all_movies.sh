#!/usr/bin/env bash
set -euo pipefail

# このスクリプトの場所からリポジトリルートを求め、どこから実行しても同じ動作にする。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MOVIE_DIR="${REPO_ROOT}/movie"
CONFIG_PATH="${1:-${REPO_ROOT}/tracker_config_example.json}"

if [[ ! -d "${MOVIE_DIR}" ]]; then
  echo "movieフォルダが見つかりません: ${MOVIE_DIR}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "設定ファイルが見つかりません: ${CONFIG_PATH}" >&2
  exit 1
fi

shopt -s nullglob
videos=(
  "${MOVIE_DIR}"/*.mp4
  "${MOVIE_DIR}"/*.MP4
  "${MOVIE_DIR}"/*.mov
  "${MOVIE_DIR}"/*.MOV
  "${MOVIE_DIR}"/*.avi
  "${MOVIE_DIR}"/*.AVI
  "${MOVIE_DIR}"/*.mkv
  "${MOVIE_DIR}"/*.MKV
)

if (( ${#videos[@]} == 0 )); then
  echo "movieフォルダに解析対象の動画がありません: ${MOVIE_DIR}"
  exit 0
fi

echo "設定ファイル: ${CONFIG_PATH}"
echo "解析対象: ${#videos[@]} 件"

for video in "${videos[@]}"; do
  echo
  echo "=== 解析開始: $(basename "${video}") ==="
  # main.py側のデフォルトにより、CSVはcsv/、注釈動画はannotation/へ保存される。
  (cd "${REPO_ROOT}" && uv run python main.py "${video}" --config "${CONFIG_PATH}")
done

echo
echo "すべての動画解析が完了しました。"
