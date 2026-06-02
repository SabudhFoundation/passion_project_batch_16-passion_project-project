#!/usr/bin/env sh
set -eu

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

VENV_PATH="${VENV_PATH:-.venv}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"
TEMP_REQUIREMENTS="$(mktemp)"

cleanup() {
    rm -f "${TEMP_REQUIREMENTS}"
}

trap cleanup EXIT

run_apt() {
    if command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        "$@"
    fi
}

if [ "${INSTALL_SYSTEM_PACKAGES}" = "1" ] && (
    ! command -v fontforge >/dev/null 2>&1 || ! command -v tesseract >/dev/null 2>&1
); then
    run_apt apt-get update
    run_apt apt-get install -y \
        fontforge \
        python3-fontforge \
        tesseract-ocr \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3-pip \
        python3-venv
fi

rm -rf "${VENV_PATH}"
python3 -m venv "${VENV_PATH}"
. "${VENV_PATH}/bin/activate"

pip install --upgrade pip
grep -vE '^(easyocr|torch|torchvision)==' require.txt > "${TEMP_REQUIREMENTS}"
pip install -r "${TEMP_REQUIREMENTS}"
pip install easyocr==1.7.2 --no-deps
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
