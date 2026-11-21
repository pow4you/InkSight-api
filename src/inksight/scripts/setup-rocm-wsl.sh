#!/usr/bin/env bash
# Prepare a WSL distro to run InkSight on an AMD GPU (ROCm), for WslRunner /
# runner="rocm".  Run inside an Ubuntu 24.04 WSL distro, as root:
#
#   wsl --install -d Ubuntu-24.04 --no-launch
#   python -m inksight build --rocm            (from Windows; runs this script)
#   wsl -d Ubuntu-24.04 -u root -- bash src/inksight/scripts/setup-rocm-wsl.sh
#
# Needs a Radeon RX 7900 / 9000 series (or matching Radeon PRO) card and AMD
# Software: Adrenalin Edition with WSL support on Windows (AMD lists 26.1.1+
# for ROCm 7.2).  Downloads several GB.  Re-running is safe.
#
# Lessons baked in:
#  * Only AMD's tensorflow-rocm wheel may be installed.  `tensorflow-text`
#    depends on the stock `tensorflow` package, which pip then installs next to
#    it and which hides the GPU - so tensorflow-text is NOT installed; inksight
#    uses its sentencepiece tokenizer route instead.
#  * inksight drops the notebook's --xla_gpu_autotune_level=0 when it sees
#    tensorflow-rocm (without autotuning, MIOpen gets no convolution workspace).
set -euo pipefail

ROCM_VERSION=7.2
AMDGPU_DEB=amdgpu-install_7.2.70200-1_all.deb
TF_WHEEL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/tensorflow_rocm-2.20.0.dev0%2Bselfbuilt-cp312-cp312-manylinux_2_28_x86_64.whl"
VENV=/opt/inksight-rocm

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget python3-venv python3-pip >/dev/null

if ! command -v rocminfo >/dev/null; then
    echo "== installing ROCm ${ROCM_VERSION} for WSL"
    tmp=$(mktemp -d)
    wget -q -O "$tmp/$AMDGPU_DEB" "https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/${AMDGPU_DEB}"
    apt-get install -y -qq "$tmp/$AMDGPU_DEB" >/dev/null
    amdgpu-install -y --usecase=wsl,rocm --no-dkms
    rm -rf "$tmp"
fi
echo "== GPUs seen by ROCm"
rocminfo | grep -E "Marketing Name" || { echo "ROCm sees no GPU - check the Windows driver"; exit 1; }

echo "== Python environment in ${VENV}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" uninstall -q -y tensorflow tensorflow-text tensorflow-rocm 2>/dev/null || true
"$VENV/bin/pip" install -q "$TF_WHEEL"
"$VENV/bin/pip" uninstall -q -y tensorflow 2>/dev/null || true      # never next to the ROCm build
"$VENV/bin/pip" install -q "sentencepiece>=0.2.1" numpy Pillow

"$VENV/bin/python" - <<'EOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow", tf.__version__, "GPUs:", gpus)
raise SystemExit(0 if gpus else "TensorFlow sees no GPU")
EOF
echo "== ready: InkSightPage(runner=\"rocm\")  (set INKSIGHT_WSL_DISTRO if this is not your default distro)"
