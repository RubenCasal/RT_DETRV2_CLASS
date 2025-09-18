#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
: "${TORCH_CU_INDEX:=https://download.pytorch.org/whl/cu121}"
: "${INSTALL_D2_DEPS:=0}"     # 1 = instala fvcore/iopath/yacs (útil si usas detectron2)
: "${INSTALL_DETECTRON2:=0}"  # 1 = instala detectron2 al final
: "${DETECTRON2_REF:=a1ce2f956a1d2212ad672e3c47d53405c2fe4312}"

echo ">> Python:"
python3 --version

# Tooling moderno para evitar errores de metadatos/build
python3 -m pip install -U "pip>=24.2" "setuptools>=75.0" "wheel>=0.44" "packaging>=24.1"

# ========= PyTorch primero (CUDA 12.1) =========
echo ">> Instalando PyTorch (CUDA 12.1) ..."
python3 -m pip install --index-url "${TORCH_CU_INDEX}" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# ========= Núcleo científico + OpenCV =========
# OpenCV 4.12.0.88 requiere NumPy >=2.0 y <2.3 en py>=3.9 -> fijamos NumPy 2.2.2
echo ">> Instalando NumPy 2.x y OpenCV ..."
python3 -m pip install --no-cache-dir \
  numpy==2.2.2 \
  opencv-contrib-python==4.12.0.88

# ========= Resto del stack de entrenamiento =========
echo ">> Instalando resto de paquetes ..."
python3 -m pip install --no-cache-dir \
  scipy==1.15.3 \
  matplotlib==3.10.6 \
  absl-py==2.3.1 \
  accelerate==1.10.1 \
  transformers==4.56.1 \
  ultralytics==8.3.198 \
  wandb==0.21.4 \
  pycocotools==2.0.10 \
  supervision==0.26.1

# ========= (Opcional) Dependencias base para Detectron2 =========
if [[ "${INSTALL_D2_DEPS}" == "1" ]]; then
  echo ">> Instalando fvcore/iopath/yacs (con toolchain del sistema) ..."
  DEBIAN_FRONTEND=noninteractive sudo apt-get update
  DEBIAN_FRONTEND=noninteractive sudo apt-get install -y \
    build-essential cmake ninja-build libgl1 libglib2.0-0 git
  sudo rm -rf /var/lib/apt/lists/* || true

  # Evita problemas de build con aislamiento
  python3 -m pip install --no-build-isolation --no-cache-dir \
    fvcore==0.1.5.post20221221 iopath==0.1.9 yacs==0.1.8
fi

# ========= (Opcional) Detectron2 (solo si lo necesitas) =========
if [[ "${INSTALL_DETECTRON2}" == "1" ]]; then
  echo ">> Instalando detectron2 @ ${DETECTRON2_REF} ..."
  # Requiere torch ya instalado (lo está). Usa no-build-isolation para respetar wheels instaladas
  python3 -m pip install --no-build-isolation --no-cache-dir \
    "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_REF}"
fi

# ========= Verificación =========
echo ">> Verificación de versiones:"
python3 - <<'PY'
import torch, cv2, transformers, accelerate, ultralytics, wandb, numpy
print("python  :", __import__("sys").version.split()[0])
print("torch   :", torch.__version__, "cuda:", torch.version.cuda)
print("numpy   :", numpy.__version__)
print("cv2     :", cv2.__version__)
print("transformers:", transformers.__version__, "| accelerate:", accelerate.__version__)
print("ultralytics:", ultralytics.__version__)
print("wandb   :", wandb.__version__)
PY

echo ">> Instalación completada correctamente."
