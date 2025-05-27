#!/bin/bash
set -e

echo "🔧 Creating Python virtual environment..."
python3.11 -m venv .venv

# Determine activation script path
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    ACTIVATE_SCRIPT=".venv/Scripts/activate"
else
    ACTIVATE_SCRIPT=".venv/bin/activate"
fi

# shellcheck source=/dev/null
source "$ACTIVATE_SCRIPT"
echo "📟 Activated environment: $ACTIVATE_SCRIPT"

echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

echo "📦 Installing build tools..."
pip install wheel setuptools cython

echo "📦 Installing PyTorch with CUDA 11.8..."
pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118

echo "🧪 Testing torch import in subprocess..."
python -c "import torch; print('Torch OK ✅', torch.__version__)"

echo "📦 Installing all other dependencies..."
pip install -r requirements.txt

echo "📦 Installing Detectron2 from local path..."
pip install ./detectron2 --no-build-isolation --no-use-pep517

echo "✅ Setup complete!"
echo "👉 To activate the environment, run:"
echo "   source $ACTIVATE_SCRIPT"
