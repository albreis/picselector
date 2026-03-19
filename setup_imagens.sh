#!/bin/bash

set -e

echo "🚀 Setup ambiente de processamento de imagens..."

# =========================
# CONFIG
# =========================
ENV_DIR="$HOME/env-imagens"
PYTHON_BIN="python3"

# =========================
# DEPENDÊNCIAS DO SISTEMA
# =========================
echo "📦 Instalando dependências do sistema..."
sudo apt update
sudo apt install -y python3-venv python3-full libimage-exiftool-perl

# =========================
# CRIAR VENV (se não existir)
# =========================
if [ ! -d "$ENV_DIR" ]; then
    echo "🧠 Criando ambiente virtual em $ENV_DIR..."
    $PYTHON_BIN -m venv "$ENV_DIR"
else
    echo "✅ Ambiente virtual já existe"
fi

# =========================
# ATIVAR VENV
# =========================
echo "⚡ Ativando ambiente virtual..."
source "$ENV_DIR/bin/activate"

# =========================
# ATUALIZAR PIP
# =========================
echo "⬆️ Atualizando pip..."
pip install --upgrade pip

# =========================
# INSTALAR LIBS
# =========================
echo "📚 Instalando bibliotecas Python..."
pip install opencv-python scikit-learn pillow numpy tqdm transformers torch

# =========================
# FINAL
# =========================
echo ""
echo "✅ Ambiente pronto!"
echo ""
echo "👉 Para ativar depois use:"
echo "source $ENV_DIR/bin/activate"
echo ""
echo "👉 Para rodar seu script:"
echo "python tag_images.py /caminho/da/pasta"
echo ""