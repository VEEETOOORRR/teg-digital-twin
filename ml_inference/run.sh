#!/bin/bash
set -e

mkdir -p models

if [ ! -f "models/teg_model.pt" ]; then
    echo "⚙️ Modelo não encontrado. Gerando dataset e treinando..."
    python generate_dataset.py
    python train.py
fi

echo "🚀 Iniciando o motor de inferência em tempo real..."
python app.py