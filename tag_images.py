#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import tempfile
import subprocess
from tqdm import tqdm

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".webp")
RESIZE_WIDTH = 480
MIN_COLOR_RATIO = 0.002

HUE_COLOR_RANGES = [
    (0, 5, "cor/vermelho"),
    (5, 13, "cor/laranja"),
    (13, 21, "cor/amarelo"),
    (21, 35, "cor/lima"),
    (35, 50, "cor/verde"),
    (50, 71, "cor/turquesa"),
    (71, 96, "cor/ciano"),
    (96, 116, "cor/azul-claro"),
    (116, 136, "cor/azul"),
    (136, 151, "cor/roxo"),
    (151, 166, "cor/magenta"),
    (166, 175, "cor/rosa"),
    (175, 180, "cor/vermelho"),
]

def detectar_cores(path, min_ratio=MIN_COLOR_RATIO):
    img = cv2.imread(path)
    if img is None:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))
    total_pixels = len(pixels)

    if total_pixels == 0:
        return []

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]

    tags = set()

    ratio_preto = np.mean(v < 45)
    ratio_branco = np.mean((s < 30) & (v > 220))
    ratio_cinza = np.mean((s < 30) & (v >= 45) & (v <= 220))

    if ratio_preto >= min_ratio:
        tags.add("cor/preto")
    if ratio_branco >= min_ratio:
        tags.add("cor/branco")
    if ratio_cinza >= min_ratio:
        tags.add("cor/cinza")

    mask_colorido = (s >= 35) & (v >= 35)
    if not np.any(mask_colorido):
        return list(tags)

    h_color = h[mask_colorido]

    for start, end, tag in HUE_COLOR_RANGES:
        ratio_cor = np.mean((h_color >= start) & (h_color < end))
        if ratio_cor >= min_ratio:
            tags.add(tag)

    return list(tags)

def detectar_luminosidade(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)

    if mean > 180:
        return ["luz/claro"]
    elif mean < 70:
        return ["luz/escuro"]
    else:
        return ["luz/medio"]

def salvar_tags(file_path, tags):
    cmd = ["exiftool", "-overwrite_original"]

    for tag in tags:
        cmd.append(f'-XMP:Subject+={tag}')

    cmd.append(file_path)

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def criar_resize_temporario(path, target_width=RESIZE_WIDTH):
    img = cv2.imread(path)
    if img is None:
        return None

    h, w = img.shape[:2]
    if w <= target_width:
        return None

    new_w = target_width
    new_h = max(1, int(h * (target_width / w)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    _, ext = os.path.splitext(path)
    if not ext:
        ext = ".jpg"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
        temp_path = temp_file.name

    if not cv2.imwrite(temp_path, resized):
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return None

    return temp_path

def processar_imagem(path):
    temp_path = criar_resize_temporario(path)
    path_processamento = temp_path if temp_path else path

    try:
        tags = set(detectar_cores(path_processamento))

        for l in detectar_luminosidade(path_processamento):
            tags.add(l)

        return list(tags)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

def main():
    if len(sys.argv) < 2:
        print("Uso: python tag_images.py /pasta")
        sys.exit(1)

    pasta = sys.argv[1]

    if not os.path.isdir(pasta):
        print("Pasta inválida")
        sys.exit(1)

    arquivos = []

    for root, dirs, files in os.walk(pasta):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXT):
                arquivos.append(os.path.join(root, file))

    arquivos.sort()

    total = len(arquivos)

    print(f"📂 Encontradas {total} imagens\n")

    for path in tqdm(arquivos, desc="Processando imagens", unit="img"):
        try:
            tags = processar_imagem(path)
            salvar_tags(path, tags)
            tqdm.write(f"[OK] {path}")
        except Exception as e:
            print(f"\n[ERRO] {path}: {e}")

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()