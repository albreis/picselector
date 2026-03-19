#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import tempfile
import subprocess
from tqdm import tqdm

HF_CACHE_DIR = os.path.join(tempfile.gettempdir(), "picselector_hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE_DIR, "transformers"))

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

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

CLOTHING_LABELS = {
    "t-shirt": "roupa/camiseta",
    "shirt": "roupa/camisa",
    "blouse": "roupa/blusa",
    "jacket": "roupa/jaqueta",
    "coat": "roupa/casaco",
    "hoodie": "roupa/moletom",
    "blazer": "roupa/blazer",
    "dress": "roupa/vestido",
    "skirt": "roupa/saia",
    "jeans": "roupa/calca-jeans",
    "pants": "roupa/calca",
    "shorts": "roupa/shorts",
    "sneakers": "roupa/tenis",
    "shoes": "roupa/sapato",
    "boots": "roupa/bota",
    "hat": "roupa/chapeu",
    "cap": "roupa/bone",
}

_clothing_classifier = None
_clothing_detector_disabled = False


def get_clothing_classifier():
    global _clothing_classifier
    global _clothing_detector_disabled

    if _clothing_detector_disabled:
        return None

    if _clothing_classifier is not None:
        return _clothing_classifier

    if pipeline is None or Image is None:
        _clothing_detector_disabled = True
        tqdm.write("[AVISO] Detecção de roupas desativada (faltam libs: transformers e/ou pillow).")
        return None

    try:
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
        _clothing_classifier = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            cache_dir=HF_CACHE_DIR,
        )
        return _clothing_classifier
    except Exception as e:
        _clothing_detector_disabled = True
        tqdm.write(f"[AVISO] Falha ao iniciar detector de roupas: {e}")
        return None


def detectar_roupas(path, threshold=0.20, max_tags=4):
    classifier = get_clothing_classifier()
    if classifier is None:
        return []

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            resultados = classifier(
                img,
                candidate_labels=list(CLOTHING_LABELS.keys()),
            )

        tags = []
        for item in resultados:
            label = item.get("label")
            score = item.get("score", 0.0)

            if label in CLOTHING_LABELS and score >= threshold:
                tags.append(CLOTHING_LABELS[label])

            if len(tags) >= max_tags:
                break

        return tags
    except Exception as e:
        tqdm.write(f"[AVISO] Erro ao detectar roupas em {path}: {e}")
        return []

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

        for r in detectar_roupas(path_processamento):
            tags.add(r)

        for l in detectar_luminosidade(path_processamento):
            tags.add(l)

        return list(tags)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def filtrar_tags_por_prefixo(tags, prefixo):
    return sorted([tag for tag in tags if tag.startswith(prefixo)])

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
            cores = filtrar_tags_por_prefixo(tags, "cor/")
            roupas = filtrar_tags_por_prefixo(tags, "roupa/")

            cores_txt = ", ".join(cores) if cores else "nenhuma"
            roupas_txt = ", ".join(roupas) if roupas else "nenhuma"

            tqdm.write(f"[OK] {path}")
            tqdm.write(f"[TAGS] cores: {cores_txt} | roupas: {roupas_txt}")
        except Exception as e:
            print(f"\n[ERRO] {path}: {e}")

    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()