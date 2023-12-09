import sys
import os


# celeba_base_path = os.environ.get("CELEBA")
# danbooru_base_path = os.environ.get("DANBOORU")

celeba_base_path = "Z:\Egyetem\DL-hf\data\celeba"
danbooru_base_path = "Z:\Egyetem\DL-hf\data\danbooru"

celeba = {
    "data": os.path.join(celeba_base_path, "data"),
    "figs": os.path.join(celeba_base_path, "figs"),
    "samples": os.path.join(celeba_base_path, "samples"),
    "models": os.path.join(celeba_base_path, "models"),
    "logs": os.path.join(celeba_base_path, "logs"),
    "test": os.path.join(celeba_base_path, "test"),
}

danbooru = {
    "data": os.path.join(danbooru_base_path, "data"),
    "figs": os.path.join(danbooru_base_path, "figs"),
    "samples": os.path.join(danbooru_base_path, "samples"),
    "models": os.path.join(danbooru_base_path, "models"),
    "logs": os.path.join(danbooru_base_path, "logs"),
    "test": os.path.join(celeba_base_path, "test"),
}
