import sys
import os

celeba_base_path = os.environ.get("CELEBA")
danbooru_base_path = os.environ.get("DANBOORU")

celeba = {
    "data": os.path.join(celeba_base_path, "data"),
    "figs": os.path.join(celeba_base_path, "figs"),
    "samples": os.path.join(celeba_base_path, "samples"),
    "models": os.path.join(celeba_base_path, "model"),
    "logs": os.path.join(celeba_base_path, "logs"),
}

danbooru = {
    "data": os.path.join(danbooru_base_path, "data"),
    "figs": os.path.join(danbooru_base_path, "figs"),
    "samples": os.path.join(danbooru_base_path, "samples"),
    "models": os.path.join(danbooru_base_path, "model"),
    "logs": os.path.join(danbooru_base_path, "logs"),
}
