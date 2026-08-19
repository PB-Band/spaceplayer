"""Manages the shuffled rotation of background images, independent of the audio track list."""

import random
from pathlib import Path

from PIL import Image

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def find_image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


class BackgroundImageManager:
    """Infinite reshuffling rotation of background images."""

    def __init__(self):
        self.files = []
        self._iterator = None

    def set_folder(self, folder):
        files = find_image_files(folder)
        self.files = files
        self._iterator = self._infinite_shuffled_paths() if files else None

    def has_images(self):
        return bool(self.files)

    def _infinite_shuffled_paths(self):
        while True:
            order = self.files.copy()
            random.shuffle(order)
            for f in order:
                yield f

    def next_image(self):
        """Return a freshly loaded RGB PIL Image for the next image in rotation, or None."""
        if self._iterator is None:
            return None
        for _ in range(len(self.files) + 1):
            path = next(self._iterator)
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                continue
        return None
