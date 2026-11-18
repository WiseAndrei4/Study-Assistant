from PIL import Image
from pix2tex.cli import LatexOCR


class OCREngine:
    def __init__(self):
        self.model = LatexOCR()

    def process(self, image_path: str) -> str:
        img = Image.open(image_path)
        return self.model(img)