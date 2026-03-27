import os
import cv2
import numpy as np
import warnings
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd

warnings.filterwarnings("ignore")


def load_images(file_path):
    if file_path.lower().endswith(".pdf"):
        return convert_from_path(file_path)
    else:
        return [Image.open(file_path).convert("RGB")]


class PaddleOCREngine:
    def __init__(self, use_gpu=True):
        print(f"Loading PaddleOCR (GPU: {use_gpu})...")

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=use_gpu,
            ocr_version='PP-OCRv4',
            show_log=False
        )

        print("✅ PaddleOCR Model loaded successfully")

    def extract_text_with_confidence(self, file_path: str) -> list:
        print(f"Running OCR on: {file_path}")

        images = load_images(file_path)
        all_results = []

        for page_num, img in enumerate(images):
            result = self.ocr.ocr(np.array(img), cls=True)

            if not result or not result[0]:
                continue

            for line in result[0]:
                bbox, (text, confidence) = line

                all_results.append({
                    "page": page_num + 1,
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "bbox": bbox
                })
        #print(all_results)
        print(all_results)
        df=pd.DataFrame(all_results)
        df.to_csv('output.csv', index=False)

        return all_results


#this is my ocr_engine.py