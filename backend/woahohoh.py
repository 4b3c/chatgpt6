#https://huggingface.co/datasets/openfoodfacts/nutrient-detection-layout
#pip install easyocr pillow

from PIL import Image
import easyocr

img_path = "label.jpg"
reader = easyocr.Reader(['en'])
results = reader.readtext(img_path, detail=1)  # [(bbox, text, conf), ...]

tokens = []
W, H = Image.open(img_path).size
for (x1,y1,x2,y2,x3,y3,x4,y4), text, conf in results:
    x0, y0 = min(x1,x2,x3,x4), min(y1,y2,y3,y4)
    x1_, y1_ = max(x1,x2,x3,x4), max(y1,y2,y3,y4)
    bbox = [int(1000*x0/W), int(1000*y0/H), int(1000*x1_/W), int(1000*y1_/H)]  # LayoutLM-style 0–1000
    tokens.append({"text": text, "bbox": bbox, "conf": conf})
