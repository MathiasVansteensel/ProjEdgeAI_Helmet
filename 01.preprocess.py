import os
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import shutil
import math

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
DATASET_PATH = "hard-hat-detection"
OUTPUT_PATH = "yolo_dataset"
os.makedirs(OUTPUT_PATH, exist_ok=True)

CLASSES = ["person_with_helmet", "person_without_helmet", "helmet", "head"]

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

def voc_to_yolo(xmin, ymin, xmax, ymax, w, h):
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    bw = xmax - xmin
    bh = ymax - ymin
    return cx / w, cy / h, bw / w, bh / h

# ---------------------------------------------------------
# PREP OUTPUT DIR STRUCTURE
# ---------------------------------------------------------
for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT_PATH}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/labels/{split}", exist_ok=True)

images_dir = Path(DATASET_PATH) / "images"
ann_dir = Path(DATASET_PATH) / "annotations"
xml_files = list(ann_dir.glob("*.xml"))

# 80 percent train
random.shuffle(xml_files)
train_files = xml_files[: int(0.8 * len(xml_files))]
val_files = xml_files[int(0.8 * len(xml_files)) :]

# ---------------------------------------------------------
# PROCESS FILES
# ---------------------------------------------------------
def process_file(xml_path, split):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_name = root.find("filename").text
    img_path = images_dir / img_name
    out_img = f"{OUTPUT_PATH}/images/{split}/{img_name}"
    shutil.copy(img_path, out_img)

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    heads = []
    helmets = []

    # first parse all objects
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bb = obj.find("bndbox")

        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        if name == "head":
            heads.append((xmin, ymin, xmax, ymax))
        elif name == "helmet":
            helmets.append((xmin, ymin, xmax, ymax))

    # now write labels
    label_path = f"{OUTPUT_PATH}/labels/{split}/{img_name.rsplit('.', 1)[0]}.txt"
    with open(label_path, "w") as f:
        # helmets
        for xmin, ymin, xmax, ymax in helmets:
            cx, cy, bw, bh = voc_to_yolo(xmin, ymin, xmax, ymax, w, h)
            f.write(f"{CLASSES.index('helmet')} {cx} {cy} {bw} {bh}\n")

        # derived head based class
        for hxmin, hymin, hxmax, hymax in heads:
            head_box = (hxmin, hymin, hxmax, hymax)

            has_helmet = False
            for hel_box in helmets:
                if bbox_iou(head_box, hel_box) >= 0.3:
                    has_helmet = True
                    break

            clsname = "person_with_helmet" if has_helmet else "person_without_helmet"
            cx, cy, bw, bh = voc_to_yolo(hxmin, hymin, hxmax, hymax, w, h)
            f.write(f"{CLASSES.index(clsname)} {cx} {cy} {bw} {bh}\n")

# run conversion
for f in train_files:
    process_file(f, "train")

for f in val_files:
    process_file(f, "val")

# write YAML
yaml = f"""
path: {OUTPUT_PATH}
train: images/train
val: images/val

names:
  0: person_with_helmet
  1: person_without_helmet
  2: helmet
  3: head
"""

with open(f"{OUTPUT_PATH}/data.yaml", "w") as f:
    f.write(yaml)

print("Preprocessing done.")