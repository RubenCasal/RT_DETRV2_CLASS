from pathlib import Path
import random, shutil, json
from PIL import Image

# === CONSTANTES ===
SRC = Path("/home/rcasal/Desktop/projects/dataset fuerteventura/datasets_internet/full_dataset").resolve()
DST= Path("/home/rcasal/Desktop/projects/PtFue/detvr2_training/dataset_prueba").resolve()
N_TRAIN, N_VAL, SEED = 100, 20, 42



CATS = [{"id":0,"name":"person"},{"id":1,"name":"vehicles"},{"id":2,"name":"boat"}]
EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
# ==================

(DST/"images/train").mkdir(parents=True, exist_ok=True)
(DST/"images/val").mkdir(parents=True, exist_ok=True)
(DST/"labels/train").mkdir(parents=True, exist_ok=True)
(DST/"labels/val").mkdir(parents=True, exist_ok=True)
(DST/"annotations").mkdir(parents=True, exist_ok=True)

imgs = sorted([p for p in (SRC/"images").iterdir() if p.suffix.lower() in EXTS and p.is_file()])
pairs = [(im, (SRC/"labels"/(im.stem + ".txt"))) for im in imgs if (SRC/"labels"/(im.stem + ".txt")).exists()]

random.seed(SEED)
k = min(N_TRAIN + N_VAL, len(pairs))
sel = random.sample(pairs, k=k)
train = sel[:min(N_TRAIN, len(sel))]
val   = sel[min(N_TRAIN, len(sel)):]

def copy_pairs(pairs, dimg, dlbl):
    for im, lb in pairs:
        shutil.copy2(im, dimg/im.name)
        shutil.copy2(lb, dlbl/(im.stem + ".txt"))

copy_pairs(train, DST/"images/train", DST/"labels/train")
copy_pairs(val,   DST/"images/val",   DST/"labels/val")

def yolo2coco(split_pairs, dimg, dlbl, out_json, cats=CATS):
    ims, anns, img_id, ann_id = [], [], 1, 1
    for im_src, _ in split_pairs:
        im = dimg/im_src.name; lb = dlbl/(im_src.stem + ".txt")
        W,H = Image.open(im).size
        ims.append({"id":img_id,"file_name":im.name,"width":W,"height":H})
        for ln in lb.read_text().splitlines():
            c,cx,cy,bw,bh = ln.split()
            cid=int(float(c)); cx,cy,bw,bh=map(float,(cx,cy,bw,bh))
            x=(cx-bw/2)*W; y=(cy-bh/2)*H; w=bw*W; h=bh*H
            x=max(0,min(x,W-1)); y=max(0,min(y,H-1)); w=max(0,min(w,W-x)); h=max(0,min(h,H-y))
            anns.append({"id":ann_id,"image_id":img_id,"category_id":cid,
                         "bbox":[x,y,w,h],"area":float(w*h),"iscrowd":0})
            ann_id+=1
        img_id+=1
    (DST/"annotations"/out_json).write_text(json.dumps({"images":ims,"annotations":anns,"categories":cats}))

yolo2coco(train, DST/"images/train", DST/"labels/train", "train.json")
yolo2coco(val,   DST/"images/val",   DST/"labels/val",   "val.json")
print(f"[OK] {DST}")
