import os
import torch
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATASET_DIR = "movie_cls_dataset"
YOLO_MODEL  = "yolov8n-cls.pt"

EPOCHS    = 100
BATCH     = 16
IMG_SIZE  = (750, 500)
# ────────────────────────────────────────────────────────────────────────────

# sanity-check
assert os.path.isdir(DATASET_DIR), f"{DATASET_DIR} not found"
assert os.path.isdir(os.path.join(DATASET_DIR, "train")), "train/ missing"
assert os.path.isdir(os.path.join(DATASET_DIR, "val")),   "val/ missing"

# detect classes automatically from train subfolders
train_dir = os.path.join(DATASET_DIR, "train")
classes   = sorted(
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
)

# write data.yaml
yaml_path = os.path.join(DATASET_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {DATASET_DIR}\n")
    f.write("train: train\n")
    f.write("val:   val\n\n")
    f.write(f"nc: {len(classes)}\n")
    f.write("names: " + str(classes) + "\n")

print(f"✅  Wrote {yaml_path} with {len(classes)} classes: {classes}")

# ─── SET UP DEVICE FOR MPS ───────────────────────────────────────────────────
# Enable PyTorch’s MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Choose 'mps' if available, else CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda")
print(f"Using device: {device}")

# ─── LAUNCH TRAINING ────────────────────────────────────────────────────────
model = YOLO(YOLO_MODEL)            # load the pretrained model
model.model.to(device)              # move model weights to MPS/CPU

model.train(
    data=DATASET_DIR,               # dataset folder
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMG_SIZE,
    rect=True,
    device=device,                  # force the device
    name="movie_poster_genre_cls"
)