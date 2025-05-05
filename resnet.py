import os
import time
import copy
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR      = "/Users/abhiramkamini/Downloads/nlp/data/movie_cls_dataset"
WEIGHTS_PATH  = "/Users/abhiramkamini/Downloads/nlp/resnet50-0676ba61.pth"
BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-3
# DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps")

# ─── LOGGING SETUP ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# ────────────────────────────────────────────────────────────────────────────

def prepare_data():
    tfms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
    }
    datasets_dict = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), tfms[x])
        for x in ("train", "val")
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            datasets_dict[x],
            batch_size=BATCH_SIZE,
            shuffle=(x=="train"),
            num_workers=4,
            pin_memory=True
        )
        for x in ("train", "val")
    }
    sizes = {x: len(datasets_dict[x]) for x in ("train", "val")}
    classes = datasets_dict["train"].classes
    logger.info(f"Found {len(classes)} classes: {classes}")
    logger.info(f"Dataset sizes — train: {sizes['train']}, val: {sizes['val']}")
    return dataloaders, sizes, classes

def build_model(num_classes):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    # load ImageNet weights, skipping the fc layer
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    return model

def train_model(model, dataloaders, sizes):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS+1):
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} starting")
        for phase in ["train", "val"]:
            is_train = (phase == "train")
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0

            dataloader = dataloaders[phase]
            desc = f"{phase.capitalize()} Epoch {epoch}"
            for inputs, labels in tqdm(dataloader, desc=desc, unit="batch"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if is_train:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if is_train:
                scheduler.step()

            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects / sizes[phase]
            logger.info(f"{phase.capitalize()} — Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            # deep copy the best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                logger.info(f"New best validation accuracy: {best_acc:.4f}")

        logger.info(f"Epoch {epoch} completed\n")

    total_time = time.time() - start_time
    mins, secs = divmod(total_time, 60)
    logger.info(f"Training complete in {int(mins)}m {int(secs)}s — Best Val Acc: {best_acc:.4f}")

    # load best model weights
    model.load_state_dict(best_wts)
    return model

if __name__ == "__main__":
    dataloaders, sizes, classes = prepare_data()
    model = build_model(num_classes=len(classes))
    best_model = train_model(model, dataloaders, sizes)
    torch.save(best_model.state_dict(), "resnet50_finetuned.pth")
    logger.info("Saved finetuned model to resnet50_finetuned.pth")