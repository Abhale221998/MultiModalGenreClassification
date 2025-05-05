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
DATA_DIR      = "movie_cls_dataset"
BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-3
# DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE        = torch.device("cuda")
# ────────────────────────────────────────────────────────────────────────────

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
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
    datasets_dict = {
        phase: datasets.ImageFolder(os.path.join(DATA_DIR, phase), tfms[phase])
        for phase in ("train", "val")
    }
    dataloaders = {
        phase: torch.utils.data.DataLoader(
            datasets_dict[phase],
            batch_size=BATCH_SIZE,
            shuffle=(phase == "train"),
            num_workers=4,
            pin_memory=True
        )
        for phase in ("train", "val")
    }
    sizes = {phase: len(datasets_dict[phase]) for phase in ("train", "val")}
    classes = datasets_dict["train"].classes
    logger.info(f"Found {len(classes)} classes: {classes}")
    logger.info(f"Dataset sizes — train: {sizes['train']}, val: {sizes['val']}")
    return dataloaders, sizes, classes

def build_model(num_classes):
    # load pretrained EfficientNet-B0
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    # replace the final classifier layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)

def train_model(model, dataloaders, sizes):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{NUM_EPOCHS} starting")
        for phase in ("train", "val"):
            is_train = (phase == "train")
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0

            loader = dataloaders[phase]
            desc = f"{phase.capitalize()} Epoch {epoch}"
            for inputs, labels in tqdm(loader, desc=desc, unit="batch"):
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
            epoch_acc  = running_corrects / sizes[phase]
            logger.info(f"{phase.capitalize()} — Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                logger.info(f"New best validation accuracy: {best_acc:.4f}")

        logger.info(f"Epoch {epoch} completed\n")

    total_time = time.time() - start_time
    mins, secs = divmod(total_time, 60)
    logger.info(f"Training complete in {int(mins)}m {int(secs)}s — Best Val Acc: {best_acc:.4f}")

    model.load_state_dict(best_wts)
    return model

if __name__ == "__main__":
    dataloaders, sizes, classes = prepare_data()
    model = build_model(num_classes=len(classes))
    best_model = train_model(model, dataloaders, sizes)
    torch.save(best_model.state_dict(), "efficientnet_b0_finetuned.pth")
    logger.info("Saved finetuned model to efficientnet_b0_finetuned.pth")