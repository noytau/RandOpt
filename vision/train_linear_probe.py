"""Train a linear probe on top of a frozen DINOv2 backbone.

Works with any dataset registered in data_handlers (cifar10, fgvc_aircraft, cub200, ...).
Produces a classifier state_dict (.pt) for use with --linear_init_path in randopt_vision.py.

Usage:
    python vision/train_linear_probe.py \
        --model_name facebook/dinov2-base \
        --dataset cifar10 \
        --num_classes 10 \
        --output_path data/cifar10/linear_probe_dinov2base.pt \
        --epochs 10
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from transformers import Dinov2Model

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_split(dataset_name: str, data_dir: str, split: str, max_samples=None):
    """Load a split via the data_handlers registry, returning PIL-transformable items."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_handlers import get_dataset_handler
    handler = get_dataset_handler(dataset_name)
    path = data_dir or handler.default_train_path
    return handler.load_data(path=path, split=split, max_samples=max_samples)


def collate_items(items, device):
    """Stack raw image tensors (already float32 [0,1] CHW) and re-normalize for DINOv2."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imgs, labels = [], []
    for item in items:
        img = item["image_tensor"]  # float32 (3, H, W) in [0,1]
        # Resize to 224 if needed
        if img.shape[-1] != 224 or img.shape[-2] != 224:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False
            ).squeeze(0)
        img = normalize(img)
        imgs.append(img)
        labels.append(item["class_id"])
    return torch.stack(imgs).to(device), torch.tensor(labels, dtype=torch.long).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/dinov2-base")
    parser.add_argument("--dataset", default="cifar10",
                        help="Dataset name from data_handlers registry")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--data_dir", default=None,
                        help="Data root dir (uses handler default if not set)")
    parser.add_argument("--output_path", default="data/cifar10/linear_probe_dinov2base.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Backbone: {args.model_name} | Dataset: {args.dataset} | Classes: {args.num_classes}")

    backbone = Dinov2Model.from_pretrained(args.model_name).to(device).eval()
    embed_dim = backbone.config.hidden_size
    classifier = nn.Linear(embed_dim, args.num_classes).to(device)

    print("Loading data...")
    train_items = load_split(args.dataset, args.data_dir, "train")
    val_items   = load_split(args.dataset, args.data_dir, "test")
    print(f"  Train: {len(train_items)} | Val: {len(val_items)}")

    print("Precomputing DINOv2 features...")
    def get_features(items):
        all_feats, all_labels = [], []
        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            imgs, lbls = collate_items(batch, device)
            with torch.no_grad():
                feats = backbone(pixel_values=imgs).pooler_output
            all_feats.append(feats.cpu())
            all_labels.append(lbls.cpu())
            if (i // args.batch_size) % 5 == 0:
                print(f"  {i+len(batch)}/{len(items)}", end="\r")
        print()
        return torch.cat(all_feats), torch.cat(all_labels)

    train_feats, train_labels = get_features(train_items)
    val_feats,   val_labels   = get_features(val_items)

    feat_dl = DataLoader(TensorDataset(train_feats, train_labels),
                         batch_size=args.batch_size, shuffle=True, num_workers=0)

    optim = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        classifier.train()
        for feats, lbls in feat_dl:
            feats, lbls = feats.to(device), lbls.to(device)
            optim.zero_grad()
            criterion(classifier(feats), lbls).backward()
            optim.step()

        classifier.eval()
        with torch.no_grad():
            val_acc = (classifier(val_feats.to(device)).argmax(dim=-1).cpu() == val_labels).float().mean().item()
        print(f"Epoch {epoch+1}/{args.epochs}: val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
            torch.save(classifier.state_dict(), args.output_path)

    print(f"\nBest val accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
