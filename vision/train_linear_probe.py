"""Train a linear probe on top of a frozen DINOv2 backbone for CIFAR-10.

Produces a classifier state_dict (.pt) that can be passed to
randopt_vision.py via --linear_init_path to start perturbation search
from a known-good point (closer to the LLM setting where the base model
already has task capability).

Usage:
    python vision/train_linear_probe.py \
        --model_name facebook/dinov2-base \
        --data_dir data/cifar10 \
        --output_path data/cifar10/linear_probe_dinov2base.pt \
        --epochs 10 \
        --lr 0.001
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import Dinov2Model

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/dinov2-base")
    parser.add_argument("--data_dir", default="data/cifar10")
    parser.add_argument("--output_path", default="data/cifar10/linear_probe_dinov2base.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model_name}")

    backbone = Dinov2Model.from_pretrained(args.model_name).to(device).eval()
    embed_dim = backbone.config.hidden_size
    classifier = nn.Linear(embed_dim, 10).to(device)

    try:
        from datasets import load_dataset
        hf_train = load_dataset("uoft-cs/cifar10", split="train")
        hf_val   = load_dataset("uoft-cs/cifar10", split="test")
        def _hf_collate(batch):
            imgs = torch.stack([_transform(b["img"]) for b in batch])
            lbls = torch.tensor([b["label"] for b in batch])
            return imgs, lbls
        train_dl = DataLoader(hf_train, batch_size=args.batch_size, shuffle=True,  collate_fn=_hf_collate, num_workers=4)
        val_dl   = DataLoader(hf_val,   batch_size=args.batch_size, shuffle=False, collate_fn=_hf_collate, num_workers=4)
        print("Loaded CIFAR-10 from HuggingFace datasets cache")
    except Exception as e:
        print(f"HF load failed ({e}), falling back to torchvision...")
        train_ds = datasets.CIFAR10(args.data_dir, train=True,  download=True, transform=_transform)
        val_ds   = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=_transform)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
        val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    optim = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Precompute features to avoid redundant backbone passes
    print("Precomputing DINOv2 features...")
    def get_features(loader):
        feats, labels = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                f = backbone(pixel_values=imgs.to(device)).pooler_output
                feats.append(f.cpu())
                labels.append(lbls)
        return torch.cat(feats), torch.cat(labels)

    train_feats, train_labels = get_features(train_dl)
    val_feats,   val_labels   = get_features(val_dl)
    feat_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    feat_dl = DataLoader(feat_ds, batch_size=args.batch_size, shuffle=True)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        classifier.train()
        for feats, lbls in feat_dl:
            feats, lbls = feats.to(device), lbls.to(device)
            optim.zero_grad()
            loss = criterion(classifier(feats), lbls)
            loss.backward()
            optim.step()

        # Validation
        classifier.eval()
        with torch.no_grad():
            logits = classifier(val_feats.to(device))
            val_acc = (logits.argmax(dim=-1).cpu() == val_labels).float().mean().item()
        print(f"Epoch {epoch+1}/{args.epochs}: val_acc={val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
            torch.save(classifier.state_dict(), args.output_path)

    print(f"\nBest val accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
