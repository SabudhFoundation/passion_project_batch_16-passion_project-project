import sys, os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')))

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.dual_channel.detector import DualChannelDetector

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--backbone',       default='inception',
                    choices=['inception','resnet'])
parser.add_argument('--data_dir',
                    default='deepfake_dataset/real-vs-fake/real-vs-fake')
parser.add_argument('--epochs',         type=int,   default=15)
parser.add_argument('--batch_size',     type=int,   default=32)
parser.add_argument('--lr',             type=float, default=1e-4)
parser.add_argument('--unfreeze_epoch', type=int,   default=5)
parser.add_argument('--feature_dim',    type=int,   default=512)
args = parser.parse_args()

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR  = 'reports/figures'
MODEL_DIR = 'models/dual_channel'
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Device  : {DEVICE}")
print(f"Backbone: {args.backbone}")

# ── Transforms ────────────────────────────────────────────────
img_size = 299 if args.backbone == 'inception' else 224
train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
val_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ── Data ──────────────────────────────────────────────────────
train_ds = ImageFolder(os.path.join(args.data_dir,'train'), train_tf)
val_ds   = ImageFolder(os.path.join(args.data_dir,'valid'), val_tf)
test_ds  = ImageFolder(os.path.join(args.data_dir,'test'),  val_tf)

train_loader = DataLoader(train_ds, args.batch_size,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   args.batch_size,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  args.batch_size,
                          shuffle=False, num_workers=0)

print(f"Train:{len(train_ds):,} | Val:{len(val_ds):,} | Test:{len(test_ds):,}")

# ── Model ─────────────────────────────────────────────────────
model     = DualChannelDetector(args.backbone, args.feature_dim).to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, args.epochs)

history  = {'train_acc':[], 'val_acc':[]}
best_acc = 0
best_path = os.path.join(MODEL_DIR,
                         f'dual_channel_{args.backbone}.pth')

# ── Training Loop ─────────────────────────────────────────────
for epoch in range(args.epochs):
    if epoch == args.unfreeze_epoch:
        print(f"\n>> Fine-tuning: Unfreezing top spatial layers...")
        model.unfreeze_spatial_top(3)
        for g in optimizer.param_groups:
            g['lr'] = args.lr * 0.1

    model.train()
    tp, tl = [], []
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.float().to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        tp.extend((out>0.5).int().cpu().tolist())
        tl.extend(labels.int().cpu().tolist())

    model.eval()
    vp, vl = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            out = model(imgs.to(DEVICE))
            vp.extend((out>0.5).int().cpu().tolist())
            vl.extend(labels.tolist())

    ta = accuracy_score(tl, tp)
    va = accuracy_score(vl, vp)
    history['train_acc'].append(ta)
    history['val_acc'].append(va)
    print(f"Epoch [{epoch+1:02d}/{args.epochs}] "
          f"Train:{ta:.4f} | Val:{va:.4f}")

    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), best_path)
        print(f"   ✅ Best saved → {best_path}")

    scheduler.step()

# ── Test Evaluation ───────────────────────────────────────────
model.load_state_dict(torch.load(best_path))
model.eval()
ap, ab, al = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        probs = model(imgs.to(DEVICE)).cpu()
        ap.extend((probs>0.5).int().tolist())
        ab.extend(probs.tolist())
        al.extend(labels.tolist())

acc  = accuracy_score (al, ap)
prec = precision_score(al, ap)
rec  = recall_score   (al, ap)
f1   = f1_score       (al, ap)
auc  = roc_auc_score  (al, ab)
cm   = confusion_matrix(al, ap)

print(f"""
╔══════════════════════════════════╗
║   FINAL TEST RESULTS             ║
╠══════════════════════════════════╣
║  Backbone  : {args.backbone:<20}  ║
║  Accuracy  : {acc:.4f}                ║
║  Precision : {prec:.4f}                ║
║  Recall    : {rec:.4f}                ║
║  F1-Score  : {f1:.4f}                ║
║  ROC-AUC   : {auc:.4f}                ║
╚══════════════════════════════════╝
Confusion Matrix:
{cm}
""")

# ── Save Plots ────────────────────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'],   label='Validation')
plt.axvline(args.unfreeze_epoch, color='r',
            linestyle='--', label='Fine-tune start')
plt.title(f'Training Curve — Dual Channel ({args.backbone})')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/dual_{args.backbone}_curve.png', dpi=150)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake','Real'],
            yticklabels=['Fake','Real'])
plt.title(f'Confusion Matrix — Dual Channel ({args.backbone})')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/dual_{args.backbone}_cm.png', dpi=150)
print(f"Plots saved → {SAVE_DIR}/")
