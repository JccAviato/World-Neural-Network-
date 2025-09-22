import argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_dataloaders(root, batch=64):
    tfm_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    train_ds = datasets.EuroSAT(root=root, download=True, transform=tfm_train)
    val_ds = datasets.EuroSAT(root=root, download=True, transform=tfm_val)
    n = len(train_ds)
    n_val = n//10
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n-n_val, n_val])
    return DataLoader(train_ds, batch_size=batch, shuffle=True), DataLoader(val_ds, batch_size=batch)

def train(root="data/eurosat", epochs=5, lr=1e-3, device="cpu", out="models/eurosat_resnet18.pt"):
    dl_train, dl_val = get_dataloaders(root)
    num_classes = 10
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = (0.0, None)
    for ep in range(1, epochs+1):
        model.train()
        tot, correct = 0, 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
            pred = logits.argmax(1)
            correct += int((pred==yb).sum().item())
            tot += xb.size(0)
        train_acc = correct/tot
        model.eval()
        vtot, vcorrect = 0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(1)
                vcorrect += int((pred==yb).sum().item())
                vtot += xb.size(0)
        val_acc = vcorrect / vtot
        print(f"[Epoch {ep:02d}] train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best[0]:
            best = (val_acc, model.state_dict())
    if best[1] is not None:
        torch.save({"state_dict":best[1], "meta":{"task":"satellite_land_cover","labels":"EuroSAT-10"}}, out)
        print(f"Saved: {out} (best_val_acc={best[0]:.3f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/eurosat")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="models/eurosat_resnet18.pt")
    args = ap.parse_args()
    train(root=args.root, epochs=args.epochs, lr=args.lr, device=args.device, out=args.out)
