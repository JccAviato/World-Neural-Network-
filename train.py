import argparse, os, random, json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from src.utils import load_countries, build_label_maps, make_country_samples, make_continent_samples, make_distance_samples
from src.models import CountryFromLatLonNN, ContinentFromLatLonNN, GreatCircleDistanceNN
from src.features import latlon_to_spherical, pair_to_spherical

def set_seed(sd=42):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)

def to_tensor(X, y=None):
    X = torch.tensor(np.array(X), dtype=torch.float32)
    if y is None:
        return X, None
    if isinstance(y[0], (int, np.integer)):
        y = torch.tensor(np.array(y), dtype=torch.long)
    else:
        y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y

def train_cls(model, dl_train, dl_val, epochs=10, lr=1e-3, device='cpu', patience=4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=max(1, patience//2))
    crit = nn.CrossEntropyLoss()
    best = {'acc':0.0, 'state':None}
    strikes = 0
    for ep in range(1, epochs+1):
        model.train()
        total, correct = 0,0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
            pred = logits.argmax(-1)
            correct += int((pred==yb).sum().item())
            total += xb.size(0)
        train_acc = correct/total
        model.eval()
        vtotal, vcorrect = 0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(-1)
                vcorrect += int((pred==yb).sum().item())
                vtotal += xb.size(0)
        val_acc = vcorrect / vtotal
        sched.step(val_acc)
        if val_acc > best['acc']:
            best = {'acc':val_acc, 'state': model.state_dict()}
            strikes = 0
        else:
            strikes += 1
        print(f"[Epoch {ep:02d}] train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if strikes >= patience:
            break
    model.load_state_dict(best['state'])
    return model, best['acc']

def train_reg(model, dl_train, dl_val, epochs=10, lr=1e-3, device='cpu', patience=4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=max(1, patience//2))
    crit = nn.SmoothL1Loss()
    best = {'mae':1e9, 'state':None}
    strikes = 0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward(); opt.step()
        model.eval()
        mae_sum, nval = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                mae = (pred - yb).abs().mean().item()
                mae_sum += mae * xb.size(0)
                nval += xb.size(0)
        val_mae = mae_sum / nval
        sched.step(val_mae)
        if val_mae < best['mae']:
            best = {'mae': val_mae, 'state': model.state_dict()}
            strikes = 0
        else:
            strikes += 1
        print(f"[Epoch {ep:02d}] val_mae_km={val_mae:.2f}")
        if strikes >= patience:
            break
    model.load_state_dict(best['state'])
    return model, best['mae']

def build_dataloaders_cls(X, y_idx, batch=256, val_frac=0.2):
    X = np.array(X); y_idx = np.array(y_idx)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y_idx, dtype=torch.long)
    ds = TensorDataset(X, y)
    n_val = int(len(ds)*val_frac)
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val])
    return DataLoader(ds_train, batch_size=batch, shuffle=True), DataLoader(ds_val, batch_size=batch)

def build_dataloaders_reg(X, y, batch=256, val_frac=0.2):
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    ds = TensorDataset(X, y)
    n_val = int(len(ds)*val_frac)
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val])
    return DataLoader(ds_train, batch_size=batch, shuffle=True), DataLoader(ds_val, batch_size=batch)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["country","continent","distance"], required=True)
    ap.add_argument("--countries_csv", default="data/countries.csv")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--features", choices=["raw","spherical"], default="raw")
    ap.add_argument("--around", choices=["country","capital"], default="country")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(42)

    if args.task in ("country","continent"):
        countries = load_countries(args.countries_csv)
        c2i, i2c, cont2i, i2cont = build_label_maps(countries)

        if args.task == "country":
            X, y = make_country_samples(countries, n_per_country=600, around=args.around)
            if args.features == 'spherical':
                import numpy as np
                X = latlon_to_spherical(np.array(X)[:,0], np.array(X)[:,1])
                in_dim = X.shape[1]
            else:
                in_dim = 2
            y_idx = [c2i[name] for name in y]
            dl_train, dl_val = build_dataloaders_cls(X, y_idx, batch=args.batch)
            model = CountryFromLatLonNN(n_countries=len(c2i), in_dim=in_dim)
            model, best = train_cls(model, dl_train, dl_val, epochs=args.epochs, device=args.device, patience=args.patience)
            meta = {"task":"country","labels":i2c,"features":args.features}
            torch.save({"state_dict":model.state_dict(), "meta":meta}, os.path.join(args.outdir,"country_from_latlon.pt"))
            print(f"Saved: {os.path.join(args.outdir,'country_from_latlon.pt')} (best_val_acc={best:.3f})")

        else:
            X, y = make_continent_samples(countries, n_per_country=700, around=args.around)
            if args.features == 'spherical':
                import numpy as np
                X = latlon_to_spherical(np.array(X)[:,0], np.array(X)[:,1])
                in_dim = X.shape[1]
            else:
                in_dim = 2
            y_idx = [cont2i[name] for name in y]
            dl_train, dl_val = build_dataloaders_cls(X, y_idx, batch=args.batch)
            model = ContinentFromLatLonNN(n_continents=len(cont2i), in_dim=in_dim)
            model, best = train_cls(model, dl_train, dl_val, epochs=args.epochs, device=args.device, patience=args.patience)
            meta = {"task":"continent","labels": {i:str(v) for i,v in i2cont.items()},"features":args.features}
            torch.save({"state_dict":model.state_dict(), "meta":meta}, os.path.join(args.outdir,"continent_from_latlon.pt"))
            print(f"Saved: {os.path.join(args.outdir,'continent_from_latlon.pt')} (best_val_acc={best:.3f})")

    else:
        X, y = make_distance_samples(n_samples=30000)
        if args.features == 'spherical':
            import numpy as np
            X = pair_to_spherical(np.array(X)[:,0], np.array(X)[:,1], np.array(X)[:,2], np.array(X)[:,3])
            in_dim = X.shape[1]
        else:
            in_dim = 4
        dl_train, dl_val = build_dataloaders_reg(X, y, batch=args.batch)
        model = GreatCircleDistanceNN(in_dim=in_dim)
        model, best_mae = train_reg(model, dl_train, dl_val, epochs=args.epochs, device=args.device, patience=args.patience)
        meta = {"task":"distance","units":"km","features":args.features}
        torch.save({"state_dict":model.state_dict(), "meta":meta}, os.path.join(args.outdir,"great_circle_distance.pt"))
        print(f"Saved: {os.path.join(args.outdir,'great_circle_distance.pt')} (best_val_mae_km={best_mae:.2f})")

if __name__ == "__main__":
    main()
