import argparse, torch, numpy as np
from src.models import CountryFromLatLonNN, ContinentFromLatLonNN, GreatCircleDistanceNN

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def load_model(ckpt_path, features="raw"):
    blob = torch.load(ckpt_path, map_location="cpu")
    meta = blob.get("meta", {})
    task = meta.get("task", "")
    if task == "country":
        n = len(meta["labels"])
        in_dim = 6 if features == "spherical" else 2
        model = CountryFromLatLonNN(n, in_dim=in_dim)
    elif task == "continent":
        n = len(meta["labels"])
        in_dim = 6 if features == "spherical" else 2
        model = ContinentFromLatLonNN(n, in_dim=in_dim)
    elif task == "distance":
        in_dim = 12 if features == "spherical" else 4
        model = GreatCircleDistanceNN(in_dim=in_dim)
    else:
        raise ValueError("Unknown task in checkpoint meta")
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    return model, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["country","continent","distance"], required=True)
    ap.add_argument("--features", choices=["raw","spherical"], default="raw")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--lat2", type=float)
    ap.add_argument("--lon2", type=float)
    args = ap.parse_args()

    model, meta = load_model(args.ckpt, features=args.features)

    if args.mode == "country":
        import numpy as np
        x = np.array([[args.lat, args.lon]])
        if args.features == "spherical":
            from src.features import latlon_to_spherical
            x = latlon_to_spherical(x[:,0], x[:,1])
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x_t).numpy()[0]
        probs = softmax(logits)
        labels = [meta["labels"][i] for i in range(len(probs))]
        order = np.argsort(-probs)[:5]
        for i in order:
            print(f"{labels[i]:20s}  p={probs[i]:.3f}")
    elif args.mode == "continent":
        import numpy as np
        x = np.array([[args.lat, args.lon]])
        if args.features == "spherical":
            from src.features import latlon_to_spherical
            x = latlon_to_spherical(x[:,0], x[:,1])
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x_t).numpy()[0]
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        print(f"Continent={meta['labels'][str(idx)]}  p={probs[idx]:.3f}")
    else:
        import numpy as np
        x = np.array([[args.lat, args.lon, args.lat2, args.lon2]])
        if args.features == "spherical":
            from src.features import pair_to_spherical
            x = pair_to_spherical(x[:,0], x[:,1], x[:,2], x[:,3])
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            d = float(model(x_t).numpy()[0])
        print(f"Great-circle distance ≈ {d:.1f} km")

if __name__ == "__main__":
    main()
