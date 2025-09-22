from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import torch, numpy as np
from src.models import CountryFromLatLonNN, ContinentFromLatLonNN, GreatCircleDistanceNN
from src.features import latlon_to_spherical, pair_to_spherical

app = FastAPI(title="World & Geography NN API", version="1.0")

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def load_model(ckpt_path: str, features: str):
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
        raise HTTPException(status_code=400, detail="Unknown task in checkpoint")
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, meta

class LatLon(BaseModel):
    lat: float
    lon: float

class DistanceReq(BaseModel):
    lat1: float
    lon1: float
    lat2: float
    lon2: float

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict/country")
def predict_country(body: LatLon, ckpt: str = Query(...), features: str = Query("raw")):
    model, meta = load_model(ckpt, features)
    x = np.array([[body.lat, body.lon]])
    if features == "spherical":
        x = latlon_to_spherical(x[:,0], x[:,1])
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_t).numpy()[0]
    probs = softmax(logits)
    labels = [meta["labels"][i] for i in range(len(probs))]
    order = np.argsort(-probs)
    top5 = [{"label": labels[i], "prob": float(probs[i])} for i in order[:5]]
    return {"top5": top5}

@app.post("/predict/continent")
def predict_continent(body: LatLon, ckpt: str = Query(...), features: str = Query("raw")):
    model, meta = load_model(ckpt, features)
    x = np.array([[body.lat, body.lon]])
    if features == "spherical":
        x = latlon_to_spherical(x[:,0], x[:,1])
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_t).numpy()[0]
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return {"continent": meta["labels"][str(idx)], "prob": float(probs[idx])}

@app.post("/predict/distance")
def predict_distance(body: DistanceReq, ckpt: str = Query(...), features: str = Query("raw")):
    model, meta = load_model(ckpt, features)
    x = np.array([[body.lat1, body.lon1, body.lat2, body.lon2]])
    if features == "spherical":
        x = pair_to_spherical(x[:,0], x[:,1], x[:,2], x[:,3])
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        d = float(model(x_t).numpy()[0])
    return {"distance_km": max(0.0, d)}
