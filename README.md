# World & Geography Neural Networks (v3)

This bundle includes:
- Country & Continent classifiers, Great-circle distance regressor (PyTorch)
- Feature encodings (`--features spherical`), capital-based sampling, early stopping/LR scheduler
- Streamlit interactive map with optional probability heatmap
- FastAPI service with JSON endpoints
- EuroSAT satellite land-cover classifier (transfer learning)
- Nearest-capital utility and terminal quiz

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train
```bash
python train.py --task country   --features spherical --around capital --epochs 15 --patience 6
python train.py --task continent --epochs 10
python train.py --task distance  --features spherical --epochs 12
```

### Inference
```bash
python infer.py --ckpt models/country_from_latlon.pt   --mode country   --features spherical --lat 48.85 --lon 2.35
python infer.py --ckpt models/continent_from_latlon.pt --mode continent --lat -33.86 --lon 151.21
python infer.py --ckpt models/great_circle_distance.pt --mode distance  --features spherical --lat 40.7128 --lon -74.0060 --lat2 51.5074 --lon2 -0.1278
```

### Streamlit app
```bash
streamlit run app.py
```
- Click map to set coordinates.
- Toggle heatmap to visualize probability around the selected point.

### FastAPI
```bash
uvicorn api:app --reload
```
Examples:
```bash
curl -X POST "http://127.0.0.1:8000/predict/country?ckpt=models/country_from_latlon.pt&features=spherical" -H "Content-Type: application/json" -d '{"lat":48.8566,"lon":2.3522}'
curl -X POST "http://127.0.0.1:8000/predict/continent?ckpt=models/continent_from_latlon.pt" -H "Content-Type: application/json" -d '{"lat":-33.86,"lon":151.21}'
curl -X POST "http://127.0.0.1:8000/predict/distance?ckpt=models/great_circle_distance.pt&features=raw" -H "Content-Type: application/json" -d '{"lat1":40.7,"lon1":-74.0,"lat2":51.5,"lon2":-0.12}'
```

### Satellite (EuroSAT)
```bash
python train_sat.py --epochs 3 --device cpu
python infer_sat.py --ckpt models/eurosat_resnet18.pt --image path/to/patch.jpg
```

### Utilities
```bash
python nearest_capital.py --lat 40.7 --lon -74.0
python quiz.py --n 10
```
