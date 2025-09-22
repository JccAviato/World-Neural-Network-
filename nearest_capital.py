import argparse, csv, math

EARTH_RADIUS_KM = 6371.0088
def haversine_km(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*EARTH_RADIUS_KM*math.asin(math.sqrt(a))

def load_capitals(csv_path):
    out = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                "country": row["country"],
                "capital": row["capital"],
                "lat": float(row["capital_lat"]),
                "lon": float(row["capital_lon"])
            })
    return out

def nearest(lat, lon, caps):
    best = None
    for c in caps:
        d = haversine_km(lat, lon, c["lat"], c["lon"])
        if best is None or d < best[0]:
            best = (d, c)
    return best

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries_csv", default="data/countries.csv")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    args = ap.parse_args()
    caps = load_capitals(args.countries_csv)
    d, c = nearest(args.lat, args.lon, caps)
    print(f"Nearest capital: {c['capital']} ({c['country']}) — {d:.1f} km away")
