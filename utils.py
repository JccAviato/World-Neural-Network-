import math
import random
import csv
from dataclasses import dataclass
from typing import List, Tuple

EARTH_RADIUS_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

@dataclass
class Country:
    name: str
    iso2: str
    continent: str
    lat: float
    lon: float
    capital: str
    capital_lat: float
    capital_lon: float
    region: str

def load_countries(csv_path: str) -> List[Country]:
    out = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(Country(
                name=row['country'],
                iso2=row['iso2'],
                continent=row['continent'],
                lat=float(row['lat']),
                lon=float(row['lon']),
                capital=row['capital'],
                capital_lat=float(row['capital_lat']),
                capital_lon=float(row['capital_lon']),
                region=row['region']
            ))
    return out

def build_label_maps(countries: List[Country]):
    countries_sorted = sorted(countries, key=lambda c: c.name)
    continents_sorted = sorted(sorted(set(c.continent for c in countries)))
    country_to_idx = {c.name: i for i, c in enumerate(countries_sorted)}
    idx_to_country = {i: c.name for i, c in enumerate(countries_sorted)}
    cont_to_idx = {c: i for i, c in enumerate(continents_sorted)}
    idx_to_cont = {i: c for i, c in enumerate(continents_sorted)}
    return country_to_idx, idx_to_country, cont_to_idx, idx_to_cont

def sample_gaussian_point(lat, lon, km_sigma=300.0):
    lat_sigma_deg = km_sigma / 111.0
    lon_sigma_deg = km_sigma / (111.0 * max(0.1, math.cos(math.radians(lat))))
    dlat = random.gauss(0, lat_sigma_deg)
    dlon = random.gauss(0, lon_sigma_deg)
    return max(-89.9999, min(89.9999, lat + dlat)), ((lon + dlon + 180) % 360) - 180

def make_country_samples(countries: List[Country], n_per_country=500, around='country'):
    X, y = [], []
    for c in countries:
        base_lat, base_lon = (c.lat, c.lon) if around == 'country' else (c.capital_lat, c.capital_lon)
        for _ in range(n_per_country):
            lat, lon = sample_gaussian_point(base_lat, base_lon, km_sigma=400.0)
            X.append([lat, lon])
            y.append(c.name)
    return X, y

def make_continent_samples(countries: List[Country], n_per_country=500, around='country'):
    X, y = [], []
    for c in countries:
        base_lat, base_lon = (c.lat, c.lon) if around == 'country' else (c.capital_lat, c.capital_lon)
        for _ in range(n_per_country):
            lat, lon = sample_gaussian_point(base_lat, base_lon, km_sigma=500.0)
            X.append([lat, lon])
            y.append(c.continent)
    return X, y

def make_distance_samples(n_samples=20000):
    X, y = [], []
    for _ in range(n_samples):
        lat1 = random.uniform(-89.9, 89.9)
        lon1 = random.uniform(-180.0, 180.0)
        lat2 = random.uniform(-89.9, 89.9)
        lon2 = random.uniform(-180.0, 180.0)
        d = haversine_km(lat1, lon1, lat2, lon2)
        X.append([lat1, lon1, lat2, lon2])
        y.append(d)
    return X, y
