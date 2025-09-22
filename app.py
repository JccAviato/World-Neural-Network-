import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import numpy as np
import torch
from src.models import CountryFromLatLonNN, ContinentFromLatLonNN, GreatCircleDistanceNN
from src.features import latlon_to_spherical, pair_to_spherical

st.set_page_config(page_title="World & Geography NN", layout="wide")
st.title("🌍 World & Geography Neural Networks — Interactive Demo")

st.sidebar.header("Models")
mode = st.sidebar.selectbox("Choose task", ["Country from lat/lon", "Continent from lat/lon", "Great-circle distance"])
features = st.sidebar.selectbox("Feature encoding", ["raw", "spherical"])

ckpt_path = st.sidebar.text_input("Path to checkpoint (.pt)", value="models/country_from_latlon.pt" if "Country" in mode else ("models/continent_from_latlon.pt" if "Continent" in mode else "models/great_circle_distance.pt"))

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

@st.cache_resource
def load_model(ckpt_path, features):
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
        st.error("Unknown model task in checkpoint")
        st.stop()
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, meta

st.subheader("Pick locations on the map")
m = folium.Map(location=[20,0], zoom_start=2, tiles="CartoDB positron")
folium.LatLngPopup().add_to(m)
out = st_folium(m, height=500, width=900)

col1, col2 = st.columns(2)
lat = col1.number_input("Latitude", value=48.8566, min_value=-89.9999, max_value=89.9999, step=0.0001, format="%.4f")
lon = col2.number_input("Longitude", value=2.3522, min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f")

if out and out.get("last_clicked"):
    lat = float(out["last_clicked"]["lat"])
    lon = float(out["last_clicked"]["lng"])

model = None
meta = None
if st.sidebar.button("Load model"):
    try:
        model, meta = load_model(ckpt_path, features)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

if model is not None:
    if "Country" in mode:
        x = np.array([[lat, lon]])
        if features == "spherical":
            x = latlon_to_spherical(x[:,0], x[:,1])
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x_t).numpy()[0]
        probs = softmax(logits)
        labels = [meta["labels"][i] for i in range(len(probs))]
        order = np.argsort(-probs)[:5]
        st.subheader("Top-5 countries")
        for i in order:
            st.write(f"{labels[i]} — p={probs[i]:.3f}")
    # Probability choropleth (by country)
    st.markdown("---")
    st.subheader("Probability choropleth (by country)")
    st.caption("Colors show the model's per-country probability at the selected point (darker = higher).")
    do_choro = st.checkbox("Render choropleth", value=False)
    if do_choro:
        try:
            import geopandas as gpd
            import pandas as pd

            # Load Natural Earth low-res countries
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

            # Build probabilities per label
            probs_map = {labels[i]: float(probs[i]) for i in range(len(probs))}

            # Some light name harmonization
            name_fix = {
                "United States of America": "United States",
                "Dem. Rep. Congo": "Congo (Kinshasa)",
                "Czechia": "Czechia",
                "Bosnia and Herz.": "Bosnia and Herzegovina",
                "Central African Rep.": "Central African Republic",
                "Dominican Rep.": "Dominican Republic",
                "S. Sudan": "South Sudan",
            }

            def norm_name(n):
                return name_fix.get(n, n)

            world["name_norm"] = world["name"].apply(norm_name)
            df = pd.DataFrame([{"name_norm": k, "prob": v} for k, v in probs_map.items()])
            merged = world.merge(df, how="left", on="name_norm")
            merged = merged.fillna({"prob": 0.0})

            # Render choropleth
            ch = folium.Map(location=[lat, lon], zoom_start=2, tiles="CartoDB positron")
            from branca.colormap import linear
            cmap = linear.YlOrRd_09.scale(0, max(merged["prob"].max(), 1e-6))
            folium.GeoJson(
                merged.__geo_interface__,
                style_function=lambda feature: {
                    "fillColor": cmap(feature["properties"].get("prob", 0.0)),
                    "color": "black",
                    "weight": 0.2,
                    "fillOpacity": 0.7 if feature["properties"].get("prob", 0.0) > 0 else 0.05,
                },
                highlight_function=lambda feat: {"weight": 1, "color": "#666", "fillOpacity": 0.8},
                tooltip=folium.GeoJsonTooltip(fields=["name", "prob"], aliases=["Country", "Probability"], localize=True)
            ).add_to(ch)
            cmap.add_to(ch)
            st_folium(ch, height=520, width=900)
        except Exception as e:
            st.error(f"Choropleth failed: {e}")
    
        st.markdown("---")
        st.subheader("Probability heatmap (experimental)")
        grid_deg = st.slider("Grid step (degrees)", 0.5, 5.0, 2.0, 0.5)
        span = st.slider("Span from center (degrees)", 2.0, 20.0, 8.0, 1.0)
        do_heat = st.checkbox("Generate heatmap around the clicked point", value=False)
        if do_heat:
            lats = np.arange(lat - span, lat + span + 1e-9, grid_deg)
            lons = np.arange(lon - span, lon + span + 1e-9, grid_deg)
            pts = []
            for la in lats:
                for lo in lons:
                    x = np.array([[la, lo]])
                    if features == "spherical":
                        x = latlon_to_spherical(x[:,0], x[:,1])
                    x_t = torch.tensor(x, dtype=torch.float32)
                    with torch.no_grad():
                        logits = model(x_t).numpy()[0]
                    p = float(np.max(np.exp(logits - logits.max())/np.exp(logits - logits.max()).sum()))
                    pts.append([la, lo, p])
            hm = folium.Map(location=[lat, lon], zoom_start=4, tiles="CartoDB positron")
            HeatMap(pts, radius=15, blur=20).add_to(hm)
            st_folium(hm, height=500, width=900)

    elif "Continent" in mode:
        x = np.array([[lat, lon]])
        if features == "spherical":
            x = latlon_to_spherical(x[:,0], x[:,1])
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x_t).numpy()[0]
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        label = meta["labels"][str(idx)]
        st.subheader("Prediction")
        st.write(f"**{label}** — p={probs[idx]:.3f}")

    else:
        col3, col4 = st.columns(2)
        lat2 = col3.number_input("Latitude (point B)", value=51.5074, min_value=-89.9999, max_value=89.9999, step=0.0001, format="%.4f")
        lon2 = col4.number_input("Longitude (point B)", value=-0.1278, min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f")
        x = np.array([[lat, lon, lat2, lon2]])
        if features == "spherical":
            x = pair_to_spherical(x[:,0], x[:,1], x[:,2], x[:,3])
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            d = float(model(x_t).numpy()[0])
        st.subheader("Distance")
        st.write(f"Great-circle distance ≈ **{d:.1f} km**")
