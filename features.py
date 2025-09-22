import numpy as np
import numpy

def latlon_to_spherical(lat, lon):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    return np.stack([
        np.sin(lat_r), np.cos(lat_r),
        np.sin(lon_r), np.cos(lon_r),
        np.sin(2*lat_r), np.sin(2*lon_r)
    ], axis=-1)

def pair_to_spherical(lat1, lon1, lat2, lon2):
    a = latlon_to_spherical(lat1, lon1)
    b = latlon_to_spherical(lat2, lon2)
    return np.concatenate([a, b], axis=-1)
