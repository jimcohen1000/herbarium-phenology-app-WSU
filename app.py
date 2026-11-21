import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import time
import re
from datetime import datetime
import urllib.parse
import plotly.express as px
import plotly.graph_objects as go

current_year = datetime.now().year

# --- RATE LIMITING GLOBALS ---
LAST_API_CALL_TIME = 0.0
API_RATE_LIMIT_DELAY = 0.7  # seconds

# ==========================================
#        APP SETUP & CANONICAL SCHEMA
# ==========================================
st.set_page_config(
    page_title="Phenology & Climate Tracker", 
    layout="wide"
)

DB_FILE = "phenology_dataset.csv"

# Programmatically generate ALL 265 ClimateNA variables
CORE_ANNUAL = [
    "MAT",
    "MWMT",
    "MCMT",
    "TD",
    "MAP",
    "MSP",
    "AHM",
    "SHM",
    "DD_0",
    "DD5",
    "DD_18",
    "DD18",
    "NFFD",
    "bFFP",
    "eFFP",
    "FFP",
    "PAS",
    "EMT",
    "EXT",
    "Eref",
    "CMD",
    "MAR",
    "RH",
    "CMI",
    "DD1040"
]

SEAS_MONTHLY = [
    "Tmax",
    "Tmin",
    "Tave",
    "PPT",
    "Rad",
    "DD_0",
    "DD5",
    "DD_18",
    "DD18",
    "NFFD",
    "PAS",
    "Eref",
    "CMD",
    "RH",
    "CMI"
]

CNA_VARS = CORE_ANNUAL.copy()

for sv in SEAS_MONTHLY:
    for s in ["wt", "sp", "sm", "at"]:
        CNA_VARS.append(f"{sv}_{s}")
        
for sv in SEAS_MONTHLY:
    for m in [f"{i:02d}" for i in range(1, 13)]:
        CNA_VARS.append(f"{sv}{m}")

# Build canonical schema
CANONICAL_COLUMNS = [
    "Data_Source", 
    "Collector", 
    "Col_Number", 
    "Barcode", 
    "Species", 
    "Year", 
    "DOY", 
    "Latitude", 
    "Longitude", 
    "Elevation", 
    "County",
    "Photoperiod_Hours", 
    "Phenology_Scored", 
    "Flowering", 
    "Fruiting", 
    "Vegetative", 
    "URL", 
    "Y_3Mo_prior_mean_Tave", 
    "N_3Mo_prior_mean_Tave", 
    "Tave_Anomaly", 
    "Y_3Mo_prior_mean_PPT", 
    "N_3Mo_prior_mean_PPT", 
    "PPT_Anomaly"
] + [f"Y_{v}" for v in CNA_VARS] + [f"N_{v}" for v in CNA_VARS]

def init_db(filename=DB_FILE):
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=CANONICAL_COLUMNS)
        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
        needs_save = False
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                if col in ["Phenology_Scored", "Flowering", "Fruiting", "Vegetative"]:
                    df[col] = False
                else:
                    df[col] = np.nan
                needs_save = True
        if needs_save:
            df = df[CANONICAL_COLUMNS]
            df.to_csv(filename, index=False)

init_db()

def save_with_ordered_columns(df_to_save, filepath=DB_FILE):
    for col in CANONICAL_COLUMNS:
        if col not in df_to_save.columns:
            df_to_save[col] = np.nan
    df_to_save = df_to_save[CANONICAL_COLUMNS]
    df_to_save.to_csv(filepath, index=False)

# ==========================================
#        DATA PIPELINE FUNCTIONS
# ==========================================

def safe_float(val):
    try: 
        if pd.notna(val) and val != '':
            return float(val)
        return None
    except: 
        return None

def safe_int(val):
    try: 
        if pd.notna(val) and val != '':
            return int(float(val))
        return None
    except: 
        return None

def parse_elev(val):
    try: 
        if val and str(val).strip() != "":
            return float(val)
        return None
    except: 
        return None

def ensure_url_scheme(url_str):
    if not url_str or pd.isna(url_str): 
        return ""
    
    url_str = str(url_str).strip()
    
    if url_str == "Manual": 
        return url_str
        
    if not url_str.startswith(('http://', 'https://')): 
        return 'https://' + url_str
        
    return url_str

@st.cache_data(persist="disk", show_spinner=False, max_entries=5000)
def get_elevation(lat, lon):
    try:
        smart_sleep()  # Slow down requests to avoid IP bans
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url, timeout=5)
        
        # This will trigger the exception block below if Open-Meteo blocks us (e.g., 429 or 403 errors)
        res.raise_for_status() 
        
        if res.status_code == 200:
            elevations = res.json().get('elevation')
            if elevations and len(elevations) > 0: 
                return float(elevations[0])
                
    except Exception as e: 
        # Now you will actually see the error in your terminal running Streamlit!
        print(f"⚠️ Elevation API failed for Lat: {lat}, Lon: {lon} | Error: {e}")
        return None
        
    return None

def normalize_location(lat, lon, provided_elev=None):
    """
    Rounds coordinates and resolves elevation to guarantee consistent cache keys.
    """
    norm_lat = round(float(lat), 4)
    norm_lon = round(float(lon), 4)
    
    # We remove 'or provided_elev == 0.0' so actual sea-level inputs aren't overwritten
    if provided_elev is None or pd.isna(provided_elev):
        norm_elev = get_elevation(norm_lat, norm_lon)
        # If API fails, return None, DO NOT force 0.0
        if norm_elev is None:
            return norm_lat, norm_lon, None
    else:
        norm_elev = float(provided_elev)
        
    return norm_lat, norm_lon, round(norm_elev, 1)

def smart_sleep():
    global LAST_API_CALL_TIME
    current_time = time.time()
    elapsed = current_time - LAST_API_CALL_TIME

    if elapsed < API_RATE_LIMIT_DELAY:
        time.sleep(API_RATE_LIMIT_DELAY - elapsed)

    LAST_API_CALL_TIME = time.time()

@st.cache_data(persist="disk", show_spinner=False, max_entries=5000)
def get_climate_data(lat, lon, el, prd):    
    if pd.isna(el) or el is None:
        return {"error": "Elevation missing", "systemic": False}
    
    base = "https://api.climatena.ca/api/cnaApi6/LatLonEl"
    url = f"{base}?ID1=1&ID2=t1&lat={lat}&lon={lon}&el={el}&prd={prd}&varYSM=YSM"
    
    try:
        smart_sleep() 
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            try:
                data = res.json()
                if isinstance(data, list):
                    return data[0]
                return data
            except ValueError:
                return {"error": "API returned an HTML block page instead of data.", "systemic": True}
        elif res.status_code in [403, 429]:
            return {"error": f"API Rate Limit Exceeded (HTTP {res.status_code})", "systemic": True}
        elif res.status_code == 400:
            return {"error": f"Server Error (HTTP 400) - Bad Request formatting.", "systemic": True}
        else:
            return {"error": f"Server Error (HTTP {res.status_code})", "systemic": True}
    except requests.exceptions.Timeout:
        return {"error": "API Request Timed Out", "systemic": True}
    except Exception as e:
        return {"error": f"Connection Failed", "systemic": True}

def calc_prior_3_months(year, doy):
    try:
        dt = datetime(year, 1, 1) + pd.Timedelta(days=doy-1)
        target_months = []
        target_years = []
        
        for i in range(1, 4):
            m = dt.month - i
            y = year
            if m <= 0:
                m += 12
                y -= 1
            target_months.append(m)
            target_years.append(y)
            
        return target_years, target_months
    except: 
        return None, None

def normalize_key(k_str):
    k_str = k_str.lower()
    if k_str.startswith('precip'): 
        k_str = k_str.replace('precip', 'ppt', 1)
    elif k_str.startswith('prcp'): 
        k_str = k_str.replace('prcp', 'ppt', 1)
    elif k_str.startswith('prec'): 
        k_str = k_str.replace('prec', 'ppt', 1)
    elif k_str.startswith('pr') and not k_str.startswith('prd') and not k_str.startswith('ppt'): 
        k_str = k_str.replace('pr', 'ppt', 1)
        
    k_str = k_str.replace('<', '_').replace('>', '')
    return k_str

def get_climate_val(data_dict, prefix, m_str):
    if not data_dict or "error" in data_dict: 
        return None
        
    target_m_padded = m_str.zfill(2)
    target_m_unpadded = str(int(m_str))
    
    if prefix.lower() == "tave":
        for k, v in data_dict.items():
            k_lower = str(k).strip().lower()
            if k_lower in [f"tave{target_m_padded}", f"tave_{target_m_padded}", f"tave{target_m_unpadded}"]:
                try: 
                    return float(v)
                except: 
                    pass
                    
    elif prefix.lower() == "ppt":
        for k, v in data_dict.items():
            k_lower = normalize_key(str(k).strip())
            valid_keys = [
                f"ppt_{target_m_padded}", 
                f"ppt{target_m_padded}", 
                f"ppt_{target_m_unpadded}", 
                f"ppt{target_m_unpadded}"
            ]
            if k_lower in valid_keys:
                try: 
                    return float(v)
                except: 
                    pass
                    
    return None

def map_api_to_canonical(api_dict, prefix="Y_"):
    mapped_dict = {}
    canonical_lower = {c.lower(): c for c in CANONICAL_COLUMNS if c.startswith(prefix)}
    
    for k, v in api_dict.items():
        k_str = normalize_key(str(k).strip())
        exact_match = f"{prefix}{k_str}".lower()
        
        if exact_match in canonical_lower:
            mapped_dict[canonical_lower[exact_match]] = v
            continue
            
        m = re.match(r"^([a-z0-9_]+?)_?(\d{1,2})$", k_str)
        if m:
            base_var = m.group(1).rstrip('_') 
            num = m.group(2).zfill(2)
            try1 = f"{prefix}{base_var}{num}".lower()
            try2 = f"{prefix}{base_var}_{num}".lower()
            
            if try1 in canonical_lower: 
                mapped_dict[canonical_lower[try1]] = v
            elif try2 in canonical_lower: 
                mapped_dict[canonical_lower[try2]] = v
                
    return mapped_dict

def thin_and_cap_data(df, target_limit, max_per_year, distribute_by_decade=False):
    valid_df = df.dropna(subset=['Latitude', 'Longitude', 'Year', 'DOY', 'URL']).copy()
    valid_df = valid_df[valid_df['URL'].str.strip() != '']
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    
    if distribute_by_decade:
        valid_df['Decade'] = (valid_df['Year'] // 10) * 10
        capped_df = valid_df.groupby('Year').head(max_per_year).copy()
        decade_bins = {d: group.to_dict('records') for d, group in capped_df.groupby('Decade')}
        decades = sorted(list(decade_bins.keys()))
        
        selected_records = []
        while len(selected_records) < target_limit and any(len(lst) > 0 for lst in decade_bins.values()):
            for d in decades:
                if len(selected_records) >= target_limit: 
                    break
                if len(decade_bins[d]) > 0: 
                    selected_records.append(decade_bins[d].pop(0))
                    
        res_df = pd.DataFrame(selected_records)
        if not res_df.empty and 'Decade' in res_df.columns: 
            res_df = res_df.drop(columns=['Decade'])
        return res_df
        
    else:
        current_max = max_per_year
        capped_df = valid_df.groupby('Year').head(current_max).sort_values('Year')
        
        while len(capped_df) < target_limit and current_max < 50 and len(capped_df) < len(valid_df):
            current_max += 1
            capped_df = valid_df.groupby('Year').head(current_max).sort_values('Year')
            
        unique_years = capped_df['Year'].unique()
        selected_years = []
        last_y = -999
        
        for y in unique_years:
            if y - last_y >= 2: 
                selected_years.append(y)
                last_y = y
                
        spaced_df = capped_df[capped_df['Year'].isin(selected_years)]
        
        if len(spaced_df) < target_limit:
            remaining_df = capped_df[~capped_df['Year'].isin(selected_years)]
            spaced_df = pd.concat([spaced_df, remaining_df.head(target_limit - len(spaced_df))])
            
        return spaced_df.head(target_limit).sample(frac=1).reset_index(drop=True)

# ==========================================
#        THE CORE ENRICHMENT PIPELINE
# ==========================================

# ==========================================
#   PROJECT COUNTIES (Wasatch & Uinta)
# ==========================================
# Official US Census county boundaries for the project's counties, taken from the open
# "us-atlas" dataset (a simplified version of the Census cartographic boundary files).
#   Utah:    Rich, Cache, Box Elder, Weber, Morgan, Davis, Salt Lake, Summit,
#            Duchesne, Wasatch, Utah, Juab, Sanpete
#   Idaho:   Caribou, Franklin, Bear Lake
#   Wyoming: Lincoln
# Each county is a list of rings; each ring is a list of (latitude, longitude) corners.
# The app merges the chosen counties into one outer outline, so the borders between two
# chosen counties never matter. The boundaries are simplified, so an outer line can be off
# by a kilometre or two -- the "edge margin" setting in the sidebar covers that.
import math

PROJECT_COUNTY_RINGS = {
    "Box Elder, UT": [[(41.9933, -114.0411), (41.9882, -113.8222), (41.9985, -113.0002), (42.0011, -112.1926), (41.9968, -112.1639), (41.928, -112.1209), (41.8653, -112.0527), (41.7949, -112.0096), (41.7004, -112.0527), (41.6626, -112.0168), (41.5922, -112.006), (41.5337, -111.9809), (41.5526, -111.945), (41.532, -111.9127), (41.4745, -111.8768), (41.4607, -111.9163), (41.4264, -111.884), (41.4126, -111.9163), (41.4203, -111.9737), (41.3654, -111.9629), (41.3362, -112.0347), (41.3362, -112.2393), (41.0767, -112.4942), (41.0003, -112.7992), (41.0003, -113.2012), (41.0003, -114.0411)]],
    "Cache, UT": [[(41.9968, -112.1639), (41.9976, -112.1101), (41.9994, -111.5071), (41.928, -111.4712), (41.9134, -111.5035), (41.8525, -111.5071), (41.8172, -111.4604), (41.7365, -111.4425), (41.7124, -111.4174), (41.6763, -111.4245), (41.6592, -111.403), (41.6179, -111.4281), (41.5535, -111.4317), (41.5037, -111.4784), (41.4229, -111.5107), (41.4221, -111.6184), (41.4075, -111.6543), (41.4169, -111.6973), (41.3757, -111.726), (41.3688, -111.7619), (41.392, -111.7978), (41.3989, -111.8696), (41.4264, -111.884), (41.4607, -111.9163), (41.4745, -111.8768), (41.532, -111.9127), (41.5526, -111.945), (41.5337, -111.9809), (41.5922, -112.006), (41.6626, -112.0168), (41.7004, -112.0527), (41.7949, -112.0096), (41.8653, -112.0527), (41.928, -112.1209)]],
    "Davis, UT": [[(41.1394, -111.8553), (41.0664, -111.8517), (41.0226, -111.8373), (40.959, -111.7763), (40.9599, -111.8158), (40.8723, -111.7691), (40.8611, -111.7404), (40.8216, -111.8696), (40.8224, -111.945), (40.8869, -111.9701), (40.9221, -111.9593), (40.9212, -112.0096), (40.8465, -112.1496), (40.7692, -112.2608), (41.0767, -112.4942), (41.1532, -112.1855), (41.1523, -111.9701), (41.1351, -111.9127)]],
    "Duchesne, UT": [[(40.6781, -110.9005), (40.7262, -110.8933), (40.7125, -110.8036), (40.7477, -110.7498), (40.74, -110.6564), (40.7692, -110.6241), (40.7572, -110.6062), (40.7614, -110.5236), (40.7795, -110.4841), (40.7889, -110.3729), (40.8233, -110.3406), (40.8336, -110.2939), (40.8207, -110.1611), (40.8078, -110.1001), (40.8138, -109.9996), (40.8096, -109.9781), (39.8062, -109.9781), (39.807, -110.5164), (39.8113, -110.5164), (39.813, -110.8574), (39.8998, -110.8574), (39.8998, -110.8933), (40.0828, -110.8969), (40.4316, -110.8969), (40.4316, -110.9041)]],
    "Juab, UT": [[(39.9058, -114.0483), (39.905, -113.2479), (39.9041, -112.3434), (39.9445, -112.3398), (39.9393, -112.2716), (39.9892, -112.225), (40.0115, -112.1783), (39.9737, -112.1101), (39.929, -112.1029), (39.8938, -112.0527), (39.8654, -112.0814), (39.8294, -112.067), (39.7821, -112.0886), (39.777, -112.0455), (39.807, -111.9701), (39.8947, -111.9091), (39.9247, -111.8337), (39.9479, -111.8265), (39.8637, -111.7368), (39.8105, -111.7655), (39.813, -111.6435), (39.7838, -111.622), (39.7838, -111.5861), (39.74, -111.5861), (39.6945, -111.7189), (39.46, -111.7117), (39.46, -111.7476), (39.3827, -111.7476), (39.3818, -111.9235), (39.3663, -111.9593), (39.3371, -111.9593), (39.3371, -112.0168), (39.3148, -112.0168), (39.3294, -112.0706), (39.3294, -112.1891), (39.49, -112.1926), (39.5536, -112.2106), (39.5527, -112.8926), (39.5527, -113.815), (39.5442, -113.815), (39.5424, -114.0483)]],
    "Morgan, UT": [[(41.3611, -111.421), (41.374, -111.3851), (41.3576, -111.3312), (41.2966, -111.263), (41.2253, -111.281), (41.215, -111.2235), (41.1643, -111.2415), (41.1437, -111.2666), (41.13, -111.3671), (41.087, -111.3815), (41.0819, -111.4461), (41.063, -111.5071), (40.9934, -111.5932), (40.9582, -111.5789), (40.9539, -111.525), (40.9195, -111.5286), (40.8585, -111.4784), (40.819, -111.5179), (40.7915, -111.5179), (40.7803, -111.5538), (40.8113, -111.6112), (40.7992, -111.6399), (40.8499, -111.665), (40.8611, -111.7404), (40.8723, -111.7691), (40.9599, -111.8158), (40.959, -111.7763), (41.0226, -111.8373), (41.0664, -111.8517), (41.1394, -111.8553), (41.1695, -111.8768), (41.1978, -111.8588), (41.2176, -111.8014), (41.2142, -111.7476), (41.1875, -111.7189), (41.1858, -111.6578), (41.2099, -111.6112), (41.2382, -111.5968), (41.2271, -111.4963), (41.2898, -111.4389), (41.3155, -111.4497)]],
    "Rich, UT": [[(41.9994, -111.5071), (42.0019, -111.0477), (41.5801, -111.0441), (41.252, -111.0477), (41.2073, -111.1015), (41.1429, -111.2343), (41.1437, -111.2666), (41.1643, -111.2415), (41.215, -111.2235), (41.2253, -111.281), (41.2966, -111.263), (41.3576, -111.3312), (41.374, -111.3851), (41.3611, -111.421), (41.3843, -111.4784), (41.4229, -111.5107), (41.5037, -111.4784), (41.5535, -111.4317), (41.6179, -111.4281), (41.6592, -111.403), (41.6763, -111.4245), (41.7124, -111.4174), (41.7365, -111.4425), (41.8172, -111.4604), (41.8525, -111.5071), (41.9134, -111.5035), (41.928, -111.4712)]],
    "Salt Lake, UT": [[(40.7692, -112.2608), (40.8465, -112.1496), (40.9212, -112.0096), (40.9221, -111.9593), (40.8869, -111.9701), (40.8224, -111.945), (40.8216, -111.8696), (40.8611, -111.7404), (40.8499, -111.665), (40.7992, -111.6399), (40.7735, -111.6507), (40.6987, -111.5932), (40.6669, -111.604), (40.6094, -111.5538), (40.5767, -111.5932), (40.5682, -111.6363), (40.5312, -111.683), (40.5295, -111.7619), (40.4754, -111.8373), (40.4719, -111.884), (40.4247, -111.945), (40.4144, -111.9809), (40.4642, -112.0383), (40.4393, -112.1137), (40.4702, -112.1424), (40.4668, -112.1711), (40.6042, -112.1747), (40.6292, -112.1962), (40.7159, -112.207)]],
    "Sanpete, UT": [[(39.813, -111.6435), (39.813, -111.2487), (39.7039, -111.2487), (39.4668, -111.2487), (39.4668, -111.3025), (39.2942, -111.2989), (39.0322, -111.2989), (39.033, -111.8517), (39.0459, -111.8517), (39.0459, -112.0132), (39.3148, -112.0168), (39.3371, -112.0168), (39.3371, -111.9593), (39.3663, -111.9593), (39.3818, -111.9235), (39.3827, -111.7476), (39.46, -111.7476), (39.46, -111.7117), (39.6945, -111.7189), (39.74, -111.5861), (39.7838, -111.5861), (39.7838, -111.622)]],
    "Summit, UT": [[(40.7992, -111.6399), (40.8113, -111.6112), (40.7803, -111.5538), (40.7915, -111.5179), (40.819, -111.5179), (40.8585, -111.4784), (40.9195, -111.5286), (40.9539, -111.525), (40.9582, -111.5789), (40.9934, -111.5932), (41.063, -111.5071), (41.0819, -111.4461), (41.087, -111.3815), (41.13, -111.3671), (41.1437, -111.2666), (41.1429, -111.2343), (41.2073, -111.1015), (41.252, -111.0477), (40.9977, -111.0477), (40.996, -110.7139), (40.9951, -110.2365), (40.9977, -110.0498), (40.9977, -109.9996), (40.8138, -109.9996), (40.8078, -110.1001), (40.8207, -110.1611), (40.8336, -110.2939), (40.8233, -110.3406), (40.7889, -110.3729), (40.7795, -110.4841), (40.7614, -110.5236), (40.7572, -110.6062), (40.7692, -110.6241), (40.74, -110.6564), (40.7477, -110.7498), (40.7125, -110.8036), (40.7262, -110.8933), (40.6781, -110.9005), (40.6807, -110.9328), (40.5931, -110.9759), (40.5776, -111.0261), (40.5948, -111.08), (40.5905, -111.1195), (40.5484, -111.1518), (40.5871, -111.2666), (40.6231, -111.3276), (40.6292, -111.3671), (40.691, -111.3958), (40.6438, -111.4748), (40.5991, -111.4891), (40.6094, -111.5538), (40.6669, -111.604), (40.6987, -111.5932), (40.7735, -111.6507)]],
    "Utah, UT": [[(40.4668, -112.1711), (40.4702, -112.1424), (40.4393, -112.1137), (40.4642, -112.0383), (40.4144, -111.9809), (40.4247, -111.945), (40.4719, -111.884), (40.4754, -111.8373), (40.5295, -111.7619), (40.5312, -111.683), (40.5682, -111.6363), (40.5767, -111.5932), (40.5475, -111.5681), (40.4865, -111.5753), (40.4496, -111.622), (40.3663, -111.5538), (40.362, -111.482), (40.3216, -111.4389), (40.2898, -111.4425), (40.2967, -111.3743), (40.2864, -111.3133), (40.2426, -111.263), (40.222, -111.2738), (40.167, -111.2271), (40.1249, -111.22), (40.0553, -111.2487), (40.0476, -111.2056), (39.9909, -111.1661), (39.9685, -111.1195), (39.9471, -111.1302), (39.9436, -111.0836), (39.8998, -111.0836), (39.8998, -110.8933), (39.8998, -110.8574), (39.813, -110.8574), (39.813, -111.2487), (39.813, -111.6435), (39.8105, -111.7655), (39.8637, -111.7368), (39.9479, -111.8265), (39.9247, -111.8337), (39.8947, -111.9091), (39.807, -111.9701), (39.777, -112.0455), (39.7821, -112.0886), (39.8294, -112.067), (39.8654, -112.0814), (39.8938, -112.0527), (39.929, -112.1029), (39.9737, -112.1101), (40.0115, -112.1783), (40.0596, -112.1819), (40.1077, -112.1496), (40.1541, -112.1747), (40.173, -112.146), (40.2116, -112.1496), (40.264, -112.1926), (40.3259, -112.1962), (40.3362, -112.1747), (40.3826, -112.1998), (40.4548, -112.2142)]],
    "Wasatch, UT": [[(40.5767, -111.5932), (40.6094, -111.5538), (40.5991, -111.4891), (40.6438, -111.4748), (40.691, -111.3958), (40.6292, -111.3671), (40.6231, -111.3276), (40.5871, -111.2666), (40.5484, -111.1518), (40.5905, -111.1195), (40.5948, -111.08), (40.5776, -111.0261), (40.5931, -110.9759), (40.6807, -110.9328), (40.6781, -110.9005), (40.4316, -110.9041), (40.4316, -110.8969), (40.0828, -110.8969), (39.8998, -110.8933), (39.8998, -111.0836), (39.9436, -111.0836), (39.9471, -111.1302), (39.9685, -111.1195), (39.9909, -111.1661), (40.0476, -111.2056), (40.0553, -111.2487), (40.1249, -111.22), (40.167, -111.2271), (40.222, -111.2738), (40.2426, -111.263), (40.2864, -111.3133), (40.2967, -111.3743), (40.2898, -111.4425), (40.3216, -111.4389), (40.362, -111.482), (40.3663, -111.5538), (40.4496, -111.622), (40.4865, -111.5753), (40.5475, -111.5681)]],
    "Weber, UT": [[(41.0767, -112.4942), (41.3362, -112.2393), (41.3362, -112.0347), (41.3654, -111.9629), (41.4203, -111.9737), (41.4126, -111.9163), (41.4264, -111.884), (41.3989, -111.8696), (41.392, -111.7978), (41.3688, -111.7619), (41.3757, -111.726), (41.4169, -111.6973), (41.4075, -111.6543), (41.4221, -111.6184), (41.4229, -111.5107), (41.3843, -111.4784), (41.3611, -111.421), (41.3155, -111.4497), (41.2898, -111.4389), (41.2271, -111.4963), (41.2382, -111.5968), (41.2099, -111.6112), (41.1858, -111.6578), (41.1875, -111.7189), (41.2142, -111.7476), (41.2176, -111.8014), (41.1978, -111.8588), (41.1695, -111.8768), (41.1394, -111.8553), (41.1351, -111.9127), (41.1523, -111.9701), (41.1532, -112.1855)]],
    "Bear Lake, ID": [[(42.5139, -111.0477), (42.0019, -111.0477), (41.9994, -111.5071), (42.0689, -111.5143), (42.099, -111.6004), (42.142, -111.5861), (42.154, -111.6076), (42.2072, -111.6255), (42.2459, -111.6148), (42.2768, -111.5609), (42.288, -111.5861), (42.3258, -111.5896), (42.3516, -111.5538), (42.3911, -111.6004), (42.4151, -111.6004), (42.4297, -111.5789), (42.5698, -111.6363), (42.5904, -111.6004), (42.5904, -111.3671), (42.5526, -111.281), (42.6033, -111.2415), (42.5723, -111.2235), (42.4787, -111.2343), (42.4736, -111.1697), (42.501, -111.062)]],
    "Caribou, ID": [[(43.0225, -112.0634), (43.0199, -111.5896), (43.0191, -111.0441), (42.7227, -111.0441), (42.5139, -111.0477), (42.501, -111.062), (42.4736, -111.1697), (42.4787, -111.2343), (42.5723, -111.2235), (42.6033, -111.2415), (42.5526, -111.281), (42.5904, -111.3671), (42.5904, -111.6004), (42.5698, -111.6363), (42.4297, -111.5789), (42.4151, -111.6004), (42.416, -111.8768), (42.489, -111.884), (42.5242, -111.9163), (42.617, -111.9163), (42.6359, -111.945), (42.6746, -111.945), (42.7184, -111.9701), (42.7184, -112.0563), (42.6995, -112.1065), (42.751, -112.1388), (42.7905, -112.146), (42.8309, -112.1209), (42.8953, -112.1496), (42.9443, -112.128), (42.9503, -112.0814), (42.9735, -112.0886), (42.9907, -112.0527)]],
    "Franklin, ID": [[(42.2854, -112.1245), (42.2854, -111.9845), (42.2562, -111.9845), (42.2562, -111.8983), (42.416, -111.8947), (42.416, -111.8768), (42.4151, -111.6004), (42.3911, -111.6004), (42.3516, -111.5538), (42.3258, -111.5896), (42.288, -111.5861), (42.2768, -111.5609), (42.2459, -111.6148), (42.2072, -111.6255), (42.154, -111.6076), (42.142, -111.5861), (42.099, -111.6004), (42.0689, -111.5143), (41.9994, -111.5071), (41.9976, -112.1101), (42.1703, -112.1101), (42.1694, -112.0706), (42.2382, -112.0706)]],
    "Lincoln, WY": [[(42.5139, -111.0477), (42.7227, -111.0441), (43.0191, -111.0441), (43.3154, -111.0441), (43.3137, -110.8144), (43.2355, -110.8144), (43.2355, -110.5775), (43.1264, -110.5775), (43.1264, -110.5882), (42.7794, -110.5882), (42.7794, -110.6241), (42.7098, -110.6241), (42.7098, -110.5918), (42.4486, -110.5918), (42.4486, -110.5416), (42.2777, -110.5416), (42.2768, -110.2078), (42.2708, -110.0534), (42.0303, -110.0534), (42.0071, -110.0463), (41.5784, -110.0463), (41.5793, -110.5811), (41.5801, -111.0441), (42.0019, -111.0477)]],
}

def _county_bbox(rings):
    lats = [p[0] for ring in rings for p in ring]
    lons = [p[1] for ring in rings for p in ring]
    return min(lats), max(lats), min(lons), max(lons)

PROJECT_COUNTY_BBOX = {name: _county_bbox(rings) for name, rings in PROJECT_COUNTY_RINGS.items()}

def point_in_polygon(lat, lon, corners):
    """Ray-casting test: is (lat, lon) inside this ring of (lat, lon) corners?"""
    inside = False
    j = len(corners) - 1
    for i in range(len(corners)):
        lat_i, lon_i = corners[i]
        lat_j, lon_j = corners[j]
        if (lat_i > lat) != (lat_j > lat):
            lon_cross = (lon_j - lon_i) * (lat - lat_i) / (lat_j - lat_i) + lon_i
            if lon < lon_cross:
                inside = not inside
        j = i
    return inside

def _point_in_rings(lat, lon, rings):
    inside = False
    for ring in rings:
        if point_in_polygon(lat, lon, ring):
            inside = not inside
    return inside

def _km_to_rings(lat, lon, rings):
    """Approximate distance (km) from a point to the nearest county-line segment."""
    kx = 111.32 * math.cos(math.radians(lat))
    ky = 110.57
    best = float("inf")
    for ring in rings:
        n = len(ring)
        for i in range(n):
            lat_a, lon_a = ring[i]
            lat_b, lon_b = ring[(i + 1) % n]
            ax, ay = (lon_a - lon) * kx, (lat_a - lat) * ky
            bx, by = (lon_b - lon) * kx, (lat_b - lat) * ky
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / seg2))
            d = math.hypot(ax + t * dx, ay + t * dy)
            if d < best:
                best = d
    return best

def merge_outline(selected_names):
    """Combine the chosen counties into one outer outline. A border shared by two selected
    counties appears twice (once per county) and cancels out; only the outer edge is left.
    Returns a list of rings (several if the counties aren't all touching, or if a hole is
    left in the middle)."""
    edge_count = {}
    for name in selected_names:
        for ring in PROJECT_COUNTY_RINGS[name]:
            n = len(ring)
            for i in range(n):
                p, q = ring[i], ring[(i + 1) % n]
                if p == q:
                    continue
                key = (p, q) if p <= q else (q, p)
                edge_count[key] = edge_count.get(key, 0) + 1
    boundary = [k for k, c in edge_count.items() if c % 2 == 1]
    neighbours = {}
    for p, q in boundary:
        neighbours.setdefault(p, []).append(q)
        neighbours.setdefault(q, []).append(p)
    used = set()
    rings = []
    for p, q in boundary:
        if (p, q) in used:
            continue
        used.add((p, q))
        ring, cur = [p], q
        while cur != p:
            ring.append(cur)
            for cand in neighbours[cur]:
                k = (cur, cand) if cur <= cand else (cand, cur)
                if k not in used:
                    used.add(k)
                    cur = cand
                    break
            else:
                break
        if len(ring) >= 3:
            rings.append(ring)
    return rings

def build_region(selected_names, margin_km):
    """Bundle the chosen counties with their merged outline and the search box
    (padded by the edge margin) that gets sent to GBIF / iNaturalist."""
    boxes = [PROJECT_COUNTY_BBOX[n] for n in selected_names]
    min_lat = min(b[0] for b in boxes)
    max_lat = max(b[1] for b in boxes)
    min_lon = min(b[2] for b in boxes)
    max_lon = max(b[3] for b in boxes)
    pad_lat = margin_km / 110.57
    pad_lon = margin_km / (111.32 * math.cos(math.radians((min_lat + max_lat) / 2)))
    return {
        "counties": {n: PROJECT_COUNTY_RINGS[n] for n in selected_names},
        "outline": merge_outline(selected_names),
        "bbox": (round(min_lat - pad_lat, 4), round(max_lat + pad_lat, 4),
                 round(min_lon - pad_lon, 4), round(max_lon + pad_lon, 4)),
        "margin_km": margin_km,
    }

def _name_county(lat, lon, counties):
    """Which of these counties holds the point? (nearest one if it sits just outside them all)"""
    for name, rings in counties.items():
        b = PROJECT_COUNTY_BBOX[name]
        if b[0] <= lat <= b[1] and b[2] <= lon <= b[3] and _point_in_rings(lat, lon, rings):
            return name
    best_name, best_d = None, None
    for name, rings in counties.items():
        d = _km_to_rings(lat, lon, rings)
        if best_d is None or d < best_d:
            best_name, best_d = name, d
    return best_name

def locate_in_region(lat, lon, region):
    """Keep-or-skip test. Returns the county name if the point is inside the combined outline
    (or within the edge margin of it), otherwise None. Borders between two selected counties
    play no part -- only the outer outline decides."""
    if not _point_in_rings(lat, lon, region["outline"]):
        margin = region["margin_km"]
        if margin <= 0:
            return None
        b = region["bbox"]
        if lat < b[0] - 0.2 or lat > b[1] + 0.2 or lon < b[2] - 0.3 or lon > b[3] + 0.3:
            return None
        if _km_to_rings(lat, lon, region["outline"]) > margin:
            return None
    return _name_county(lat, lon, region["counties"])

def county_label(lat, lon):
    """Project county that holds the point, or None if it's outside all 17 (used for the County column)."""
    for name, rings in PROJECT_COUNTY_RINGS.items():
        b = PROJECT_COUNTY_BBOX[name]
        if b[0] <= lat <= b[1] and b[2] <= lon <= b[3] and _point_in_rings(lat, lon, rings):
            return name
    return None

# ==========================================
#   DUPLICATE SKIPPING + LAST-FETCH SUMMARY
# ==========================================

def _normalize_barcode(v):
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "<na>"):
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if s.isdigit():
        # Spreadsheets silently drop leading zeros, so compare without them.
        s = s.lstrip("0") or "0"
    return s

def _record_keys(df):
    """One identity key per record: source + barcode + year + day-of-year.
    Falls back to source + URL when a record has no barcode."""
    keys = []
    for src, bc, yr, doy, url in zip(df["Data_Source"], df["Barcode"], df["Year"], df["DOY"], df["URL"]):
        b = _normalize_barcode(bc)
        if b:
            keys.append(f"{str(src).strip()}|{b}|{safe_int(yr)}|{safe_int(doy)}")
        else:
            keys.append(f"{str(src).strip()}|{str(url).strip()}")
    return pd.Series(keys, index=df.index)

def load_existing_records_for_dedupe():
    """Fresh read of Master + Submissions so other collectors' latest saves count too."""
    if not os.path.exists(DB_FILE):
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.read_csv(DB_FILE)

def drop_already_downloaded(raw_df, notes):
    """Remove records already stored in the database, plus repeats inside this fetch.
    Returns (new_df, n_already_in_database, n_repeated_in_batch)."""
    raw_df = raw_df.copy()
    raw_df["_dedupe_key"] = _record_keys(raw_df)

    before = len(raw_df)
    raw_df = raw_df.drop_duplicates(subset="_dedupe_key", keep="first")
    n_batch = before - len(raw_df)

    n_db = 0
    try:
        existing = load_existing_records_for_dedupe()
        if not existing.empty:
            existing_keys = set(_record_keys(existing))
            is_dupe = raw_df["_dedupe_key"].isin(existing_keys)
            n_db = int(is_dupe.sum())
            raw_df = raw_df[~is_dupe]
    except Exception as e:
        notes.append(
            f"Couldn't check the database for existing records ({type(e).__name__}), "
            "so duplicates were NOT skipped this time."
        )
    return raw_df.drop(columns="_dedupe_key").reset_index(drop=True), n_db, n_batch

def render_fetch_summary():
    """Show the most recent fetch's record counts. Kept in session_state so it
    survives the automatic page refresh that follows a successful save."""
    summary = st.session_state.get("last_fetch_summary")
    if not summary:
        return
    st.sidebar.markdown(f"**📊 Last fetch summary** ({summary.get('when', '')})")
    for line in summary["lines"]:
        st.sidebar.markdown(f"- {line}")
    for note in summary["notes"]:
        st.sidebar.warning(note)
    if st.sidebar.button("Dismiss summary", key="dismiss_fetch_summary"):
        st.session_state.pop("last_fetch_summary", None)
        st.rerun()

def pipeline_enrich_and_save(raw_df, target_limit, max_per_year=3, distribute_by_decade=False, fetch_stats=None):
    summary = {
        "lines": list((fetch_stats or {}).get("lines", [])),
        "notes": list((fetch_stats or {}).get("notes", [])),
    }

    def _publish():
        summary["when"] = datetime.now().strftime("%H:%M:%S")
        st.session_state["last_fetch_summary"] = summary

    if raw_df.empty:
        _publish()
        st.sidebar.warning("No records found (or none passed the filters).")
        return

    # --- Skip specimens that are already stored in the shared database ---
    raw_df, n_dupes_db, n_dupes_batch = drop_already_downloaded(raw_df, summary["notes"])
    if n_dupes_db:
        summary["lines"].append(f"{n_dupes_db:,} already in the database (skipped)")
    if n_dupes_batch:
        summary["lines"].append(f"{n_dupes_batch:,} repeated within this fetch (skipped)")
    summary["lines"].append(f"{len(raw_df):,} new records available")

    if raw_df.empty:
        summary["notes"].append("Every usable record from this search is already in the database, so nothing was added.")
        _publish()
        st.sidebar.info("Nothing new to add -- all matching records are already in the database.")
        return

    st.sidebar.text("Applying de-clustering & sample distribution...")
    cleaned_df = thin_and_cap_data(
        raw_df,
        target_limit=target_limit,
        max_per_year=max_per_year,
        distribute_by_decade=distribute_by_decade
    )
    summary["lines"].append(
        f"{len(cleaned_df):,} selected after thinning "
        f"(limit {target_limit}, max {max_per_year} per year"
        f"{', spread across decades' if distribute_by_decade else ''})"
    )

    if cleaned_df.empty:
        _publish()
        st.sidebar.warning("No valid records remained after filtering.")
        return

    records = []
    progress_bar = st.sidebar.progress(0.0)
    status_text = st.sidebar.empty()
    alert_placeholder = st.sidebar.empty() 
    
    # Define it here so it always exists globally for the function!
    sys_error = None 
    
    for count, (idx, row) in enumerate(cleaned_df.iterrows()):
        progress_bar.progress(count / len(cleaned_df))
        
        row_dict = {
            col: row.get(col, np.nan) 
            for col in CANONICAL_COLUMNS 
            if not (col.startswith("Y_") or col.startswith("N_") or col.endswith("_Anomaly"))
        }
        row_dict['Phenology_Scored'] = False
        
        lat = safe_float(row_dict.get('Latitude'))
        lon = safe_float(row_dict.get('Longitude'))
        year = safe_int(row_dict.get('Year'))
        doy = safe_int(row_dict.get('DOY'))
        el = safe_float(row_dict.get('Elevation'))
        
        if lat is not None and lon is not None and year is not None:
            # --- CACHE OPTIMIZATION: Normalize coords & resolve elevation ---
        # --- CACHE OPTIMIZATION: Normalize coords & resolve elevation ---
            if el is None or el == 0.0 or pd.isna(el):
                status_text.text(f"Fetching elevation... ({count+1}/{len(cleaned_df)})")
                
                # --- ADD THIS LINE TO SLOW IT DOWN ---
                time.sleep(1)
                
            # (Your existing code that gets elevation)
            lat, lon, el = normalize_location(lat, lon, el)
            
            # (The diagnostic check we just added)
            # --- ADD THIS SAFETY CHECK ---
            if el is None:
                st.sidebar.error(f"⚠️ Dropped: Open-Meteo Elevation failed for {lat}, {lon}. You might be rate-limited!")
                continue
                
            # -----------------------------
            
            row_dict['Latitude'] = lat
            row_dict['Longitude'] = lon
            row_dict['Elevation'] = el
            # ----------------------------------------------------------------
            
            if doy is not None:
                lat_rad = np.radians(lat)
                dec_rad = np.radians(23.45 * np.sin(2 * np.pi * (284 + doy) / 365.25))
                val_clip = np.clip(-np.tan(lat_rad) * np.tan(dec_rad), -1.0, 1.0)
                row_dict['Photoperiod_Hours'] = np.round((24.0 / np.pi) * np.arccos(val_clip), 2)
            
            climate_year = min(year, current_year - 1) 
            status_text.text(f"Fetching ClimateNA for Year_{climate_year}.ann... ({count+1}/{len(cleaned_df)})")
            
            year_data = get_climate_data(lat, lon, el, f"Year_{climate_year}.ann")
            norm_data = get_climate_data(lat, lon, el, "Normal_1961_1990")
            
            sys_error = None
            if isinstance(year_data, dict) and "error" in year_data and year_data.get("systemic"): 
                sys_error = year_data["error"]
            elif isinstance(norm_data, dict) and "error" in norm_data and norm_data.get("systemic"): 
                sys_error = norm_data["error"]
                
            if sys_error:
                alert_placeholder.error(f"🚨 **ClimateNA Download Limit Reached!**\n\n**Details:** {sys_error}\n\n*Saved the {len(records)} records successfully processed up to this point.*")
                break 
                
            if "error" in year_data or "error" in norm_data: 
                err_msg = year_data.get('error') if isinstance(year_data, dict) and 'error' in year_data else norm_data.get('error')
                st.sidebar.error(f"⚠️ Dropped: ClimateNA API blocked the request -> {err_msg}")
                continue
                
            if isinstance(year_data, dict):
                y_mapped = map_api_to_canonical(year_data, "Y_")
                row_dict.update(y_mapped)
                
            if isinstance(norm_data, dict):
                n_mapped = map_api_to_canonical(norm_data, "N_")
                row_dict.update(n_mapped)
            
            if doy is not None:
                ty, tm = calc_prior_3_months(climate_year, doy)
                if ty and tm:
                    min_y = min(ty)
                    prev_year_data = {}
                    
                    if min_y < climate_year: 
                        prev_year_data = get_climate_data(lat, lon, el, f"Year_{min_y}.ann")
                        if "error" in prev_year_data and prev_year_data.get("systemic"):
                            alert_placeholder.error(f"🚨 **ClimateNA Download Limit Reached!**\n\n*Saved {len(records)} successful records.*")
                            break
                        elif "error" in prev_year_data: 
                            continue 
                    
                    y_t_vals = []
                    n_t_vals = []
                    y_p_vals = []
                    n_p_vals = []
                    
                    for y_t, m_t in zip(ty, tm):
                        m_str = f"{m_t:02d}"
                        n_t = get_climate_val(norm_data, "tave", m_str)
                        n_p = get_climate_val(norm_data, "ppt", m_str)
                        
                        if n_t is not None: 
                            n_t_vals.append(n_t)
                        if n_p is not None: 
                            n_p_vals.append(n_p)
                        
                        target_data = year_data if y_t >= climate_year else prev_year_data
                        y_t_v = get_climate_val(target_data, "tave", m_str)
                        y_p_v = get_climate_val(target_data, "ppt", m_str)
                        
                        if y_t_v is not None: 
                            y_t_vals.append(y_t_v)
                        if y_p_v is not None: 
                            y_p_vals.append(y_p_v)
                    
                    if len(y_t_vals) == 3 and len(n_t_vals) == 3:
                        row_dict['Y_3Mo_prior_mean_Tave'] = round(sum(y_t_vals) / 3.0, 2)
                        row_dict['N_3Mo_prior_mean_Tave'] = round(sum(n_t_vals) / 3.0, 2)
                        row_dict['Tave_Anomaly'] = round(row_dict['Y_3Mo_prior_mean_Tave'] - row_dict['N_3Mo_prior_mean_Tave'], 2)
                    
                    if len(y_p_vals) == 3 and len(n_p_vals) == 3:
                        row_dict['Y_3Mo_prior_mean_PPT'] = round(sum(y_p_vals) / 3.0, 2)
                        row_dict['N_3Mo_prior_mean_PPT'] = round(sum(n_p_vals) / 3.0, 2)
                        row_dict['PPT_Anomaly'] = round(row_dict['Y_3Mo_prior_mean_PPT'] - row_dict['N_3Mo_prior_mean_PPT'], 2)

        records.append(row_dict)
        
    progress_bar.progress(1.0)

    not_saved = len(cleaned_df) - len(records)
    summary["lines"].append(f"{len(records):,} saved to the database")
    if not_saved > 0:
        summary["notes"].append(
            f"{not_saved} selected record(s) weren't saved because their elevation or ClimateNA lookup "
            "failed or hit a rate limit. Running the same fetch again later will pick them up -- "
            "records already saved are skipped."
        )
    if sys_error:
        summary["notes"].append(f"Processing stopped early: {sys_error}")
    _publish()

    if sys_error is None:
        status_text.text("Finished processing pipeline!")
        st.sidebar.success(f"Successfully processed {len(records)} records!")

    if records:
        final_new_df = pd.DataFrame(records)
        with st.spinner("Saving new records to the database file..."):
            master_df = pd.read_csv(DB_FILE)
            combined_df = pd.concat([master_df, final_new_df], ignore_index=True)
            save_with_ordered_columns(combined_df, DB_FILE)

        if sys_error is not None:
            render_fetch_summary()
            st.stop()
        else:
            time.sleep(2)
            st.rerun()

# ==========================================
#        SIDEBAR: DATA ENTRY
# ==========================================
st.sidebar.header("Data Entry & Ingestion")

with st.sidebar.expander("✏️ Manual Entry", expanded=False):
    with st.form("manual_entry_form"):
        st.markdown("**Add a New Specimen Record**")
        
        m_spp = st.text_input("Species Name:", placeholder="e.g., Lithospermum ruderale")
        m_date = st.date_input("Collection Date:", value=datetime.now().date())
        m_yr = m_date.year
        m_doy = m_date.timetuple().tm_yday
        
        c_m1, c_m2 = st.columns(2)
        with c_m1: 
            m_col = st.text_input("Collector:")
        with c_m2: 
            m_col_num = st.text_input("Col. Number:")
            
        c_m3, c_m4 = st.columns(2)
        with c_m3: 
            m_barcode = st.text_input("Barcode:")
        with c_m4: 
            m_elev = st.number_input("Elev. Override:", format="%.2f", value=0.0)
            
        c3, c4 = st.columns(2)
        with c3: 
            m_lat = st.number_input("Latitude:", format="%.5f", value=0.0)
        with c4: 
            m_lon = st.number_input("Longitude:", format="%.5f", value=0.0)
        
        if st.form_submit_button("Add & Process Record", use_container_width=True):
            if not m_spp or m_lat == 0.0 or m_lon == 0.0: 
                st.error("Species, Latitude, and Longitude are required!")
            else:
                new_record = [{
                    "Data_Source": "Manual Entry", 
                    "Collector": m_col, 
                    "Col_Number": m_col_num,
                    "Barcode": m_barcode, 
                    "Species": m_spp, 
                    "Year": m_yr, 
                    "DOY": m_doy,
                    "Latitude": m_lat, 
                    "Longitude": m_lon, 
                    "Elevation": m_elev if m_elev != 0.0 else np.nan,
                    "URL": "Manual", 
                    "Flowering": False, 
                    "Fruiting": False, 
                    "Vegetative": False
                }]
                pipeline_enrich_and_save(
                    pd.DataFrame(new_record), 
                    target_limit=1, 
                    max_per_year=1, 
                    distribute_by_decade=False
                )

with st.sidebar.expander("📍 Project Counties (Wasatch & Uinta)", expanded=False):
    st.caption(
        "Keeps only specimens located inside the selected counties, for both GBIF and iNaturalist, "
        "using official US Census county boundaries (simplified). The counties are merged into one "
        "outer outline, so borders between two selected counties never cause a miss. When it's on, "
        "the State/County text boxes below are ignored."
    )
    region_on = st.checkbox("Only keep specimens inside the selected counties", value=False, key="region_on")
    region_names = st.multiselect(
        "Counties:", options=list(PROJECT_COUNTY_RINGS.keys()),
        default=list(PROJECT_COUNTY_RINGS.keys()), key="region_counties"
    )
    region_margin = st.slider(
        "Edge margin (km):", min_value=0, max_value=15, value=3, key="region_margin",
        help="Also keep specimens up to this far outside the outer outline. The boundaries are "
             "simplified and specimen coordinates are often approximate, so a small margin avoids "
             "losing records right at the edge. Set 0 for the strict outline."
    )
    active_region = None
    region_blocked = False
    if not region_names:
        st.warning("Pick at least one county.")
        region_blocked = bool(region_on)
    else:
        preview_region = build_region(region_names, region_margin)
        st.caption(
            f"Search box sent to GBIF / iNaturalist: latitude {preview_region['bbox'][0]:.2f} to "
            f"{preview_region['bbox'][1]:.2f}, longitude {preview_region['bbox'][2]:.2f} to "
            f"{preview_region['bbox'][3]:.2f}"
        )
        if region_on:
            active_region = preview_region
        if st.checkbox("Show the outline on a map", key="region_preview"):
            try:
                map_lat, map_lon = [], []
                for ring in preview_region["outline"]:
                    closed_ring = list(ring) + [ring[0]]
                    map_lat += [p[0] for p in closed_ring] + [None]
                    map_lon += [p[1] for p in closed_ring] + [None]
                fig_region = go.Figure(go.Scattergeo(
                    lat=map_lat, lon=map_lon, mode="lines", fill="toself", hoverinfo="skip"
                ))
                fig_region.update_geos(
                    scope="usa", fitbounds="locations", resolution=50,
                    showsubunits=True, showlakes=True
                )
                fig_region.update_layout(height=320, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_region, use_container_width=True)
                n_rings = len(preview_region["outline"])
                st.caption(
                    "Outer outline of the selected counties."
                    + (f" It has {n_rings} separate pieces." if n_rings > 1 else "")
                )
            except Exception:
                st.caption("Map preview isn't available in this setup.")

with st.sidebar.expander("🌐 Fetch from GBIF", expanded=False):
    gbif_spp = st.text_input("Species Name (GBIF):", key="g_spp")

    
    col_yr1, col_yr2 = st.columns(2)
    with col_yr1: 
        g_start = st.number_input("Start Year:", min_value=1902, max_value=2026, value=1902, key="g_start_yr")
    with col_yr2: 
        g_end = st.number_input("End Year:", min_value=1902, max_value=current_year, value=current_year, key="g_end_yr")
        
    col_lim1, col_lim2 = st.columns(2)
    with col_lim1: 
        g_limit = st.number_input("Total Records:", min_value=5, max_value=200, value=25, step=5, key="g_limit_rec")
    with col_lim2: 
        g_max_yr = st.number_input("Max per Year:", min_value=1, max_value=20, value=3, key="g_max_yr")
    
    g_decade = st.checkbox("Spread evenly across decades (Round-Robin)", value=False, key="g_dec_toggle")
    g_states = st.text_input("State(s):", help="Comma-separated", key="g_states")
    g_counties = st.text_input("County(s):", help="Comma-separated", key="g_counties")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1: 
        g_emin = st.text_input("Min Elev (m):", key="g_emin")
    with col_e2: 
        g_emax = st.text_input("Max Elev (m):", key="g_emax")
    
    if region_blocked:
        st.warning("Fetching is paused: the project counties are switched on but none are selected (see 📍 Project Counties).")
    if st.button("Fetch & Process GBIF", type="primary", use_container_width=True, disabled=region_blocked):
        with st.spinner("Paginating GBIF..."):
            spp_encoded = urllib.parse.quote(gbif_spp)
            e_min = parse_elev(g_emin)
            e_max = parse_elev(g_emax)
            base_url = f"https://api.gbif.org/v1/occurrence/search?scientificName={spp_encoded}&year={g_start},{g_end}&hasCoordinate=true&basisOfRecord=PRESERVED_SPECIMEN"

            if g_states and not active_region:
                for s in g_states.split(','):
                    base_url += f"&stateProvince={urllib.parse.quote(s.strip())}"
            if g_counties and not active_region:
                for c in g_counties.split(','):
                    base_url += f"&county={urllib.parse.quote(c.strip())}"
            if e_min is not None or e_max is not None:
                base_url += f"&elevation={e_min if e_min else ''},{e_max if e_max else ''}"
            if active_region:
                a_min_lat, a_max_lat, a_min_lon, a_max_lon = active_region["bbox"]
                base_url += f"&decimalLatitude={a_min_lat},{a_max_lat}&decimalLongitude={a_min_lon},{a_max_lon}"

            raw_records = []
            offset = 0
            end_of_records = False
            max_to_fetch = max(g_limit * 10, 500)
            stats = {"lines": [], "notes": []}
            gbif_total = None
            n_scanned = 0
            n_no_image = 0
            n_no_date = 0
            n_outside_region = 0
            by_county = {}

            while len(raw_records) < max_to_fetch and not end_of_records and offset < 9000:
                try:
                    res = requests.get(base_url + f"&limit=300&offset={offset}", timeout=10)
                    if res.status_code == 200:
                        data = res.json()
                        results = data.get('results', [])
                        if gbif_total is None:
                            gbif_total = data.get('count')

                        if not results:
                            end_of_records = True
                            break

                        for obs in results:
                            n_scanned += 1
                            county_hit = None
                            o_lat = safe_float(obs.get('decimalLatitude'))
                            o_lon = safe_float(obs.get('decimalLongitude'))
                            if o_lat is not None and o_lon is not None:
                                if active_region:
                                    county_hit = locate_in_region(o_lat, o_lon, active_region)
                                else:
                                    county_hit = county_label(o_lat, o_lon)
                            if active_region and county_hit is None:
                                n_outside_region += 1
                                continue
                            media = obs.get('media', [])
                            rec_url = ""

                            for m in media:
                                if m.get('type') == 'StillImage' and m.get('identifier'):
                                    rec_url = m.get('identifier')
                                    break

                            if not rec_url:
                                n_no_image += 1
                                continue

                            rec_url = ensure_url_scheme(rec_url)
                            y = obs.get('year') or (int(obs['eventDate'][:4]) if obs.get('eventDate') else None)
                            m = obs.get('month')
                            d = obs.get('day')

                            if not y or not m or not d:
                                n_no_date += 1
                                continue

                            try:
                                doy = datetime(int(y), int(m), int(d)).timetuple().tm_yday
                            except:
                                n_no_date += 1
                                continue

                            raw_records.append({
                                "Data_Source": "GBIF Herbarium",
                                "Collector": obs.get('recordedBy', ''),
                                "Col_Number": obs.get('recordNumber', ''),
                                "Barcode": obs.get('catalogNumber', '') or obs.get('occurrenceID', ''),
                                "Species": obs.get('species', gbif_spp),
                                "Year": safe_int(y),
                                "DOY": doy,
                                "Latitude": obs.get('decimalLatitude'),
                                "Longitude": obs.get('decimalLongitude'),
                                "Elevation": obs.get('elevation', np.nan),
                                "URL": rec_url,
                                "County": county_hit or "",
                                "Flowering": False,
                                "Fruiting": False,
                                "Vegetative": False
                            })
                            if county_hit:
                                by_county[county_hit] = by_county.get(county_hit, 0) + 1

                        end_of_records = data.get('endOfRecords', True)
                        offset += 300
                    else:
                        stats["notes"].append(
                            f"GBIF returned HTTP {res.status_code} at offset {offset}, so the fetch stopped early "
                            "and later records weren't retrieved. Try again in a moment."
                        )
                        break
                except Exception as e:
                    stats["notes"].append(
                        f"The GBIF request failed at offset {offset} ({type(e).__name__}), so the fetch stopped early "
                        "and later records weren't retrieved. Try again in a moment."
                    )
                    break

            if not end_of_records and not stats["notes"]:
                if offset >= 9000:
                    stats["notes"].append(
                        "Stopped after scanning 9,000 GBIF records (the app's scan cap); more matching specimens exist. "
                        "Narrow the search with years, State or County to reach them."
                    )
                elif len(raw_records) >= max_to_fetch:
                    stats["notes"].append(
                        f"Stopped once {max_to_fetch:,} usable records were collected, to keep the fetch quick; "
                        "more matching specimens exist."
                    )

            if gbif_total is not None:
                stats["lines"].append(
                    f"GBIF reports {gbif_total:,} matching specimens with coordinates"
                    + (" inside the counties' search box" if active_region else "")
                )
            stats["lines"].append(f"{n_scanned:,} scanned")
            if n_outside_region:
                stats["lines"].append(f"{n_outside_region:,} skipped: outside the selected counties")
            if active_region and ((g_states or "").strip() or (g_counties or "").strip()):
                stats["notes"].append("The project counties are on, so your State/County text boxes were ignored for this fetch.")
            if n_no_image:
                stats["lines"].append(f"{n_no_image:,} skipped: no image")
            if n_no_date:
                stats["lines"].append(f"{n_no_date:,} skipped: no full collection date (year, month and day)")
            stats["lines"].append(f"{len(raw_records):,} usable records")
            if by_county:
                stats["lines"].append(
                    "Usable records by project county: "
                    + ", ".join(f"{k} {v}" for k, v in sorted(by_county.items(), key=lambda kv: -kv[1]))
                )

            pipeline_enrich_and_save(
                pd.DataFrame(raw_records),
                target_limit=g_limit,
                max_per_year=g_max_yr,
                distribute_by_decade=g_decade,
                fetch_stats=stats
            )

with st.sidebar.expander("📸 Fetch from iNaturalist", expanded=False):
    inat_spp = st.text_input("Species Name (iNat):", key="i_spp")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1: 
        i_start = st.number_input("Start Year:", min_value=1902, max_value=2026, value=1980, key="i_start_yr")
    with col_in2: 
        i_end = st.number_input("End Year:", min_value=1902, max_value=current_year, value=current_year, key="i_end_yr")
        
    col_ilim1, col_ilim2 = st.columns(2)
    with col_ilim1: 
        i_limit = st.number_input("Total Records:", min_value=5, max_value=200, value=25, step=5, key="i_limit_rec")
    with col_ilim2: 
        i_max_yr = st.number_input("Max per Year:", min_value=1, max_value=20, value=3, key="i_max_yr")
    
    i_decade = st.checkbox("Spread evenly across decades (Round-Robin)", value=False, key="i_dec_toggle")
    i_states = st.text_input("State(s):", key="i_states")
    i_counties = st.text_input("County(s):", key="i_counties")
    
    col_ie1, col_ie2 = st.columns(2)
    with col_ie1: 
        i_emin = st.text_input("Min Elev (m):", key="i_emin")
    with col_ie2: 
        i_emax = st.text_input("Max Elev (m):", key="i_emax")
        
    if region_blocked:
        st.warning("Fetching is paused: the project counties are switched on but none are selected (see 📍 Project Counties).")
    if st.button("Fetch & Process iNat", type="primary", use_container_width=True, disabled=region_blocked):
        with st.spinner(f"Paginating iNaturalist..."):
            spp_encoded = urllib.parse.quote(inat_spp)
            e_min = parse_elev(i_emin)
            e_max = parse_elev(i_emax)

            s_list = [s.strip().lower() for s in i_states.split(',')] if (i_states and not active_region) else []
            c_list = [c.strip().lower() for c in i_counties.split(',')] if (i_counties and not active_region) else []

            raw_records = []
            page = 1
            max_to_fetch = max(i_limit * 10, 500)
            stats = {"lines": [], "notes": []}
            inat_total = None
            exhausted = False
            n_scanned = 0
            n_no_uri = 0
            n_place = 0
            n_no_loc = 0
            n_elev_out = 0
            n_elev_fail = 0
            n_bad_date = 0
            n_outside_region = 0
            by_county = {}
            if active_region:
                a_min_lat, a_max_lat, a_min_lon, a_max_lon = active_region["bbox"]

            while len(raw_records) < max_to_fetch and page <= 15:
                try:
                    url = f"https://api.inaturalist.org/v1/observations?taxon_name={spp_encoded}&d1={i_start}-01-01&d2={i_end}-12-31&per_page=200&page={page}&quality_grade=research"

                    if active_region:
                        url += f"&nelat={a_max_lat}&nelng={a_max_lon}&swlat={a_min_lat}&swlng={a_min_lon}"
                    
                    # Mandatory delay to respect iNaturalist's API limits
                    time.sleep(1.0)

                    res = requests.get(url, timeout=10)

                    if res.status_code == 200:
                        payload = res.json()
                        results = payload.get('results', [])
                        if inat_total is None:
                            inat_total = payload.get('total_results')

                        if not results:
                            exhausted = True
                            break

                        for obs in results:
                            n_scanned += 1
                            county_hit = None
                            rec_url = obs.get('uri', '')

                            if not rec_url:
                                n_no_uri += 1
                                continue

                            rec_url = ensure_url_scheme(rec_url)
                            # place_guess can be null for obscured/unlabelled observations
                            pg = (obs.get('place_guess') or '').lower()

                            if s_list and not any(s in pg for s in s_list):
                                n_place += 1
                                continue
                            if c_list and not any(c in pg for c in c_list):
                                n_place += 1
                                continue

                            if obs.get('location') and obs.get('observed_on'):
                                lat_str, lon_str = obs['location'].split(',')
                                lat_flt = float(lat_str)
                                lon_flt = float(lon_str)
                                if active_region:
                                    county_hit = locate_in_region(lat_flt, lon_flt, active_region)
                                    if county_hit is None:
                                        n_outside_region += 1
                                        continue
                                else:
                                    county_hit = county_label(lat_flt, lon_flt)
                                fetched_elev = np.nan

                                if e_min is not None or e_max is not None:
                                    time.sleep(1)
                                    el = get_elevation(lat_flt, lon_flt)

                                    if el is None:
                                        n_elev_fail += 1
                                        continue

                                    if (e_min and el < e_min) or (e_max and el > e_max):
                                        n_elev_out += 1
                                        continue

                                    fetched_elev = el

                                try:
                                    dt = datetime.strptime(obs['observed_on'], "%Y-%m-%d")
                                    obs_year = dt.year
                                    obs_doy = dt.timetuple().tm_yday
                                except:
                                    n_bad_date += 1
                                    continue

                                raw_records.append({
                                    "Data_Source": "iNaturalist",
                                    "Collector": (obs.get('user') or {}).get('login', ''),
                                    "Col_Number": np.nan,
                                    "Barcode": str(obs.get('id', '')),
                                    "Species": (obs.get('taxon') or {}).get('name', inat_spp),
                                    "Year": obs_year,
                                    "DOY": obs_doy,
                                    "Latitude": lat_flt,
                                    "Longitude": lon_flt,
                                    "Elevation": fetched_elev,
                                    "URL": rec_url,
                                    "County": county_hit or "",
                                    "Flowering": False,
                                    "Fruiting": False,
                                    "Vegetative": False
                                })
                                if county_hit:
                                    by_county[county_hit] = by_county.get(county_hit, 0) + 1
                            else:
                                n_no_loc += 1

                        if len(results) < 200:
                            exhausted = True
                            break
                        page += 1
                    else:
                        stats["notes"].append(
                            f"iNaturalist returned HTTP {res.status_code} on page {page}, so the fetch stopped early "
                            "and later observations weren't retrieved. Try again in a moment."
                        )
                        break
                except Exception as e:
                    stats["notes"].append(
                        f"The iNaturalist fetch hit an error on page {page} ({type(e).__name__}: {e}), "
                        "so it stopped early and later observations weren't retrieved."
                    )
                    break

            if not exhausted and not stats["notes"]:
                if len(raw_records) >= max_to_fetch:
                    stats["notes"].append(
                        f"Stopped once {max_to_fetch:,} usable records were collected, to keep the fetch quick; "
                        "more matching observations exist."
                    )
                elif page > 15:
                    stats["notes"].append(
                        "Stopped after 15 pages (3,000 observations), the app's page cap; more matching observations "
                        "exist. Narrow the search with years, State or County to reach them."
                    )

            if inat_total is not None:
                stats["lines"].append(
                    f"iNaturalist reports {inat_total:,} matching research-grade observations"
                    + (" inside the counties' search box" if active_region else "")
                )
            stats["lines"].append(f"{n_scanned:,} scanned")
            for n, label in [
                (n_no_uri, "skipped: no record link"),
                (n_place, "skipped: outside your State/County text filter"),
                (n_no_loc, "skipped: no public location or date (often obscured or private)"),
                (n_outside_region, "skipped: outside the selected counties"),
                (n_elev_fail, "dropped: elevation lookup failed"),
                (n_elev_out, "skipped: outside your elevation range"),
                (n_bad_date, "skipped: unreadable date"),
            ]:
                if n:
                    stats["lines"].append(f"{n:,} {label}")
            stats["lines"].append(f"{len(raw_records):,} usable records")
            if by_county:
                stats["lines"].append(
                    "Usable records by project county: "
                    + ", ".join(f"{k} {v}" for k, v in sorted(by_county.items(), key=lambda kv: -kv[1]))
                )

            if active_region and ((i_states or "").strip() or (i_counties or "").strip()):
                stats["notes"].append("The project counties are on, so your State/County text boxes were ignored for this fetch.")
            
            if not raw_records:
                st.sidebar.warning("No records matched your search. If you used State/County filters, try clearing them!")

            pipeline_enrich_and_save(
                pd.DataFrame(raw_records),
                target_limit=i_limit,
                max_per_year=i_max_yr,
                distribute_by_decade=i_decade,
                fetch_stats=stats
            )

render_fetch_summary()

st.sidebar.write("---")
if st.sidebar.button("🗑️ Clear Entire Database"):
    pd.DataFrame(columns=CANONICAL_COLUMNS).to_csv(DB_FILE, index=False)
    st.rerun()

# ==========================================
#        MAIN UI: EXPLORER SETTINGS
# ==========================================
st.title("🌱 Phenology & Climate Dataset Builder")

# Dynamically fill Photoperiod for legacy records in the DB
df = pd.read_csv(DB_FILE)
if 'Photoperiod_Hours' not in df.columns:
    df['Photoperiod_Hours'] = np.nan
    
mask = df['Latitude'].notna() & df['DOY'].notna() & df['Photoperiod_Hours'].isna()

if mask.any():
    lat_rad = np.radians(df.loc[mask, 'Latitude'].astype(float))
    dec_rad = np.radians(23.45 * np.sin(2 * np.pi * (284 + df.loc[mask, 'DOY'].astype(float)) / 365.25))
    val_clip = np.clip(-np.tan(lat_rad) * np.tan(dec_rad), -1.0, 1.0)
    df.loc[mask, 'Photoperiod_Hours'] = np.round((24.0 / np.pi) * np.arccos(val_clip), 2)

with st.expander("⚙️ Global Analysis & Outlier Settings", expanded=True):
    if df.empty:
        st.info("Your database is empty. Fetch or add records first.")
        plot_df = pd.DataFrame()
        use_outlier_filter = False
    else:
        sources = df['Data_Source'].fillna('Unknown').unique().tolist()
        species = df['Species'].dropna().unique().tolist()
        
        c1, c2 = st.columns(2)
        with c1: 
            sel_src = st.multiselect("Filter Data Source:", sources, default=sources)
        with c2: 
            sel_spp = st.multiselect("Filter Species:", species, default=species)
        
        plot_df = df[df['Data_Source'].isin(sel_src) & df['Species'].isin(sel_spp)].copy()

        # Workspace Focus Checkbox
        st.markdown("---")
        st.markdown("**👤 Scope & Data Focus Control**")
        focus_collector = st.checkbox(
            "🔍 Focus active workspace/tabs on data being collected by a specific Collector", 
            value=False,
            help="When checked, the Active Data Ledger, Map View, Trends, and Rapid Scoring tabs will automatically filter to show ONLY data from your chosen collector."
        )
        
        if focus_collector:
            collectors = sorted(df['Collector'].dropna().astype(str).unique().tolist())
            selected_collector = st.selectbox("Choose Target Collector / User:", options=[""] + collectors, index=0)
            if selected_collector:
                plot_df = plot_df[plot_df['Collector'].astype(str) == selected_collector].copy()
        
        st.markdown("---")
        
        # Spatial Thinning Subsampler
        spatial_thin = st.checkbox("🌐 Apply Spatial Thinning (~1 km grid cell grouping)", value=False)
        
        if spatial_thin and not plot_df.empty:
            plot_df['lat_grid'] = np.round(plot_df['Latitude'].astype(float), 2)
            plot_df['lon_grid'] = np.round(plot_df['Longitude'].astype(float), 2)
            plot_df = plot_df.drop_duplicates(subset=['Species', 'Year', 'lat_grid', 'lon_grid']).drop(columns=['lat_grid', 'lon_grid'])
            
        for col in ['Year', 'DOY', 'Latitude', 'Longitude', 'Elevation', 'Photoperiod_Hours']:
            if col in plot_df.columns: 
                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                
        for col in [c for c in plot_df.columns if c.startswith('Y_') or c.startswith('N_')]:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

        num_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
        st.markdown("---")
        s_c1, s_c2, s_c3 = st.columns([1, 1, 2])
        
        if len(num_cols) >= 2:
            with s_c1: 
                selected_x = st.selectbox("X-Axis (For Trendline):", num_cols, index=num_cols.index('Year') if 'Year' in num_cols else 0)
            with s_c2: 
                selected_y = st.selectbox("Y-Axis (For Trendline):", num_cols, index=num_cols.index('DOY') if 'DOY' in num_cols else 1)
            with s_c3:
                use_outlier_filter = st.checkbox(f"Highlight Outliers based on {selected_y}", value=False)
                if use_outlier_filter:
                    std_devs = st.slider("Flag outside standard deviations:", 1.0, 5.0, 2.5, 0.1)
        else:
            use_outlier_filter = False
        
        plot_df['Is_Outlier'] = False
        if use_outlier_filter and len(plot_df.dropna(subset=[selected_y])) > 0:
            valid_y_df = plot_df.dropna(subset=[selected_y])
            mean_y = valid_y_df[selected_y].mean()
            std_y = valid_y_df[selected_y].std()
            plot_df['Is_Outlier'] = np.abs(plot_df[selected_y] - mean_y) > (std_devs * std_y)
            outlier_count = plot_df['Is_Outlier'].sum()
            st.info(f"Identified **{outlier_count}** outliers (outside {mean_y:.1f} ± {std_devs*std_y:.1f}).")
        
        plot_df['Map_Label'] = plot_df['Species'].fillna('Unknown') + plot_df['Is_Outlier'].apply(lambda x: ' 🔴 [OUTLIER]' if x else '')
        
        st.markdown("---")
        st.download_button(
            "📥 Download Filtered Subset (CSV)", 
            data=plot_df.to_csv(index=False).encode('utf-8'), 
            file_name="filtered_phenology_subset.csv", 
            mime="text/csv", 
            use_container_width=True
        )

# ==========================================
#        MAIN UI: TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗃️ Active Data Ledger", 
    "🗺️ Map View", 
    "📊 Trends & Outliers", 
    "🎯 Rapid Scoring", 
    "🤖 Machine Learning", 
    "🗄️ Full Master Database"
])

with tab1:
    st.info("💡 Editing your **active filtered workspace subset** here. Edits will sync cleanly back to the main database repository upon saving.")
    if plot_df.empty:
        st.warning("No records match your active search filters.")
    else:
        edited_active_df = st.data_editor(
            plot_df, 
            num_rows="fixed", 
            use_container_width=True, 
            key="active_ledger_editor",
            column_config={
                "URL": st.column_config.LinkColumn("Record Link"), 
                "Year": st.column_config.NumberColumn("Year", format="%d"), 
                "DOY": st.column_config.NumberColumn("DOY", format="%d")
            }
        )
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("💾 Save Workspace Edits", type="primary", key="save_active_btn"):
                for idx, row in edited_active_df.iterrows():
                    if (row['Flowering'] or row['Fruiting'] or row['Vegetative']) and not row['Phenology_Scored']:
                        edited_active_df.at[idx, 'Phenology_Scored'] = True
                    if idx in df.index:
                        df.loc[idx] = edited_active_df.loc[idx]
                save_with_ordered_columns(df, DB_FILE)
                st.success("Workspace edits written to master file!")
                time.sleep(1)
                st.rerun()
        with col_btn2:
            st.download_button(
                "📥 Download Active Subset (CSV)", 
                data=edited_active_df.to_csv(index=False).encode('utf-8'), 
                file_name="active_phenology_subset.csv", 
                mime="text/csv"
            )

with tab2:
    if plot_df.empty: 
        st.warning("No data to map.")
    else:
        map_df = plot_df.dropna(subset=['Latitude', 'Longitude']).copy()
        if map_df.empty: 
            st.warning("No coordinate data available in the filtered dataset.")
        else:
            map_df['Data_Source'] = map_df['Data_Source'].fillna('Unknown')
            map_df['Species_Source'] = map_df['Species'].fillna('Unknown') + ' (' + map_df['Data_Source'] + ')'
            
            map_color_by = st.radio(
                "Color Map Points By:", 
                options=["Species & Data Source", "Data Source (GBIF vs iNaturalist)", "Species & Outliers"],
                horizontal=True
            )
            
            if "Species & Data Source" in map_color_by: 
                target_color_col = 'Species_Source'
            elif "Data Source" in map_color_by: 
                target_color_col = 'Data_Source'
            else: 
                target_color_col = 'Map_Label'
            
            map_center = dict(lat=map_df["Latitude"].mean(), lon=map_df["Longitude"].mean())
            fig_map = px.scatter_map(
                map_df, lat="Latitude", lon="Longitude", color=target_color_col, 
                hover_data=["Year", "DOY", "Data_Source", "Collector", "Species"], 
                zoom=5, center=map_center, map_style="carto-positron", title="Specimen Collection Sites"
            )
            fig_map.update_traces(marker=dict(size=9, opacity=0.8))
            fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    if plot_df.empty: 
        st.info("No data available for graphing.")
    elif len(num_cols) < 2: 
        st.warning("Not enough numeric columns to graph.")
    else:
        valid_plot_df = plot_df.dropna(subset=[selected_x, selected_y]).copy()
        valid_plot_df['Point_Type'] = valid_plot_df['Is_Outlier'].apply(lambda x: 'Outlier' if x else 'Normal')
        
        if len(valid_plot_df) >= 2:
            st.subheader("📊 Regression & Climate-Phenology Interactions")

            if use_outlier_filter:
                color_var = 'Point_Type' 
            else:
                color_var = 'Species' if 'Species' in valid_plot_df.columns else None
            
            fig = px.scatter(
                valid_plot_df, x=selected_x, y=selected_y, color=color_var, 
                symbol='Data_Source', color_discrete_map={'Normal': '#636EFA', 'Outlier': '#EF553B'}, 
                hover_data=['Species', 'Collector', 'Year', 'DOY', 'Data_Source'], 
                template="streamlit", title=f"Scatter Trend: {selected_y} vs {selected_x}"
            )
            
            trend_df = valid_plot_df[valid_plot_df['Is_Outlier'] == False]
            regression_stats = []
            
            # --- CUSTOM TARGETED TRENDLINE MENU ---
            st.markdown("### 🎛️ Custom Trendline Selector")
            st.write("Select exact combinations of species and data sources to analyze:")
            
            line_options = ["Overall Combined"]
            available_srcs = trend_df['Data_Source'].dropna().unique()
            available_spps = trend_df['Species'].dropna().unique()
            
            for src in available_srcs:
                line_options.append(f"All Species ({src} Only)")
            for spp in available_spps:
                line_options.append(f"{spp} (Combined Sources)")
                for src in trend_df[trend_df['Species'] == spp]['Data_Source'].dropna().unique():
                    line_options.append(f"{spp} ({src})")
                    
            selected_lines = st.multiselect(
                "Draw trendlines for:", 
                options=line_options, 
                default=["Overall Combined"] if "Overall Combined" in line_options else []
            )
            
            def add_trendline(df_sub, name, color, dash_style):
                if len(df_sub) > 1:
                    x_v = df_sub[selected_x].astype(float).values
                    y_v = df_sub[selected_y].astype(float).values
                    if np.var(x_v) > 0 and np.var(y_v) > 0:
                        from scipy.stats import linregress
                        slope, intercept, r_value, p_value, std_err = linregress(x_v, y_v)
                        r2 = r_value**2
                        x_seq = np.array([x_v.min(), x_v.max()])
                        y_seq = slope * x_seq + intercept
                        
                        fig.add_trace(go.Scatter(
                            x=x_seq, y=y_seq, mode='lines', name=f'{name} (R²={r2:.2f})',
                            line=dict(color=color, width=3, dash=dash_style)
                        ))
                        
                        eq_sign = "+" if intercept >= 0 else "-"
                        equation = f"y = {slope:.4f}x {eq_sign} {abs(intercept):.4f}"
                        p_fmt = f"{p_value:.4e}" if p_value < 0.0001 else f"{p_value:.4f}"
                        
                        return {
                            "Dataset Line": name, 
                            "Linear Equation": equation, 
                            "R²": f"{r2:.4f}", 
                            "p-value": p_fmt, 
                            "Sample Size (n)": len(df_sub)
                        }
                return None

            line_palette = ['#000000', '#FF4B4B', '#1C83E1', '#00D4B2', '#9A52BF', '#F68410', '#3D4B53']
            color_idx = 0
            
            for line_choice in selected_lines:
                if line_choice == "Overall Combined":
                    stat = add_trendline(trend_df, line_choice, line_palette[color_idx % len(line_palette)], 'solid')
                    if stat: 
                        regression_stats.append(stat)
                elif line_choice.startswith("All Species (") and "Only)" in line_choice:
                    src = line_choice.replace("All Species (", "").replace(" Only)", "")
                    sub_df = trend_df[trend_df['Data_Source'] == src]
                    stat = add_trendline(sub_df, line_choice, line_palette[color_idx % len(line_palette)], 'dash')
                    if stat: 
                        regression_stats.append(stat)
                elif "(Combined Sources)" in line_choice:
                    spp = line_choice.replace(" (Combined Sources)", "")
                    sub_df = trend_df[trend_df['Species'] == spp]
                    stat = add_trendline(sub_df, line_choice, line_palette[color_idx % len(line_palette)], 'dot')
                    if stat: 
                        regression_stats.append(stat)
                else:
                    for spp in available_spps:
                        for src in available_srcs:
                            if line_choice == f"{spp} ({src})":
                                sub_df = trend_df[(trend_df['Species'] == spp) & (trend_df['Data_Source'] == src)]
                                stat = add_trendline(sub_df, line_choice, line_palette[color_idx % len(line_palette)], 'dashdot')
                                if stat: 
                                    regression_stats.append(stat)
                color_idx += 1

            fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig, use_container_width=True)
            
            if regression_stats:
                st.markdown("### 🧮 Regression Statistics Matrices")
                st.table(pd.DataFrame(regression_stats))
            
            # --- ADVANCED ECOLOGICAL ANALYTICS PANEL ---
            st.write("---")
            st.subheader("🔬 Advanced Ecological Analytics")

            # 1. Historical Epoch Analysis
            with st.expander("⏳ Climate Sensitivity Over Time (Custom Epochs)", expanded=False):
                st.markdown("#### Time Range Split Analysis")
                st.write("Compare trend metrics across specific eras. E.g., How does the trend in 1900-1950 compare to 1980-2020?")
                
                num_epochs = st.number_input("Number of Epoch Windows:", min_value=1, max_value=4, value=2)
                epochs = []
                for i in range(num_epochs):
                    ec_a, ec_b = st.columns(2)
                    with ec_a: 
                        start_y = st.number_input(f"Epoch {i+1} Start Year", min_value=1800, max_value=2030, value=1900 + (i*50), key=f"eps_{i}")
                    with ec_b: 
                        end_y = st.number_input(f"Epoch {i+1} End Year", min_value=1902, max_value=current_year + 10, value=1950 + (i*50), key=f"epe_{i}")
                    epochs.append((start_y, end_y))
                
                epoch_cols = st.columns(num_epochs)
                for i, (y_start, y_end) in enumerate(epochs):
                    df_ep = valid_plot_df[(valid_plot_df['Year'] >= y_start) & (valid_plot_df['Year'] <= y_end)]
                    with epoch_cols[i]:
                        st.markdown(f"**{y_start} to {y_end}** (N = {len(df_ep)})")
                        if len(df_ep) > 1:
                            ex_v, ey_v = df_ep[selected_x].values, df_ep[selected_y].values
                            if np.var(ex_v) > 0 and np.var(ey_v) > 0:
                                eslope, eintercept = np.polyfit(ex_v, ey_v, 1)
                                er2 = np.corrcoef(ex_v, ey_v)[0, 1]**2
                                st.metric("Sensitivity Slope", f"{eslope:.3f}")
                                st.write(f"Equation: `y = {eslope:.3f}x + {eintercept:.1f}`")
                                st.write(f"R² Fit: `{er2:.3f}`")
                            else: 
                                st.warning("Insufficient variance.")
                        else: 
                            st.warning("Not enough records.")

            # 2. Multivariate Linear Modeling
            with st.expander("🧮 Multivariate Climate Modeling & Regression", expanded=False):
                st.markdown("#### Multiple Linear Regression (MLR)")
                st.write("Analyze structural interactions across multiple predictors simultaneously to explain phenotypic variations.")
                
                candidate_predictors = [c for c in num_cols if c != selected_y]
                defaults = [c for c in ['Y_MAT', 'Y_MAP', 'Tave_Anomaly', 'Year', 'Photoperiod_Hours'] if c in candidate_predictors]
                selected_predictors = st.multiselect("Select Independent Driver Variables:", options=candidate_predictors, default=defaults, key="mlr_preds")
                
                if not selected_predictors:
                    st.info("Select one or more climatic parameters to isolate effects.")
                else:
                    mlr_df = valid_plot_df[[selected_y] + selected_predictors].dropna()
                    if len(mlr_df) < len(selected_predictors) + 2:
                        st.warning("Insufficient independent dataset dimensions to resolve linear equations.")
                    else:
                        y_mat = mlr_df[selected_y].values
                        X_mat = mlr_df[selected_predictors].values
                        X_design = np.hstack([np.ones((X_mat.shape[0], 1)), X_mat])
                        try:
                            beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y_mat
                            y_pred = X_design @ beta
                            ss_res = np.sum((y_mat - y_pred) ** 2)
                            ss_tot = np.sum((y_mat - np.mean(y_mat)) ** 2)
                            mlr_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                            
                            st.markdown(f"**Regression Output** (N = {len(mlr_df)}) | **Overall R² Fit Matrix:** `{mlr_r2:.3f}`")
                            coeff_df = pd.DataFrame({"Predictor Node": ["Intercept Coefficient"] + selected_predictors, "Calculated Effect (Beta Weight)": beta})
                            st.dataframe(coeff_df, use_container_width=True, hide_index=True)
                            
                            fig_importance = px.bar(
                                coeff_df[coeff_df['Predictor Node'] != 'Intercept Coefficient'],
                                x='Predictor Node', y='Calculated Effect (Beta Weight)',
                                title="Variable Contribution Weights (Direction and Magnitude)",
                                color='Calculated Effect (Beta Weight)', color_continuous_scale="RdBu", color_continuous_midpoint=0.0
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                        except Exception as e:
                            st.error(f"OLS Solver execution error: {e}")

            # 3. Correlation Heatmap Matrix
            with st.expander("🔥 Correlation Heatmap Matrix", expanded=False):
                st.markdown("#### Pearson r Interaction Matrix")
                st.write("Isolate parameters with heavy structural collinearity to choose the best independent features.")
                
                heatmap_candidates = [selected_y, 'Year', 'Latitude', 'Elevation', 'Tave_Anomaly', 'PPT_Anomaly', 'Y_MAT', 'Y_MAP', 'Photoperiod_Hours']
                default_hm = [v for v in heatmap_candidates if v in num_cols]
                
                heatmap_vars = st.multiselect("Select Heatmap Targets:", options=num_cols, default=default_hm, key="hm_vars")
                if len(heatmap_vars) < 2: 
                    st.info("Requires at least two intersecting parameters to chart matrix boundaries.")
                else:
                    corr_matrix = plot_df[heatmap_vars].corr(method='pearson')
                    fig_hm = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1.0, zmax=1.0, title="Variable Cross-Correlation Matrix")
                    st.plotly_chart(fig_hm, use_container_width=True)
            
            # 4. Generate Export Scripts 
            with st.expander("💾 Generate Export Scripts (R / Python)", expanded=False):
                st.markdown("#### Reproduce your Scatter Plots Locally")
                st.write("Copy and paste these scripts into your local IDE. Ensure you have downloaded the **Filtered Subset (CSV)** from the sidebar and saved it in your working directory.")
                
                script_lang = st.radio("Select Language:", ["Python", "R (ggplot2)"], horizontal=True)
                if script_lang == "Python":
                    py_code = f"""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Load the filtered dataset downloaded from the app
df = pd.read_csv('filtered_phenology_subset.csv')

# Drop missing values for the selected axes
df = df.dropna(subset=['{selected_x}', '{selected_y}'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='{selected_x}', y='{selected_y}', hue='Species', style='Data_Source', s=60, alpha=0.8)

# Calculate and plot regression line (Combined)
slope, intercept, r_value, p_value, std_err = linregress(df['{selected_x}'], df['{selected_y}'])
plt.plot(df['{selected_x}'], slope * df['{selected_x}'] + intercept, color='black', linewidth=2,
         label=f'y = {{slope:.3f}}x + {{intercept:.3f}}\\nR² = {{r_value**2:.3f}}\\np = {{p_value:.3e}}')

plt.title('{selected_y} vs {selected_x}')
plt.xlabel('{selected_x}')
plt.ylabel('{selected_y}')
plt.legend()
plt.tight_layout()
plt.show()
"""
                    st.code(py_code, language="python")
                else:
                    r_code = f"""library(ggplot2)
library(dplyr)

# Load the filtered dataset downloaded from the app
df <- read.csv('filtered_phenology_subset.csv')

# Filter missing data
df <- df %>% filter(!is.na({selected_x}) & !is.na({selected_y}))

# Calculate linear model
model <- lm({selected_y} ~ {selected_x}, data=df)
summary_model <- summary(model)
r_squared <- summary_model$r.squared
p_val <- pf(summary_model$fstatistic[1], summary_model$fstatistic[2], summary_model$fstatistic[3], lower.tail=FALSE)

eq_label <- sprintf("y = %.3fx + %.3f\\nR² = %.3f\\np = %.3e", 
                    coef(model)[2], coef(model)[1], r_squared, p_val)

# Plot Output
ggplot(df, aes(x={selected_x}, y={selected_y})) +
  geom_point(aes(color=Species, shape=Data_Source), size=3, alpha=0.7) +
  geom_smooth(method="lm", color="black", se=FALSE) +
  annotate("text", x=min(df${selected_x}, na.rm=TRUE), y=max(df${selected_y}, na.rm=TRUE), 
           label=eq_label, hjust=0) +
  labs(title=paste("{selected_y} vs {selected_x}"),
       x="{selected_x}", y="{selected_y}") +
  theme_minimal()
"""
                    st.code(r_code, language="r")
# --- PHENO-CLINE SPACE-FOR-TIME DASHBOARD ---
            st.write("---")
            st.subheader("🌍 Pheno-cline: Space-for-Time Substitution")
            st.write("Hopkin's Bioclimatic Law suggests phenology is delayed by altitude and latitude. These gradients act as spatial proxies for climate change.")
            
            c_pc1, c_pc2 = st.columns(2)
            with c_pc1:
                if 'Latitude' in valid_plot_df.columns:
                    fig_lat = px.scatter(
                        valid_plot_df, x='Latitude', y='DOY', color='Species', 
                        trendline="ols", title="DOY vs. Latitude (Hopkin's Law)",
                        hover_data=['Year', 'Data_Source']
                    )
                    st.plotly_chart(fig_lat, use_container_width=True)
            with c_pc2:
                if 'Elevation' in valid_plot_df.columns:
                    fig_elev = px.scatter(
                        valid_plot_df, x='Elevation', y='DOY', color='Species', 
                        trendline="ols", title="DOY vs. Elevation",
                        hover_data=['Year', 'Data_Source']
                    )
                    st.plotly_chart(fig_elev, use_container_width=True)

            # --- CLIMATE ANOMALY VIOLIN PLOTS ---
            st.write("---")
            st.subheader("🎻 Climate Anomaly Distributions")
            st.write("How does the phenological window shift during extreme climatic years?")
            
            if 'Tave_Anomaly' in plot_df.columns and 'PPT_Anomaly' in plot_df.columns:
                anomaly_df = plot_df.dropna(subset=['DOY', 'Tave_Anomaly', 'PPT_Anomaly']).copy()
                
                # Classify anomalies into regimes based on deviations from 0
                anomaly_df['Temp_Regime'] = anomaly_df['Tave_Anomaly'].apply(lambda x: 'Warm' if x > 0 else 'Cold')
                anomaly_df['Precip_Regime'] = anomaly_df['PPT_Anomaly'].apply(lambda x: 'Wet' if x > 0 else 'Dry')
                anomaly_df['Climate_Regime'] = anomaly_df['Temp_Regime'] + " & " + anomaly_df['Precip_Regime']
                
                fig_viol = px.violin(
                    anomaly_df, x='Climate_Regime', y='DOY', color='Climate_Regime', 
                    box=True, points="all", title="DOY Distribution grouped by Climate Regime",
                    category_orders={"Climate_Regime": ["Warm & Dry", "Warm & Wet", "Cold & Dry", "Cold & Wet"]}
                )
                st.plotly_chart(fig_viol, use_container_width=True)

                # Calculate summary statistics for the anomaly table
                anomaly_summary = anomaly_df.groupby('Climate_Regime').agg(
                    N=('DOY', 'count'),
                    Mean_DOY=('DOY', 'mean'),
                    Median_DOY=('DOY', 'median'),
                    Mean_Temp_Anomaly=('Tave_Anomaly', 'mean'),
                    Mean_Precip_Anomaly=('PPT_Anomaly', 'mean')
                ).reset_index()
                
                # Round numeric columns for cleaner UI display
                anomaly_summary = anomaly_summary.round(2)
                
                st.markdown("#### Climate Regime Summary Statistics")
                st.write("This table defines the average temperature (°C) and precipitation (mm) deviations that characterize each regime, along with the resulting shift in the Day of Year (DOY).")
                st.dataframe(anomaly_summary, use_container_width=True, hide_index=True)
                
with tab4:
    st.subheader("🎯 Rapid Image Scoring")
    # Safety check: Ensure columns exist before filtering to prevent KeyErrors
if 'Phenology_Scored' not in plot_df.columns:
    plot_df['Phenology_Scored'] = False
if 'URL' not in plot_df.columns:
    plot_df['URL'] = ''

# Your existing line 1332 (leave this as is):
unscored = plot_df[(plot_df['Phenology_Scored'] == False) & (plot_df['URL'].notna()) & (plot_df['URL'].str.strip() != '') & (plot_df['URL'] != 'Manual')]
unscored = plot_df[(plot_df['Phenology_Scored'] == False) & (plot_df['URL'].notna()) & (plot_df['URL'].str.strip() != '') & (plot_df['URL'] != 'Manual')]
if unscored.empty: 
        st.success("🎉 All URL records within this focused workspace configuration have been scored!")
else:
        st.info(f"You have **{len(unscored)}** unscored records remaining inside this workspace focus.")
        target_idx = st.selectbox("Select Record:", unscored.index, format_func=lambda x: f"Index {x}: {df.loc[x, 'Species']} ({df.loc[x, 'Data_Source']}) - Year {df.loc[x, 'Year']}")
        row = df.loc[target_idx]
        safe_url = ensure_url_scheme(row['URL'])
        st.link_button("🔗 Click Here to View Original Specimen Image/Page", safe_url, type="primary", use_container_width=True)
        st.caption("If the button above does not work, copy and paste this link into your browser:")
        st.code(safe_url, language="text")
        st.write("---")
        
        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1: 
            f1 = st.checkbox("🌸 Flowering", value=bool(row['Flowering']), key=f"f1_{target_idx}")
        with c_s2: 
            f2 = st.checkbox("🍒 Fruiting", value=bool(row['Fruiting']), key=f"f2_{target_idx}")
        with c_s3: 
            f3 = st.checkbox("🍃 Vegetative (Leaves)", value=bool(row['Vegetative']), key=f"f3_{target_idx}")
            
        if st.button("💾 Save Phenology & Load Next", type="secondary"):
            df.at[target_idx, 'Flowering'] = f1
            df.at[target_idx, 'Fruiting'] = f2
            df.at[target_idx, 'Vegetative'] = f3
            df.at[target_idx, 'Phenology_Scored'] = True
            save_with_ordered_columns(df, DB_FILE)
            st.rerun()

with tab5:
    st.subheader("🌲 Machine Learning: DOY Predictive Modeler")
    st.write("Train a Random Forest to identify non-linear relationships. It supports BOTH numeric climate drivers and categorical data (like Species and Data Source).")
    
    if plot_df.empty:
        st.warning("No data available to train a model.")
    else:
        num_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = [c for c in ['Species', 'Data_Source'] if c in plot_df.columns]
        
        if len(num_cols) < 2:
            st.warning("Not enough numeric columns for machine learning.")
        else:
            ml_target = st.selectbox("Select Target Variable (y):", num_cols, index=num_cols.index('DOY') if 'DOY' in num_cols else 0)
            
            ml_candidates = [c for c in num_cols if c != ml_target] + cat_cols
            default_ml_feats = [c for c in ['Species', 'Data_Source', 'Year', 'Latitude', 'Elevation', 'Y_MAT', 'Photoperiod_Hours'] if c in ml_candidates]
            
            ml_features = st.multiselect("Select Predictor Features (X):", ml_candidates, default=default_ml_feats)
            
            if st.checkbox("🚀 Enable & Train Random Forest Model", value=False):
                if not ml_features:
                    st.warning("Please select at least one predictor feature.")
                else:
                    ml_df = plot_df[[ml_target] + ml_features].dropna()
                    if len(ml_df) < 10:
                        st.error("Not enough valid data points to train a reliable model. You need at least 10 complete rows.")
                    else:
                        with st.spinner("One-Hot Encoding Categoricals & Training Model..."):
                            from sklearn.ensemble import RandomForestRegressor
                            import numpy as np
                            
                            # 1. Prepare Features and strictly cast Target to float
                            X_raw = ml_df[ml_features].copy()
                            y = ml_df[ml_target].astype(float)
                            
                            # 2. Apply Circular Time Transformation if DOY is used as a predictor
                            if 'DOY' in X_raw.columns:
                                X_raw['DOY_sin'] = np.sin(2 * np.pi * X_raw['DOY'] / 365.25)
                                X_raw['DOY_cos'] = np.cos(2 * np.pi * X_raw['DOY'] / 365.25)
                                X_raw = X_raw.drop(columns=['DOY'])
                                
                            # 3. Explicitly identify categoricals to prevent "Ghost Strings"
                            present_cats = [c for c in cat_cols if c in X_raw.columns]
                            
                            # 4. One-Hot Encode and force EVERYTHING to float to prevent Boolean errors
                            X_encoded = pd.get_dummies(X_raw, columns=present_cats, drop_first=False).astype(float)
                            
                            from sklearn.model_selection import train_test_split
                            
                            # --- 70/30 SPLIT ---
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_encoded, y, test_size=0.30, random_state=42
                            )
                            
                            # 5. Train Model (on the 70%)
                            rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf.fit(X_train, y_train)
                            
                            # Score Model (on the hidden 30%)
                            score = rf.score(X_test, y_test)
                            
                            st.success(f"**Model trained successfully!** | Train Size: {len(X_train)} | Test Size: {len(X_test)}\n\nOut-of-Sample Predictive R²: `{score:.3f}`")
                            
                            st.success(f"**Model trained successfully on {len(ml_df)} samples!** |  Overall Predictive R²: `{score:.3f}`")
                            
                            importances = rf.feature_importances_
                            imp_df = pd.DataFrame({"Feature": X_encoded.columns, "Importance Weight": importances}).sort_values('Importance Weight', ascending=True).tail(15)
                            
                            fig_rf = px.bar(imp_df, x='Importance Weight', y='Feature', orientation='h', 
                                            title="Top 15 Feature Importances (Non-Linear Impact)",
                                            color='Importance Weight', color_continuous_scale="Viridis")
                            st.plotly_chart(fig_rf, use_container_width=True)
                            
                            st.write("---")
                            st.markdown("### 🔮 DOY Sensitivity & Distribution Simulator")
                            st.write(f"Calculate how the predicted `{ml_target}` changes when a specific environmental variable naturally fluctuates **between its Mean ± 1 Standard Deviation**, while holding all other parameters constant.")
                            
                            numeric_features = [f for f in ml_features if f not in cat_cols]
                            
                            if not numeric_features:
                                st.info("Select at least one continuous numeric feature to run the sensitivity distribution.")
                            else:
                                sim_c1, sim_c2, sim_c3 = st.columns(3)
                                
                                base_profile = {}
                                if 'Species' in ml_features:
                                    with sim_c1:
                                        base_profile['Species'] = st.selectbox("Simulate for Species:", ml_df['Species'].unique())
                                if 'Data_Source' in ml_features:
                                    with sim_c2:
                                        base_profile['Data_Source'] = st.selectbox(
                                            "Simulate for Data Source:", 
                                            options=["GBIF Herbarium", "iNaturalist", "Both"]
                                        )
                                        
                                with sim_c3:
                                    target_sim_var = st.selectbox("Variable to Fluctuate (± 1 SD):", numeric_features)
                                
                                if target_sim_var:
                                    mu = X_raw[target_sim_var].mean()
                                    sigma = X_raw[target_sim_var].std()
                                    
                                    st.markdown(f"**📊 Statistical Baseline Profiles for `{target_sim_var}`:**")
                                    m_col1, m_col2, m_col3 = st.columns(3)
                                    with m_col1: 
                                        st.metric("Dataset Mean", f"{mu:.2f}")
                                    with m_col2: 
                                        st.metric("1 Standard Deviation (SD)", f"{sigma:.2f}")
                                    with m_col3: 
                                        st.metric("Fluctuation Span (±1 SD)", f"{mu - sigma:.2f} to {mu + sigma:.2f}")
                                    
                                    sim_values = np.linspace(mu - sigma, mu + sigma, 100)
                                    sim_df = pd.DataFrame({target_sim_var: sim_values})
                                    
                                    for f in numeric_features:
                                        if f != target_sim_var:
                                            sim_df[f] = X_raw[f].mean()
                                            
                                    if 'Species' in ml_features:
                                        sim_df['Species'] = base_profile['Species']
                                        
                                    if 'Data_Source' in ml_features:
                                        if base_profile['Data_Source'] == "Both":
                                            sim_df_gbif = sim_df.copy()
                                            sim_df_gbif['Data_Source'] = "GBIF Herbarium"
                                            sim_encoded_gbif = pd.get_dummies(sim_df_gbif).reindex(columns=X_encoded.columns, fill_value=0)
                                            preds_gbif = rf.predict(sim_encoded_gbif)
                                            
                                            sim_df_inat = sim_df.copy()
                                            sim_df_inat['Data_Source'] = "iNaturalist"
                                            sim_encoded_inat = pd.get_dummies(sim_df_inat).reindex(columns=X_encoded.columns, fill_value=0)
                                            preds_inat = rf.predict(sim_encoded_inat)
                                            
                                            sim_preds = (preds_gbif + preds_inat) / 2.0
                                        else:
                                            sim_df['Data_Source'] = base_profile['Data_Source']
                                            sim_encoded = pd.get_dummies(sim_df).reindex(columns=X_encoded.columns, fill_value=0)
                                            sim_preds = rf.predict(sim_encoded)
                                    else:
                                        sim_encoded = pd.get_dummies(sim_df).reindex(columns=X_encoded.columns, fill_value=0)
                                        sim_preds = rf.predict(sim_encoded)
                                    
                                    min_pred = sim_preds.min()
                                    max_pred = sim_preds.max()
                                    shift_days = max_pred - min_pred
                                    
                                    plot_sim_df = pd.DataFrame({target_sim_var: sim_values, "Predicted DOY": sim_preds})
                                    fig_sim = px.line(
                                        plot_sim_df, x=target_sim_var, y="Predicted DOY", 
                                        title=f"Predicted Shift in {ml_target} across 1 SD of {target_sim_var}",
                                        labels={target_sim_var: f"{target_sim_var} (Span Range)"}
                                    )
                                    
                                    fig_sim.add_vline(x=mu, line_dash="dash", line_color="black", annotation_text="Mean")
                                    fig_sim.add_vline(x=mu - sigma, line_dash="dot", line_color="gray", annotation_text="-1 SD")
                                    fig_sim.add_vline(x=mu + sigma, line_dash="dot", line_color="gray", annotation_text="+1 SD")
                                    
                                    st.plotly_chart(fig_sim, use_container_width=True)
                                    
                                    st.success(f"**Insight:** For **{base_profile.get('Species', 'All Species')}** ({base_profile.get('Data_Source', 'Combined')}), as `{target_sim_var}` naturally fluctuates within 1 Standard Deviation, the model predicts the {ml_target} shifts by **{shift_days:.1f} days** (from {min_pred:.1f} to {max_pred:.1f}).")

                                    # --- PHENOPHASE CLASSIFICATION (PROBABILITY CURVES) ---
    st.write("---")
    st.subheader("🌸 Phenophase Classification: Flowering Probability Curve")
    st.write("Instead of predicting an exact day, this model calculates the probability of a plant flowering on any given day of the year based on its environmental conditions.")
    
    if 'Flowering' in plot_df.columns:
        class_df = plot_df.dropna(subset=['Flowering']).copy()
        
        # Convert boolean (True/False) to binary (1/0) for the ML model
        class_df['Flowering'] = class_df['Flowering'].astype(int) 
        
        clf_candidates = [c for c in num_cols if c != 'DOY'] + cat_cols
        default_clf_feats = [c for c in ['Y_MAT', 'Latitude', 'Elevation', 'Photoperiod_Hours'] if c in clf_candidates]
        
        clf_features = st.multiselect(
            "Select Environmental Predictors (DOY is automatically included):", 
            clf_candidates, 
            default=default_clf_feats, 
            key="clf_feats"
        )
        
        if st.button("📈 Train Probability Model & Generate Curve", type="primary"):
            if class_df['Flowering'].nunique() < 2:
                st.warning("Your dataset must contain both Flowering (True) and Non-Flowering (False) records to train this model. Keep scoring records!")
            else:
                with st.spinner("Training Phenophase Classifier..."):
                    from sklearn.ensemble import RandomForestClassifier
                    
                    # 1. Prepare Data
                    model_features = ['DOY'] + clf_features
                    valid_class_df = class_df.dropna(subset=model_features).copy()
                    
                    X_clf = valid_class_df[model_features].copy()
                    y_clf = valid_class_df['Flowering']
                    
                    # 2. Circular DOY transformation
                    X_clf['DOY_sin'] = np.sin(2 * np.pi * X_clf['DOY'] / 365.25)
                    X_clf['DOY_cos'] = np.cos(2 * np.pi * X_clf['DOY'] / 365.25)
                    X_clf = X_clf.drop(columns=['DOY'])
                    
                   # 3. One-hot encoding for categorical variables
                    X_clf_encoded = pd.get_dummies(X_clf, drop_first=False)
                    
                    from sklearn.model_selection import train_test_split
                    
                    # --- 70/30 SPLIT ---
                    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                        X_clf_encoded, y_clf, test_size=0.30, random_state=42
                    )
                    
                    # 4. Train Model (on the 70%)
                    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
                    clf.fit(X_train_c, y_train_c)
                    
                    # Score Model (on the hidden 30%)
                    accuracy = clf.score(X_test_c, y_test_c)
                    
                    st.success(f"**Classification Model Trained!** | Train Size: {len(X_train_c)} | Test Size: {len(X_test_c)}\n\nOut-of-Sample Accuracy: `{accuracy*100:.1f}%`")
                    
                    st.success(f"**Classification Model Trained!** | Accuracy: `{accuracy*100:.1f}%`")
                    
                    # 5. Simulate the probability curve across a full year (DOY 1 to 365)
                    sim_doy = np.arange(1, 366)
                    sim_data = pd.DataFrame({'DOY': sim_doy})
                    
                    # Circularize the simulated DOY
                    sim_data['DOY_sin'] = np.sin(2 * np.pi * sim_data['DOY'] / 365.25)
                    sim_data['DOY_cos'] = np.cos(2 * np.pi * sim_data['DOY'] / 365.25)
                    sim_data = sim_data.drop(columns=['DOY'])
                    
                    # Fill all other selected features with their mean (or mode for categoricals)
                    for col in clf_features:
                        if col in cat_cols:
                            mode_val = valid_class_df[col].mode()[0]
                            sim_data[col] = mode_val
                        else:
                            mean_val = valid_class_df[col].mean()
                            sim_data[col] = mean_val
                            
                    # Align columns with the trained model
                    sim_encoded = pd.get_dummies(sim_data).reindex(columns=X_clf_encoded.columns, fill_value=0)
                    
                    # 6. Predict Probabilities ([:, 1] extracts the probability for class 1: Flowering)
                    probs = clf.predict_proba(sim_encoded)[:, 1] 
                    
                    # 7. Plot the Bell Curve
                    prob_df = pd.DataFrame({'Day of Year': sim_doy, 'Flowering Probability': probs})
                    fig_prob = px.line(
                        prob_df, x='Day of Year', y='Flowering Probability',
                        title="Simulated Flowering Probability over the Year",
                        labels={'Flowering Probability': 'Probability (0 to 1)'}
                    )
                    
                    # Fill the area under the curve for a cleaner look
                    fig_prob.update_traces(fill='tozeroy', line_color='#E45756')
                    fig_prob.update_layout(yaxis_range=[0, 1.05])
                    st.plotly_chart(fig_prob, use_container_width=True)
                    st.caption("Note: This curve isolates the effect of time by holding your chosen environmental parameters constant at their statistical mean.")
with tab6:
    st.markdown("### 🗄️ Comprehensive Database Master Ledger")
    st.info("⚠️ This dashboard exposes the raw, unfiltered master database. You are free to view, append, modify, or delete any record across the entire study project directly inside this workspace.")
    
    edited_master_df = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, key="master_ledger_editor",
        column_config={
            "URL": st.column_config.LinkColumn("Record Link"), 
            "Year": st.column_config.NumberColumn("Year", format="%d"), 
            "DOY": st.column_config.NumberColumn("DOY", format="%d")
        }
    )
    
    col_btn1_m, col_btn2_m = st.columns([1, 4])
    with col_btn1_m:
        if st.button("💾 Save Master Edits", type="primary", key="save_master_btn"):
            for idx, row in edited_master_df.iterrows():
                if (row['Flowering'] or row['Fruiting'] or row['Vegetative']) and not row['Phenology_Scored']:
                    edited_master_df.at[idx, 'Phenology_Scored'] = True
            save_with_ordered_columns(edited_master_df, DB_FILE)
            st.success("Global database repository successfully overwritten and updated!")
            time.sleep(1)
            st.rerun()
    with col_btn2_m:
        st.download_button(
            "📥 Download Full Master Dataset (CSV)", 
            data=edited_master_df.to_csv(index=False).encode('utf-8'), 
            file_name="phenology_dataset.csv", 
            mime="text/csv", 
            key="download_master_btn"
        )
