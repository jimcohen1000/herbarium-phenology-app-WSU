import streamlit as st
import pandas as pd
import plotly.express as px
import os
import requests
import time
import random
from datetime import date, datetime

st.set_page_config(layout="wide")
st.title("Herbarium Tracker: Full Ledger & Analytics")

db_file = "herbarium_database_expanded.csv"

# 1. INITIALIZE DATABASE WITH HEADERS
base_headers = [
    "Data_Source", "Collector", "Col_Number", "Barcode", "Species", "DOY", "Year",
    "Flowering", "Fruiting", "Vegetative", "Latitude", "Longitude", "Elevation", "URL"
]

if not os.path.exists(db_file):
    pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)

# --- Helpers: API Fetchers & Formatting ---
# --- Helpers: API Fetchers & Formatting ---
def get_elevation(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            elevations = res.json().get('elevation')
            if elevations and len(elevations) > 0:
                return float(elevations[0])
        else:
            st.error(f"⚠️ Elevation API failed with status: {res.status_code}")
    except Exception as e: 
        st.error(f"⚠️ Elevation Network Error: {e}")
    return None

def get_climate_data(lat, lon, el, prd):
    if el is None: 
        st.warning(f"⚠️ Skipping ClimateNA for {prd}: Elevation is missing.")
        return {} 
        
    base = "https://api.climatena.ca/api/cnaApi6/LatLonEl"
    url = f"{base}?ID1=1&ID2=t1&lat={lat}&lon={lon}&el={el}&prd={prd}&varYSM=YSM"
    
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data[0] if isinstance(data, list) else data
        else:
            st.error(f"⚠️ ClimateNA API Error ({res.status_code}): Could not fetch {prd}.")
            st.code(res.text)  # Shows the exact error from the server
            return {}
    except Exception as e: 
        st.error(f"⚠️ ClimateNA Network/Timeout Error: {e}")
        return {}

def save_with_ordered_columns(df_to_save, filepath):
    new_order = [c for c in base_headers if c in df_to_save.columns]
    priority_climate = ["Y_MAT", "N_MAT", "Y_MAP", "N_MAP"] 
    new_order += [c for c in priority_climate if c in df_to_save.columns and c not in new_order]
    new_order += [c for c in df_to_save.columns if c not in new_order]
    df_to_save[new_order].to_csv(filepath, index=False)

# --- Layout ---
c1, c2 = st.columns([1, 2.2])

# Column 1: Data Collection
with c1:
    st.subheader("Data Collection")
    tab1, tab2, tab3 = st.tabs(["🌿 Manual", "🦋 iNaturalist", "🏛️ Digital Herbaria"])
    
    # --- TAB 1: MANUAL HERBARIUM ENTRY ---
    with tab1:
        with st.form("data_entry_form"):
            collector = st.text_input("Collector Name")
            col_num = st.text_input("Collector Number")
            barcode = st.text_input("Barcode")
            spp = st.text_input("Species")
            
            date_val = st.date_input("Date", min_value=date(1900, 1, 1), max_value=date(2030, 12, 31), value=date(2020, 5, 1))
            
            st.write("**Phenology:**")
            col_f, col_fr, col_v = st.columns(3)
            with col_f: flow = st.checkbox("Flowering", value=True)
            with col_fr: fruit = st.checkbox("Fruiting")
            with col_v: veg = st.checkbox("Vegetative")
            
            lat = st.number_input("Lat", format="%.5f", value=51.1764)
            lon = st.number_input("Lon", format="%.5f", value=-115.5682)
            el = st.number_input("Elev (m)", value=1420)
            
            submitted = st.form_submit_button("💾 SAVE ENTRY", type="primary", use_container_width=True)
        
        if submitted:
            with st.spinner("Fetching climate models..."):
                year_data = get_climate_data(lat, lon, el, f"Year_{date_val.year}")
                norm_data = get_climate_data(lat, lon, el, "Normal_1961_1990")
                
                row = {
                    "Data_Source": "Manual Entry",
                    "Collector": collector, "Col_Number": col_num, "Barcode": barcode,
                    "Species": spp, "DOY": int(date_val.strftime("%j")), "Year": date_val.year,
                    "Flowering": flow, "Fruiting": fruit, "Vegetative": veg,
                    "Latitude": lat, "Longitude": lon, "Elevation": el, "URL": ""
                }
                
                for k, v in year_data.items(): row[f"Y_{k}"] = v
                for k, v in norm_data.items(): row[f"N_{k}"] = v
                
                try:
                    df_existing = pd.read_csv(db_file)
                except Exception:
                    df_existing = pd.DataFrame(columns=base_headers)
                    
                df_combined = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
                save_with_ordered_columns(df_combined, db_file)
                st.success("Entry saved!")
                # st.rerun()

    # --- TAB 2: INATURALIST BATCH IMPORT ---
    with tab2:
        with st.form("inat_import_form"):
            inat_spp = st.text_input("Target Species", "Anemone patens")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1: d1 = st.date_input("Start Date", value=date(1950, 1, 1))
            with col_d2: d2 = st.date_input("End Date", value=date(2022, 12, 31))
            
            inat_limit = st.slider("Records to Fetch", 5, 50, 25, step=5)
            st.info("Randomly samples North American records -> Dedups -> Calculates Elev -> Fetches ClimateNA.")
            
            inat_submitted = st.form_submit_button("📥 FETCH iNATURALIST", type="primary", use_container_width=True)
            
        if inat_submitted:
            try:
                df_existing = pd.read_csv(db_file)
                existing_urls = set(df_existing["URL"].dropna().tolist())
            except Exception:
                existing_urls = set()

            records = []
            available_years = list(range(d1.year, d2.year + 1))
            random_years = random.choices(available_years, k=inat_limit * 4) # extra buffer for skipped dupes
            
            st.write(f"Contacting iNaturalist for {inat_limit} randomized North American records...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            collected_data = []
            
            for y in random_years:
                if len(collected_data) >= inat_limit: break
                
                # Bounding box added: nelat=83&nelng=-50&swlat=15&swlng=-170 (North America)
                url = f"https://api.inaturalist.org/v1/observations?taxon_name={inat_spp}&quality_grade=research&d1={y}-01-01&d2={y}-12-31&nelat=83&nelng=-50&swlat=15&swlng=-170&per_page=5"
                try:
                    res = requests.get(url, timeout=5)
                    if res.status_code == 200:
                        data = res.json().get('results', [])
                        for obs in data:
                            if len(collected_data) >= inat_limit: break
                            obs_url = obs.get('uri', "")
                            
                            # Deduplication Check
                            if obs_url in existing_urls or obs.get('id') in [d.get('id') for d in collected_data]:
                                continue
                                
                            collected_data.append(obs)
                            status_text.text(f"Found new record from {y}... ({len(collected_data)}/{inat_limit})")
                except Exception:
                    pass
            
            if collected_data:
                for i, obs in enumerate(collected_data):
                    if obs.get('location') and obs.get('observed_on'):
                        lat_str, lon_str = obs['location'].split(',')
                        try:
                            dt = datetime.strptime(obs['observed_on'], "%Y-%m-%d")
                            lat, lon = float(lat_str), float(lon_str)
                            
                            status_text.text(f"Processing {i+1}/{len(collected_data)}: Finding elevation...")
                            el = get_elevation(lat, lon)
                            
                            row = {
                                "Data_Source": "iNaturalist",
                                "Species": obs.get('taxon', {}).get('name', inat_spp),
                                "Latitude": lat, "Longitude": lon, "Elevation": el,
                                "Year": dt.year, "DOY": dt.timetuple().tm_yday,
                                "Flowering": False, "Fruiting": False, "Vegetative": False,
                                "URL": obs.get('uri', "")
                            }
                            
                            if el is not None:
                                status_text.text(f"Processing {i+1}/{len(collected_data)}: Pulling climate models...")
                                year_data = get_climate_data(lat, lon, el, f"Year_{dt.year}")
                                norm_data = get_climate_data(lat, lon, el, "Normal_1961_1990")
                                for k, v in year_data.items(): row[f"Y_{k}"] = v
                                for k, v in norm_data.items(): row[f"N_{k}"] = v
                            
                            records.append(row)
                        except Exception: pass
                    progress_bar.progress((i + 1) / len(collected_data))
                
                status_text.text("Finished processing!")
                if records:
                    try:
                        df_existing = pd.read_csv(db_file)
                    except Exception:
                        df_existing = pd.DataFrame(columns=base_headers)
                    df_combined = pd.concat([df_existing, pd.DataFrame(records)], ignore_index=True)
                    save_with_ordered_columns(df_combined, db_file)
                    st.success(f"Added {len(records)} temporally randomized iNaturalist records!")
                    time.sleep(1)
                    st.rerun()
            else:
                st.warning("Could not find enough valid records.")

    # --- TAB 3: DIGITAL HERBARIA (GBIF) BATCH IMPORT ---
    with tab3:
        with st.form("gbif_import_form"):
            st.markdown("Query global databases (including **CCH2**, **Intermountain Biota**, and **PNW Herbaria**) via GBIF.")
            gbif_spp = st.text_input("Target Species", "Anemone patens", key="g_spp")
            
            col_g1, col_g2 = st.columns(2)
            with col_g1: g_d1 = st.date_input("Start Date", value=date(1900, 1, 1), key="g_d1")
            with col_g2: g_d2 = st.date_input("End Date", value=date(2022, 12, 31), key="g_d2")
            
            gbif_limit = st.slider("Records to Fetch", 5, 50, 25, step=5, key="g_lim")
            st.info("Randomly samples North American records -> Dedups -> Pulls Data -> Fetches ClimateNA.")
            
            gbif_submitted = st.form_submit_button("📥 FETCH HERBARIA DATA", type="primary", use_container_width=True)
            
        if gbif_submitted:
            try:
                df_existing = pd.read_csv(db_file)
                existing_urls = set(df_existing["URL"].dropna().tolist())
                existing_barcodes = set(df_existing["Barcode"].dropna().tolist())
            except Exception:
                existing_urls = set()
                existing_barcodes = set()

            records = []
            available_years = list(range(g_d1.year, g_d2.year + 1))
            random_years = random.choices(available_years, k=gbif_limit * 4)
            
            st.write(f"Scouting global databases for {gbif_limit} North American records...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            collected_data = []
            
            for y in random_years:
                if len(collected_data) >= gbif_limit: break
                
                # Bounding box added: decimalLatitude=15,83 & decimalLongitude=-170,-50
                url = f"https://api.gbif.org/v1/occurrence/search?scientificName={gbif_spp}&hasCoordinate=true&mediaType=StillImage&basisOfRecord=PRESERVED_SPECIMEN&year={y}&decimalLatitude=15,83&decimalLongitude=-170,-50&limit=5"
                try:
                    res = requests.get(url, timeout=5)
                    if res.status_code == 200:
                        data = res.json().get('results', [])
                        for obs in data:
                            if len(collected_data) >= gbif_limit: break
                            
                            rec_url = obs.get('references', '')
                            if not rec_url and obs.get('media'):
                                rec_url = obs.get('media')[0].get('identifier', '')
                            cat_num = obs.get('catalogNumber', '')
                            
                            # Deduplication Check
                            if rec_url in existing_urls or (cat_num and cat_num in existing_barcodes) or obs.get('key') in [d.get('key') for d in collected_data]:
                                continue
                                
                            collected_data.append(obs)
                            status_text.text(f"Found new record from {y}... ({len(collected_data)}/{gbif_limit})")
                except Exception:
                    pass
            
            if collected_data:
                for i, obs in enumerate(collected_data):
                    y, m, d = obs.get('year'), obs.get('month'), obs.get('day')
                    lat, lon = obs.get('decimalLatitude'), obs.get('decimalLongitude')
                    
                    if y and m and d and lat and lon:
                        try:
                            dt = datetime(int(y), int(m), int(d))
                            el = obs.get('elevation')
                            if pd.isna(el) or el is None:
                                status_text.text(f"Processing {i+1}/{len(collected_data)}: Finding elevation...")
                                el = get_elevation(lat, lon)
                            
                            rec_url = obs.get('references', '')
                            if not rec_url and obs.get('media'):
                                rec_url = obs.get('media')[0].get('identifier', '')
                                
                            row = {
                                "Data_Source": "Digitized Herbarium",
                                "Collector": obs.get('recordedBy', ''),
                                "Col_Number": obs.get('recordNumber', ''),
                                "Barcode": obs.get('catalogNumber', ''),
                                "Species": obs.get('species', gbif_spp),
                                "Latitude": float(lat), "Longitude": float(lon), "Elevation": el,
                                "Year": dt.year, "DOY": dt.timetuple().tm_yday,
                                "Flowering": False, "Fruiting": False, "Vegetative": False,
                                "URL": rec_url
                            }
                            
                            if el is not None:
                                status_text.text(f"Processing {i+1}/{len(collected_data)}: Pulling climate models...")
                                year_data = get_climate_data(lat, lon, el, f"Year_{dt.year}")
                                norm_data = get_climate_data(lat, lon, el, "Normal_1961_1990")
                                for k, v in year_data.items(): row[f"Y_{k}"] = v
                                for k, v in norm_data.items(): row[f"N_{k}"] = v
                            
                            records.append(row)
                        except Exception: pass
                    progress_bar.progress((i + 1) / len(collected_data))
                
                status_text.text("Finished processing!")
                if records:
                    try:
                        df_existing = pd.read_csv(db_file)
                    except Exception:
                        df_existing = pd.DataFrame(columns=base_headers)
                    df_combined = pd.concat([df_existing, pd.DataFrame(records)], ignore_index=True)
                    save_with_ordered_columns(df_combined, db_file)
                    st.success(f"Successfully added {len(records)} temporally randomized herbarium records!")
                    time.sleep(1.5)
                    st.rerun()
            else:
                st.warning("Could not find enough valid records.")

# Column 2: Graphing & Database
with c2:
    st.subheader("📊 Analysis Dashboard")
    try:
        df = pd.read_csv(db_file)
    except Exception:
        df = pd.DataFrame(columns=base_headers)
        
    for col in base_headers:
        if col not in df.columns:
            df[col] = None
            
    if not df.empty:
        for col in ["Flowering", "Fruiting", "Vegetative"]:
            df[col] = df[col].fillna(False).astype(bool)
    
    # --- GRAPH SECTION ---
    if not df.empty and len(df) > 0:
        plot_vars = ["Year", "Latitude", "Longitude", "Elevation"] + [c for c in df.columns if c.startswith('Y_') or c.startswith('N_')]
        if len(plot_vars) > 0:
            g1, g2 = st.columns(2)
            with g1: x_var = st.selectbox("Select X-Axis Variable:", plot_vars)
            with g2: 
                species_list = df["Species"].dropna().unique()
                selected_spp = st.multiselect("Filter Species:", species_list, default=species_list)
            
            st.write("**Require Phenology Traits:**")
            p1, p2, p3 = st.columns(3)
            with p1: req_flow = st.checkbox("Flowering Only")
            with p2: req_fruit = st.checkbox("Fruiting Only")
            with p3: req_veg = st.checkbox("Vegetative Only")

            plot_df = df[df["Species"].isin(selected_spp)].copy()
            if req_flow: plot_df = plot_df[plot_df["Flowering"] == True]
            if req_fruit: plot_df = plot_df[plot_df["Fruiting"] == True]
            if req_veg: plot_df = plot_df[plot_df["Vegetative"] == True]
            
            if x_var in plot_df.columns:
                plot_df[x_var] = pd.to_numeric(plot_df[x_var], errors='coerce')
                plot_df = plot_df.dropna(subset=[x_var, "DOY"])
                if not plot_df.empty:
                    fig = px.scatter(
                        plot_df, x=x_var, y="DOY", 
                        color="Species", 
                        symbol="Data_Source", 
                        hover_data=["Year", "URL"] if "URL" in plot_df.columns else ["Year"],
                        trendline="ols", 
                        title=f"Phenology (DOY) vs {x_var}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough valid data points to graph after applying filters.")
    else:
        st.info("No data collected yet. Add your first entry on the left to see the graph!")
        
    st.write("---")
    
    # --- TABLE SECTION ---
    st.subheader("📋 Formatted Database Ledger")
    if not df.empty:
        df = df.sort_values(by=["Year", "DOY"], ascending=[False, False])
            
    # Session state key protects data from wiping, dynamically sizing based on columns
    dynamic_key = f"herbarium_ledger_{len(df.columns)}"
    
    edited_df = st.data_editor(
        df, 
        key=dynamic_key,
        use_container_width=True, hide_index=True, num_rows="dynamic", 
        column_config={
            "Year": st.column_config.NumberColumn("Year", format="%d"),
            "DOY": st.column_config.NumberColumn("DOY"),
            "Latitude": st.column_config.NumberColumn("Lat", format="%.4f"),
            "Longitude": st.column_config.NumberColumn("Lon", format="%.4f"),
            "Elevation": st.column_config.NumberColumn("Elev", format="%d m"),
            "URL": st.column_config.LinkColumn("Link"),
            "Data_Source": st.column_config.TextColumn("Source", disabled=True), 
            "Flowering": st.column_config.CheckboxColumn("Flowering"),
            "Fruiting": st.column_config.CheckboxColumn("Fruiting"),
            "Vegetative": st.column_config.CheckboxColumn("Vegetative"),
            "Species": st.column_config.TextColumn("Species") 
        }
    )
    
    col_save, col_dl, _ = st.columns([1, 1, 2])
    with col_save:
        if st.button("💾 Save Ledger Edits", type="primary", use_container_width=True):
            save_with_ordered_columns(edited_df, db_file)
            st.success("Database updated successfully!")
            st.rerun()

    with col_dl:
        # Pull straight from the active dataframe 'df' loaded at the top of column 2
        st.download_button(
            label="📥 Download Full CSV", 
            data=df.to_csv(index=False).encode('utf-8'), 
            file_name="herbarium_full_data.csv", 
            mime="text/csv",
            use_container_width=True
        )
        
    with st.expander("⚠️ Danger Zone"):
        st.write("Wiping the database will clear your current view, but a timestamped backup will automatically be saved in your folder first.")
        if st.button("Wipe Entire Database", type="secondary"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"herbarium_backup_{timestamp}.csv"
            if os.path.exists(db_file):
                pd.read_csv(db_file).to_csv(backup_file, index=False)
            pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)
            st.success(f"Database wiped! Backed up to: {backup_file}")
            time.sleep(2) 
            st.rerun()
