import streamlit as st
import pandas as pd
import plotly.express as px
import os
import requests
import time
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
def get_elevation(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            elevations = res.json().get('elevation')
            if elevations and len(elevations) > 0:
                return float(elevations[0])
    except Exception: 
        return None
    return None

def get_climate_data(lat, lon, el, prd):
    if el is None: return {} 
    base = "https://api.climatena.ca/api/cnaApi6/LatLonEl"
    url = f"{base}?ID1=1&ID2=t1&lat={lat}&lon={lon}&el={el}&prd={prd}&varYSM=YSM"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data[0] if isinstance(data, list) else data
    except Exception: 
        return {}
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
    tab1, tab2 = st.tabs(["🌿 Herbarium", "🦋 iNaturalist"])
    
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
                    "Data_Source": "Herbarium",
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
                
                st.success("Herbarium Entry saved!")
                st.rerun()

    # --- TAB 2: INATURALIST BATCH IMPORT ---
    with tab2:
        with st.form("inat_import_form"):
            inat_spp = st.text_input("Target Species", "Anemone patens")
            
            st.write("**Date Range (Ensures ClimateNA compatibility):**")
            col_d1, col_d2 = st.columns(2)
            with col_d1: d1 = st.date_input("Start Date", value=date(2000, 1, 1))
            with col_d2: d2 = st.date_input("End Date", value=date(2022, 12, 31))
            
            inat_limit = st.slider("Records to Fetch", 5, 50, 25, step=5)
            st.info("Pulls Location -> Calculates Elevation -> Fetches ClimateNA models.")
            
            inat_submitted = st.form_submit_button("📥 FETCH & PROCESS DATA", type="primary", use_container_width=True)
            
        if inat_submitted:
            records = []
            url = f"https://api.inaturalist.org/v1/observations?taxon_name={inat_spp}&quality_grade=research&per_page={inat_limit}&d1={d1}&d2={d2}"
            
            st.write(f"Contacting iNaturalist for {inat_limit} records between {d1.year} and {d2.year}...")
            res = requests.get(url, timeout=15)
            
            if res.status_code == 200:
                data = res.json().get('results', [])
                if data:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, obs in enumerate(data):
                        if obs.get('location') and obs.get('observed_on'):
                            lat_str, lon_str = obs['location'].split(',')
                            date_str = obs['observed_on']
                            
                            try:
                                dt = datetime.strptime(date_str, "%Y-%m-%d")
                                lat, lon = float(lat_str), float(lon_str)
                                
                                status_text.text(f"Processing {i+1}/{len(data)}: Finding elevation...")
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
                                    status_text.text(f"Processing {i+1}/{len(data)}: Pulling climate models...")
                                    year_data = get_climate_data(lat, lon, el, f"Year_{dt.year}")
                                    norm_data = get_climate_data(lat, lon, el, "Normal_1961_1990")
                                    for k, v in year_data.items(): row[f"Y_{k}"] = v
                                    for k, v in norm_data.items(): row[f"N_{k}"] = v
                                
                                records.append(row)
                            except Exception:
                                pass
                        
                        progress_bar.progress((i + 1) / len(data))
                    
                    status_text.text("Finished processing!")
                    
                    if records:
                        try:
                            df_existing = pd.read_csv(db_file)
                        except Exception:
                            df_existing = pd.DataFrame(columns=base_headers)
                            
                        df_combined = pd.concat([df_existing, pd.DataFrame(records)], ignore_index=True)
                        save_with_ordered_columns(df_combined, db_file)
                        st.success(f"Successfully added {len(records)} iNaturalist records!")
                        st.rerun()
                else:
                    st.warning("No records found for that species.")
            else:
                st.error("Failed to connect to iNaturalist.")

# Column 2: Graphing & Database
with c2:
    st.subheader("📊 Analysis Dashboard")
    
    # Safely load the CSV. If it's corrupted or truly empty, load headers only.
    try:
        df = pd.read_csv(db_file)
    except Exception:
        df = pd.DataFrame(columns=base_headers)
        
    for col in base_headers:
        if col not in df.columns:
            df[col] = None
            
    # SAFETY CHECK: Always force phenology to boolean for the filters and table
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
            
            # --- NEW PHENOLOGY FILTERS ---
            st.write("**Require Phenology Traits:**")
            p1, p2, p3 = st.columns(3)
            with p1: req_flow = st.checkbox("Flowering Only")
            with p2: req_fruit = st.checkbox("Fruiting Only")
            with p3: req_veg = st.checkbox("Vegetative Only")

            # Apply filters
            plot_df = df[df["Species"].isin(selected_spp)].copy()
            if req_flow: plot_df = plot_df[plot_df["Flowering"] == True]
            if req_fruit: plot_df = plot_df[plot_df["Fruiting"] == True]
            if req_veg: plot_df = plot_df[plot_df["Vegetative"] == True]
            
            if x_var in plot_df.columns:
                plot_df[x_var] = pd.to_numeric(plot_df[x_var], errors='coerce')
                plot_df = plot_df.dropna(subset=[x_var, "DOY"])
                if not plot_df.empty:
                    # --- UPDATED PLOTLY GRAPH ---
                    fig = px.scatter(
                        plot_df, x=x_var, y="DOY", 
                        color="Species",           # 👈 Species are now colors
                        symbol="Data_Source",      # 👈 Data Source is now shapes
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
    st.info("💡 You can edit Species names, check off Phenology, or select and delete rows directly in this table! Click 'Save Ledger Edits' below to write changes to the CSV.")
    
    if not df.empty:
        df = df.sort_values(by=["Year", "DOY"], ascending=[False, False])
            
    edited_df = st.data_editor(
        df, 
        use_container_width=True, 
        hide_index=True,
        num_rows="dynamic", 
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
        st.download_button(
            "📥 Download Full CSV", 
            data=df.to_csv(index=False), 
            file_name="full_data.csv", 
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
            
            st.success(f"Database wiped! Your previous data was safely backed up to: {backup_file}")
            time.sleep(3) 
            st.rerun()
