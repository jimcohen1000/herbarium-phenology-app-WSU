import streamlit as st
import pandas as pd
import os
import time
import requests
import urllib.parse
from datetime import datetime
import plotly.express as px

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="Herbarium Climate Ledger")
db_file = "herbarium_database.csv"

base_headers = [
    "Data_Source", "Collector", "Col_Number", "Barcode", "Species", 
    "DOY", "Year", "Flowering", "Fruiting", "Vegetative", 
    "Latitude", "Longitude", "Elevation", "URL"
]

if not os.path.exists(db_file):
    pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)


# ==========================================
#         DATA SANITIZERS
# ==========================================

def safe_float(val):
    try:
        if pd.isna(val) or str(val).strip().lower() in ["", "nan", "none", "<na>"]:
            return None
        return float(val)
    except (ValueError, TypeError):
        return None

def safe_int(val):
    v = safe_float(val)
    return int(v) if v is not None else None


# ==========================================
#         REAL API & CLIMATE HOOKS
# ==========================================

def get_elevation(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        headers = {"User-Agent": "HerbariumLedger/1.0"}
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if "elevation" in data:
                el_data = data["elevation"]
                if isinstance(el_data, list) and len(el_data) > 0:
                    return float(el_data[0])
                elif isinstance(el_data, (int, float)):
                    return float(el_data)
    except Exception as e:
        print(f"Elevation Error: {e}")
    return None

def get_climate_data(lat, lon, el, prd):
    # ClimateNA fallback: always provide an integer elevation, defaulting to 0 if missing
    el_val = int(float(el)) if el is not None else 0 
    base = "http://api.climatena.ca/api/cnaApi6/LatLonEl"
    url = f"{base}?ID1=1&ID2=t1&lat={lat}&lon={lon}&el={el_val}&prd={prd}&varYSM=YSM"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=15)
        
        if res.status_code == 429 or "limit" in res.text.lower():
            return {"_LIMIT_REACHED": True}
            
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict):
                return data
    except Exception as e:
        print(f"ClimateNA Error: {e}")
        
    return {}


# ==========================================
#         DATA PIPELINE FUNCTIONS
# ==========================================

def thin_and_cap_data(df, target_limit, max_per_year=3):
    """Safely deduplicates and caps data without using .apply() which drops columns."""
    if df.empty: return df
    
    # 1. Deduplicate based on standard fields
    if 'Collector' in df.columns and 'Col_Number' in df.columns:
        df['Col_Number'] = df['Col_Number'].fillna('')
        df = df.drop_duplicates(subset=['Collector', 'Col_Number'], keep='first')
        
    # 2. Safely Thin by Year (Native Pandas approach)
    if 'Year' in df.columns:
        # Randomize rows to get a true sample, then take the first N per group
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.groupby('Year', dropna=False).head(max_per_year)
        
    return df.head(target_limit).reset_index(drop=True)


def pipeline_enrich_and_save(raw_df, target_limit):
    if raw_df.empty:
        st.sidebar.warning("No records found in that date range.")
        return

    st.sidebar.text("Applying de-clustering filters...")
    cleaned_df = thin_and_cap_data(raw_df, target_limit=target_limit, max_per_year=3)
    
    records = []
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    limit_reached_flag = False 
    
    for i, row in cleaned_df.iterrows():
        row_dict = row.to_dict()
        
        lat = safe_float(row_dict.get('Latitude'))
        lon = safe_float(row_dict.get('Longitude'))
        year = safe_int(row_dict.get('Year'))
        el = safe_float(row_dict.get('Elevation'))
        
        # Enforce clean Year assignment
        row_dict['Year'] = year if year is not None else pd.NA
        
        if lat is not None and lon is not None:
            # 1. Fetch Elevation (Does NOT require Year to be present anymore)
            if el is None or el == 0.0 or pd.isna(el):
                status_text.text(f"Fetching elevation... ({i+1}/{len(cleaned_df)})")
                fetched_el = get_elevation(lat, lon)
                el = fetched_el if fetched_el is not None else 0.0 # Failsafe so ClimateNA runs
                row_dict['Elevation'] = el
            else:
                row_dict['Elevation'] = el
            
            # 2. Fetch Climate Data (Requires Lat, Lon, and Year)
            if year is not None:
                if not limit_reached_flag:
                    climate_year = min(year, 2022)
                    status_text.text(f"Fetching ClimateNA for {climate_year}... ({i+1}/{len(cleaned_df)})")
                    year_data = get_climate_data(lat, lon, el, f"Year_{climate_year}")
                    
                    if year_data.get("_LIMIT_REACHED"):
                        st.sidebar.error("⚠️ ClimateNA 50-request limit reached! Adding remaining records without climate data.")
                        limit_reached_flag = True
                    elif year_data: 
                        norm_data = get_climate_data(lat, lon, el, "Normal_1961_1990")
                        if norm_data.get("_LIMIT_REACHED"):
                            limit_reached_flag = True
                        else:
                            for k, v in year_data.items(): row_dict[f"Y_{k}"] = v
                            for k, v in norm_data.items(): row_dict[f"N_{k}"] = v
                
        records.append(row_dict)
        progress_bar.progress((i + 1) / len(cleaned_df))
        
    status_text.text("Finished processing pipeline!")
    
    if records:
        final_new_df = pd.DataFrame(records)
        master_df = pd.read_csv(db_file)
        
        # Force all base headers to exist so Pandas doesn't ignore empty columns
        for col in base_headers:
            if col not in final_new_df.columns: final_new_df[col] = pd.NA
            if col not in master_df.columns: master_df[col] = pd.NA
            
        combined_df = pd.concat([master_df, final_new_df], ignore_index=True)
        
        # Sort columns to keep standard structure first
        new_order = [c for c in base_headers if c in combined_df.columns]
        new_order += [c for c in combined_df.columns if c not in new_order]
        
        combined_df[new_order].to_csv(db_file, index=False)
        st.sidebar.success(f"Added {len(final_new_df)} processed records!")
        time.sleep(2)
        st.rerun()


# ==========================================
#      SIDEBAR: DATA ENTRY & EXPORT
# ==========================================
with st.sidebar:
    st.title("📥 Data Entry")
    
    with st.expander("🌐 Fetch from GBIF", expanded=False):
        gbif_spp = st.text_input("Species Name (GBIF):", placeholder="e.g., Lithospermum ruderale", key="g_spp")
        col_yr1, col_yr2 = st.columns(2)
        with col_yr1: g_start = st.number_input("Start Year:", min_value=1800, max_value=2026, value=1950, key="g_start")
        with col_yr2: g_end = st.number_input("End Year:", min_value=1800, max_value=2026, value=2026, key="g_end")
        g_limit = st.number_input("GBIF Record Limit:", min_value=5, max_value=200, value=25, step=5)
        
        if st.button("Fetch & Process GBIF", type="primary", use_container_width=True):
            with st.spinner("Querying GBIF..."):
                spp_encoded = urllib.parse.quote(gbif_spp)
                fetch_limit = min(g_limit * 5, 300) 
                url = f"https://api.gbif.org/v1/occurrence/search?scientificName={spp_encoded}&year={g_start},{g_end}&limit={fetch_limit}&hasCoordinate=true&basisOfRecord=PRESERVED_SPECIMEN"
                
                raw_records = []
                try:
                    res = requests.get(url, timeout=10)
                    if res.status_code == 200:
                        for obs in res.json().get('results', []):
                            rec_url = obs.get('references', '')
                            if not rec_url and obs.get('media'): rec_url = obs.get('media')[0].get('identifier', '')
                            
                            # Extremely safe date extraction
                            y = obs.get('year')
                            m = obs.get('month')
                            d = obs.get('day')
                            doy = pd.NA
                            
                            # Fallback if GBIF hides year in eventDate string
                            if not y and obs.get('eventDate'):
                                try: y = int(obs['eventDate'][:4])
                                except: pass

                            if y and m and d:
                                try: doy = datetime(int(y), int(m), int(d)).timetuple().tm_yday
                                except: pass
                            
                            raw_records.append({
                                "Data_Source": "Digitized Herbarium",
                                "Collector": obs.get('recordedBy', ''),
                                "Col_Number": obs.get('recordNumber', ''),
                                "Barcode": obs.get('catalogNumber', ''),
                                "Species": obs.get('species', gbif_spp),
                                "Year": safe_int(y) if y else pd.NA,
                                "DOY": doy,
                                "Latitude": obs.get('decimalLatitude'),
                                "Longitude": obs.get('decimalLongitude'),
                                "Elevation": obs.get('elevation', pd.NA),
                                "URL": rec_url,
                                "Flowering": False, "Fruiting": False, "Vegetative": False
                            })
                except Exception as e:
                    st.sidebar.error(f"GBIF Error: {e}")
                
                pipeline_enrich_and_save(pd.DataFrame(raw_records), target_limit=g_limit)

    with st.expander("📸 Fetch from iNaturalist", expanded=False):
        inat_spp = st.text_input("Species Name (iNat):", placeholder="e.g., Lithospermum ruderale", key="i_spp")
        col_in1, col_in2 = st.columns(2)
        with col_in1: i_start = st.number_input("Start Year:", min_value=1800, max_value=2026, value=2000, key="i_start")
        with col_in2: i_end = st.number_input("End Year:", min_value=1800, max_value=2026, value=2026, key="i_end")
        i_limit = st.number_input("iNat Record Limit:", min_value=5, max_value=200, value=25, step=5, key="i_lim")
            
        if st.button("Fetch & Process iNaturalist", type="primary", use_container_width=True):
            with st.spinner(f"Downloading observations..."):
                spp_encoded = urllib.parse.quote(inat_spp)
                fetch_limit = min(i_limit * 5, 200) 
                url = f"https://api.inaturalist.org/v1/observations?taxon_name={spp_encoded}&d1={i_start}-01-01&d2={i_end}-12-31&per_page={fetch_limit}&quality_grade=research"
                
                raw_records = []
                try:
                    res = requests.get(url, timeout=10)
                    if res.status_code == 200:
                        for obs in res.json().get('results', []):
                            if obs.get('location') and obs.get('observed_on'):
                                lat_str, lon_str = obs['location'].split(',')
                                try:
                                    dt = datetime.strptime(obs['observed_on'], "%Y-%m-%d")
                                    obs_year, obs_doy = dt.year, dt.timetuple().tm_yday
                                except:
                                    obs_year, obs_doy = pd.NA, pd.NA
                                    
                                raw_records.append({
                                    "Data_Source": "iNaturalist",
                                    "Collector": obs.get('user', {}).get('login', ''),
                                    "Species": obs.get('taxon', {}).get('name', inat_spp),
                                    "Year": obs_year, "DOY": obs_doy,
                                    "Latitude": float(lat_str), "Longitude": float(lon_str),
                                    "Elevation": pd.NA,
                                    "URL": obs.get('uri', ""),
                                    "Flowering": False, "Fruiting": False, "Vegetative": False
                                })
                except Exception as e:
                    st.sidebar.error(f"iNaturalist Error: {e}")
                    
                pipeline_enrich_and_save(pd.DataFrame(raw_records), target_limit=i_limit)

    with st.expander("✍️ Manual Data Entry", expanded=False):
        m_species = st.text_input("Species Name:", placeholder="e.g., Quercus alba")
        m_collector = st.text_input("Collector's Name:")
        m_col_num = st.text_input("Collection Number / Sheet ID:")
        m_barcode = st.text_input("Barcode Number:")
        col_m1, col_m2 = st.columns(2)
        with col_m1: m_year = st.number_input("Collection Year:", min_value=1700, max_value=2026, value=2026)
        with col_m2: m_doy = st.number_input("Day of Year (DOY):", min_value=1, max_value=366, value=150)
        col_m3, col_m4, col_m5 = st.columns(3)
        with col_m3: m_lat = st.number_input("Latitude (°N):", format="%.5f", value=40.0)
        with col_m4: m_lon = st.number_input("Longitude (°W):", format="%.5f", value=-111.0)
        with col_m5: m_elev = st.number_input("Elevation (m):", value=0.0)
        m_url = st.text_input("Image/Record URL:")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1: m_flowering = st.checkbox("Flowering")
        with col_p2: m_fruiting = st.checkbox("Fruiting")
        with col_p3: m_vegetative = st.checkbox("Vegetative")

        if st.button("Commit Manual Record", type="primary", use_container_width=True):
            new_row_df = pd.DataFrame([{
                "Data_Source": "Manual_Entry", "Collector": m_collector, "Col_Number": m_col_num,
                "Barcode": m_barcode, "Species": m_species, "DOY": m_doy, "Year": m_year,
                "Flowering": m_flowering, "Fruiting": m_fruiting, "Vegetative": m_vegetative,
                "Latitude": m_lat, "Longitude": m_lon, "Elevation": m_elev if m_elev != 0.0 else pd.NA, 
                "URL": m_url
            }])
            pipeline_enrich_and_save(new_row_df, target_limit=1)

    st.markdown("---")
    st.title("💾 Export Database")
    
    df_export = pd.read_csv(db_file)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_csv = b""
    full_csv_bytes = b""
    
    if not df_export.empty:
        existing_main_cols = [c for c in base_headers if c in df_export.columns]
        clean_csv = df_export[existing_main_cols].to_csv(index=False).encode('utf-8-sig')
        full_csv_bytes = df_export.to_csv(index=False).encode('utf-8-sig')
        
    st.download_button("📥 Download Clean Ledger", data=clean_csv, file_name=f"herbarium_clean_{current_time}.csv", mime="text/csv", use_container_width=True, disabled=(len(clean_csv)==0))
    st.download_button("📦 Download Full CSV", data=full_csv_bytes, file_name=f"herbarium_full_{current_time}.csv", mime="text/csv", use_container_width=True, disabled=(len(full_csv_bytes)==0))
    
    st.markdown("---")
    with st.expander("⚠️ Danger Zone"):
        if st.button("Wipe Entire Database", type="secondary", use_container_width=True):
            if os.path.exists(db_file):
                pd.read_csv(db_file).to_csv(f"herbarium_backup_{current_time}.csv", index=False)
            pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)
            st.success("Database cleared!")
            time.sleep(1)
            st.rerun()


# ==========================================
#         MAIN WORKSPACE LEDGER
# ==========================================
st.title("🌱 Herbarium Specimen Tracker")

df = pd.read_csv(db_file)

if not df.empty:
    df = df.dropna(how='all')
    for col in ["Flowering", "Fruiting", "Vegetative"]:
        if col in df.columns: df[col] = df[col].fillna(False).astype(bool)
    for col in ["Year", "DOY", "Latitude", "Longitude", "Elevation"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ["Data_Source", "URL", "Species", "Collector", "Barcode", "Col_Number"]:
        if col in df.columns: df[col] = df[col].fillna("").astype(str)

st.subheader("📋 Database Ledger")
if not df.empty:
    df = df.sort_values(by=["Year", "DOY"], ascending=[False, False])
df = df.reset_index(drop=True)

edited_df = st.data_editor(
    df, 
    width="stretch", hide_index=True, num_rows="dynamic", 
    column_config={
        "Year": st.column_config.NumberColumn("Year", format="%.0f"),
        "DOY": st.column_config.NumberColumn("DOY"),
        "Latitude": st.column_config.NumberColumn("Lat", format="%.4f"),
        "Longitude": st.column_config.NumberColumn("Lon", format="%.4f"),
        "Elevation": st.column_config.NumberColumn("Elev", format="%.0f m"),
        "URL": st.column_config.LinkColumn("Link"),
        "Data_Source": st.column_config.TextColumn("Source", disabled=True), 
        "Flowering": st.column_config.CheckboxColumn("Flowering"),
        "Fruiting": st.column_config.CheckboxColumn("Fruiting"),
        "Vegetative": st.column_config.CheckboxColumn("Vegetative")
    }
)

col_space1, col_save, col_space2 = st.columns([1, 1.5, 1])
with col_save:
    if st.button("💾 Save Ledger Edits", type="primary", use_container_width=True):
        edited_df.to_csv(db_file, index=False)
        st.success("Database file written successfully!")
        time.sleep(1)
        st.rerun()

# ==========================================
#          GRAPHING & TRENDS DASHBOARD
# ==========================================
st.write("---")
st.subheader("📊 Climate Trend Analytics")

if df.empty:
    st.info("The ledger is currently empty. Ingest specimen data via the sidebar to generate figures.")
else:
    plot_df = edited_df.copy()
    
    # Ensure standard base columns are explicitly numeric
    for col in ['Year', 'DOY', 'Latitude', 'Longitude', 'Elevation']:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
            
    # Ensure ALL dynamically added ClimateNA columns (Y_ and N_) are numeric for graphing
    climate_cols = [c for c in plot_df.columns if c.startswith('Y_') or c.startswith('N_')]
    for col in climate_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

    if 'Y_MAT' not in plot_df.columns:
        has_climate_data = False
    else:
        has_climate_data = True

    plot_df = plot_df.dropna(subset=['Year', 'DOY'])
    
    if not plot_df.empty:
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            st.markdown("**Chronological Trend: Mean Annual Temp vs. Collection Year**")
            if has_climate_data:
                yearly_summary = plot_df.groupby('Year')['Y_MAT'].mean().reset_index()
                fig_temp = px.line(
                    yearly_summary, x='Year', y='Y_MAT', 
                    labels={'Y_MAT': 'Mean Annual Temp (°C)', 'Year': 'Collection Year'},
                    markers=True, template="streamlit"
                )
                fig_temp.update_layout(margin=dict(l=20, r=20, t=10, b=20))
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.warning("No ClimateNA data found yet. Fetch records to generate this graph.")

        with fig_col2:
            st.markdown("**Phenological Profile: Dynamic Scatter Plot**")
            
            # Dynamically grab all numeric columns for the dropdown
            numeric_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
            
            # Remove DOY from the options so we aren't graphing DOY vs DOY
            if 'DOY' in numeric_cols:
                numeric_cols.remove('DOY')
                
            # Set a sensible default starting variable (Latitude if it exists)
            default_ix = numeric_cols.index('Latitude') if 'Latitude' in numeric_cols else 0
            
            # The dropdown selector
            selected_x_var = st.selectbox(
                "Select X-Axis Variable:", 
                options=numeric_cols, 
                index=default_ix
            )
            
            # Generate the scatter plot using the selected variable
            fig_pheno = px.scatter(
                plot_df, 
                x=selected_x_var, 
                y='DOY', 
                color='Species' if 'Species' in plot_df.columns else None,
                hover_data=['Collector', 'Year'],
                labels={'DOY': 'Day of Year (DOY)', selected_x_var: selected_x_var},
                template="streamlit"
            )
            fig_pheno.update_layout(margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_pheno, use_container_width=True)
