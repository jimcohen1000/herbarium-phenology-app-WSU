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
#         REAL API & CLIMATE HOOKS
# ==========================================

def get_elevation(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            elevations = res.json().get('elevation')
            if elevations and len(elevations) > 0:
                return float(elevations[0])
    except Exception as e:
        print("Elevation error:", e)
    return None

def get_climate_data(lat, lon, el, prd):
    if pd.isna(el) or el is None: 
        return {} 
    # THE FIX: Reverted strictly to http://. The API server fails SSL checks on https://
    base = "http://api.climatena.ca/api/cnaApi6/LatLonEl"
    url = f"{base}?ID1=1&ID2=t1&lat={lat}&lon={lon}&el={el}&prd={prd}&varYSM=YSM"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data[0] if isinstance(data, list) else data
    except Exception as e:
        print("ClimateNA error:", e)
    return {}


# ==========================================
#         DATA PIPELINE FUNCTIONS
# ==========================================

def remove_duplicate_collections(df):
    if df.empty: return df
    df = df.copy()
    if 'Col_Number' in df.columns:
        df['Col_Number'] = df['Col_Number'].astype(str).replace(['nan', 'None', '<NA>', ''], pd.NA)
        has_col_num = df[df['Col_Number'].notna()]
        no_col_num = df[df['Col_Number'].isna()]
        subset_cols = ['Collector', 'Col_Number'] if 'Collector' in df.columns else ['Col_Number']
        clean_has_col = has_col_num.drop_duplicates(subset=subset_cols, keep='first')
        df = pd.concat([clean_has_col, no_col_num], ignore_index=True)
    return df

def thin_and_cap_data(df, target_limit, max_per_year=3):
    """Filters duplicates, caps at 3 per year, and returns exactly the target limit."""
    if df.empty: return df
    
    # 1. Strip identical sheets
    df = remove_duplicate_collections(df)
    
    # 2. Flatten temporal clustering
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        has_year = df[df['Year'].notna()].copy()
        no_year = df[df['Year'].isna()]
        
        has_year['Year'] = has_year['Year'].astype(int)
        thinned = has_year.groupby('Year', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_per_year))
        )
        df = pd.concat([thinned, no_year], ignore_index=True)
        
    # 3. Return exactly the amount the user asked for
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
    
    # Fetch Elevation and Climate NA for the surviving records
    for i, row in cleaned_df.iterrows():
        row_dict = row.to_dict()
        lat, lon = row_dict.get('Latitude'), row_dict.get('Longitude')
        year, el = row_dict.get('Year'), row_dict.get('Elevation')
        
        if pd.notna(lat) and pd.notna(lon) and pd.notna(year):
            # 1. Fetch Elevation if missing
            if pd.isna(el) or el == "" or el == 0 or el == 0.0:
                status_text.text(f"Fetching elevation... ({i+1}/{len(cleaned_df)})")
                el = get_elevation(lat, lon)
                row_dict['Elevation'] = el
            
            # 2. Fetch Climate Data
            if el is not None and not pd.isna(el):
                status_text.text(f"Fetching ClimateNA for {int(year)}... ({i+1}/{len(cleaned_df)})")
                year_data = get_climate_data(lat, lon, el, f"Year_{int(year)}")
                norm_data = get_climate_data(lat, lon, el, "Normal_1961_1990")
                
                # Append new columns to the dictionary 
                for k, v in year_data.items(): row_dict[f"Y_{k}"] = v
                for k, v in norm_data.items(): row_dict[f"N_{k}"] = v
                
        records.append(row_dict)
        progress_bar.progress((i + 1) / len(cleaned_df))
        
    status_text.text("Finished processing pipeline!")
    
    # Save to master database
    if records:
        final_new_df = pd.DataFrame(records)
        master_df = pd.read_csv(db_file)
        combined_df = pd.concat([master_df, final_new_df], ignore_index=True)
        
        # Ensure base tracking columns stay on the left side
        new_order = [c for c in base_headers if c in combined_df.columns]
        new_order += [c for c in combined_df.columns if c not in new_order]
        
        combined_df[new_order].to_csv(db_file, index=False)
        st.sidebar.success(f"Added {len(final_new_df)} fully processed climate records!")
        time.sleep(1.5)
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
                fetch_limit = min(g_limit * 5, 300) # Cap safety to avoid hitting API hard limits
                url = f"https://api.gbif.org/v1/occurrence/search?scientificName={spp_encoded}&year={g_start},{g_end}&limit={fetch_limit}&hasCoordinate=true&basisOfRecord=PRESERVED_SPECIMEN"
                
                raw_records = []
                try:
                    res = requests.get(url, timeout=10)
                    if res.status_code == 200:
                        for obs in res.json().get('results', []):
                            rec_url = obs.get('references', '')
                            if not rec_url and obs.get('media'): rec_url = obs.get('media')[0].get('identifier', '')
                            
                            raw_records.append({
                                "Data_Source": "Digitized Herbarium",
                                "Collector": obs.get('recordedBy', ''),
                                "Col_Number": obs.get('recordNumber', ''),
                                "Barcode": obs.get('catalogNumber', ''),
                                "Species": obs.get('species', gbif_spp),
                                "Year": obs.get('year', pd.NA),
                                "DOY": datetime(int(obs['year']), int(obs['month']), int(obs['day'])).timetuple().tm_yday if obs.get('month') and obs.get('day') else pd.NA,
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
                fetch_limit = min(i_limit * 5, 200) # Cap safety
                url = f"https://api.inaturalist.org/v1/observations?taxon_name={spp_encoded}&d1={i_start}-01-01&d2={i_end}-12-31&per_page={fetch_limit}&quality_grade=research"
                
                raw_records = []
                try:
                    res = requests.get(url, timeout=10)
                    if res.status_code == 200:
                        for obs in res.json().get('results', []):
                            if obs.get('location') and obs.get('observed_on'):
                                lat_str, lon_str = obs['location'].split(',')
                                dt = datetime.strptime(obs['observed_on'], "%Y-%m-%d")
                                raw_records.append({
                                    "Data_Source": "iNaturalist",
                                    "Collector": obs.get('user', {}).get('login', ''),
                                    "Species": obs.get('taxon', {}).get('name', inat_spp),
                                    "Year": dt.year, "DOY": dt.timetuple().tm_yday,
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

# --- TYPE SAFETY FIX FOR STREAMLIT ---
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
        "Year": st.column_config.NumberColumn("Year", format="%d"),
        "DOY": st.column_config.NumberColumn("DOY"),
        "Latitude": st.column_config.NumberColumn("Lat", format="%.4f"),
        "Longitude": st.column_config.NumberColumn("Lon", format="%.4f"),
        "Elevation": st.column_config.NumberColumn("Elev", format="%d m"),
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
    plot_df['Year'] = pd.to_numeric(plot_df['Year'], errors='coerce')
    plot_df['DOY'] = pd.to_numeric(plot_df['DOY'], errors='coerce')
    plot_df['Latitude'] = pd.to_numeric(plot_df['Latitude'], errors='coerce')
    
    if 'Y_MAT' not in plot_df.columns:
        has_climate_data = False
    else:
        plot_df['Y_MAT'] = pd.to_numeric(plot_df['Y_MAT'], errors='coerce')
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
            st.markdown("**Phenological Profile: Collection Day of Year vs. Latitude**")
            fig_pheno = px.scatter(
                plot_df, x='Latitude', y='DOY', color='Species' if 'Species' in plot_df.columns else None,
                hover_data=['Collector', 'Year'],
                labels={'DOY': 'Day of Year (DOY)', 'Latitude': 'Latitude (°N)'},
                template="streamlit"
            )
            fig_pheno.update_layout(margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_pheno, use_container_width=True)
