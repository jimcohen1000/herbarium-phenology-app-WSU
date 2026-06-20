import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime
import urllib.parse
import time
import os
import numpy as np

# ==========================================
#          APP SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="Phenology & Climate Tracker", layout="wide")

if "climate_limit_hit" not in st.session_state:
    st.session_state.climate_limit_hit = False

db_file = "specimen_ledger.csv"
base_headers = [
    "Data_Source", "Collector", "Col_Number", "Barcode", "Species",
    "Year", "DOY", "Latitude", "Longitude", "Elevation",
    "Flowering", "Fruiting", "Vegetative", "URL"
]

if not os.path.exists(db_file):
    pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)

# ==========================================
#        DATA PIPELINE FUNCTIONS
# ==========================================

def safe_float(val):
    try: return float(val) if pd.notna(val) and val != '' else None
    except: return None

def safe_int(val):
    try: return int(float(val)) if pd.notna(val) and val != '' else None
    except: return None

def get_elevation(lat, lon):
    try:
        url = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            results = res.json().get('results')
            if results and len(results) > 0:
                return results[0].get('elevation')
    except: pass
    return None

def get_climate_data(lat, lon, el, period):
    # Reverted to the correct ClimateNA API endpoint
    url = "https://climatena.ca/api/data/AnyPoint"
    params = {
        "lat": lat, "lon": lon, "el": el, 
        "period": period 
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        
        if res.status_code in [429, 403]: 
            return {"_LIMIT_REACHED": True}
            
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict) and "limit" in str(data.get("error", "")).lower():
                return {"_LIMIT_REACHED": True}
                
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict):
                return data
    except Exception as e:
        print(f"Climate API Error: {e}")
    return {}

def thin_and_cap_data(df, target_limit, max_per_year):
    valid_df = df.dropna(subset=['Latitude', 'Longitude', 'Year', 'URL']).copy()
    valid_df = valid_df[valid_df['URL'].str.strip() != '']
    
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    capped_df = valid_df.groupby('Year').head(max_per_year)
    
    capped_df = capped_df.sort_values('Year')
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
        needed = target_limit - len(spaced_df)
        spaced_df = pd.concat([spaced_df, remaining_df.head(needed)])
        
    return spaced_df.head(target_limit).sample(frac=1).reset_index(drop=True)

def pipeline_enrich_and_save(raw_df, target_limit, max_per_year=3):
    if raw_df.empty:
        st.sidebar.warning("No records found (or none passed the URL/duplication filters).")
        return

    st.sidebar.text(f"Applying de-clustering & year distribution...")
    cleaned_df = thin_and_cap_data(raw_df, target_limit=target_limit, max_per_year=max_per_year)
    
    records = []
    progress_bar = st.sidebar.progress(0.0)
    status_text = st.sidebar.empty()
    
    limit_reached_flag = st.session_state.climate_limit_hit 
    
    for count, (idx, row) in enumerate(cleaned_df.iterrows()):
        row_dict = row.to_dict()
        
        lat = safe_float(row_dict.get('Latitude'))
        lon = safe_float(row_dict.get('Longitude'))
        year = safe_int(row_dict.get('Year'))
        el = safe_float(row_dict.get('Elevation'))
        
        row_dict['Year'] = year if year is not None else pd.NA
        
        if lat is not None and lon is not None:
            if el is None or el == 0.0 or pd.isna(el):
                status_text.text(f"Fetching elevation... ({count+1}/{len(cleaned_df)})")
                fetched_el = get_elevation(lat, lon)
                el = fetched_el if fetched_el is not None else 0.0 
                row_dict['Elevation'] = el
            else:
                row_dict['Elevation'] = el
            
            if year is not None and not limit_reached_flag:
                climate_year = min(year, 2022) 
                status_text.text(f"Fetching ClimateNA for {climate_year}... ({count+1}/{len(cleaned_df)})")
                
                year_data = get_climate_data(lat, lon, el, f"Year_{climate_year}")
                if not year_data: 
                    year_data = get_climate_data(lat, lon, el, str(climate_year))
                
                if year_data.get("_LIMIT_REACHED"):
                    st.session_state.climate_limit_hit = True
                    limit_reached_flag = True
                elif year_data: 
                    norm_data = get_climate_data(lat, lon, el, "Normal_1961_1990")
                    for k, v in year_data.items(): 
                        if k not in ["ID1", "ID2", "lat", "lon", "elev", "prd", "varYSM", "period"]:
                            row_dict[f"Y_{k}"] = v
                    if isinstance(norm_data, dict):
                        for k, v in norm_data.items(): 
                            if k not in ["ID1", "ID2", "lat", "lon", "elev", "prd", "varYSM", "period"]:
                                row_dict[f"N_{k}"] = v
                
        records.append(row_dict)
        progress_val = min((count + 1) / len(cleaned_df), 1.0)
        progress_bar.progress(progress_val)
        
    status_text.text("Finished processing pipeline!")
    
    if records:
        final_new_df = pd.DataFrame(records)
        master_df = pd.read_csv(db_file)
        
        for col in base_headers:
            if col not in final_new_df.columns: final_new_df[col] = pd.NA
            if col not in master_df.columns: master_df[col] = pd.NA
            
        combined_df = pd.concat([master_df, final_new_df], ignore_index=True)
        
        new_order = [c for c in base_headers if c in combined_df.columns]
        new_order += [c for c in combined_df.columns if c not in new_order]
        
        combined_df[new_order].to_csv(db_file, index=False)
        st.sidebar.success(f"Added {len(final_new_df)} processed records!")
        time.sleep(2)
        st.rerun()

# ==========================================
#        SIDEBAR: DATA ENTRY
# ==========================================
st.sidebar.header("Data Entry & Ingestion")

if st.session_state.climate_limit_hit:
    st.sidebar.error("⚠️ **ClimateNA Limit Reached!**\n\nWait 1 hour before querying more.")
    if st.sidebar.button("Dismiss Warning", type="primary"):
        st.session_state.climate_limit_hit = False
        st.rerun()

with st.sidebar.expander("✏️ Manual Entry", expanded=False):
    with st.form("manual_entry_form"):
        st.markdown("**Add a New Specimen Record**")
        m_spp = st.text_input("Species Name:", placeholder="e.g., Lithospermum ruderale")
        
        m_date = st.date_input("Collection Date:", value=datetime.now().date())
        m_yr = m_date.year
        m_doy = m_date.timetuple().tm_yday
        
        c_m1, c_m2 = st.columns(2)
        with c_m1: m_col = st.text_input("Collector:")
        with c_m2: m_col_num = st.text_input("Col. Number:")
        
        c_m3, c_m4 = st.columns(2)
        with c_m3: m_barcode = st.text_input("Barcode:")
        with c_m4: m_elev = st.number_input("Elev. Override:", format="%.2f", value=0.0)
        
        c3, c4 = st.columns(2)
        with c3: m_lat = st.number_input("Latitude:", format="%.5f", value=0.0)
        with c4: m_lon = st.number_input("Longitude:", format="%.5f", value=0.0)
        
        if st.form_submit_button("Add & Process Record", use_container_width=True):
            if not m_spp or m_lat == 0.0 or m_lon == 0.0:
                st.error("Species, Latitude, and Longitude are required!")
            else:
                new_record = [{
                    "Data_Source": "Manual Entry", "Collector": m_col, "Col_Number": m_col_num,
                    "Barcode": m_barcode, "Species": m_spp, "Year": m_yr, "DOY": m_doy,
                    "Latitude": m_lat, "Longitude": m_lon, 
                    "Elevation": m_elev if m_elev != 0.0 else pd.NA,
                    "URL": "Manual", "Flowering": False, "Fruiting": False, "Vegetative": False
                }]
                pipeline_enrich_and_save(pd.DataFrame(new_record), target_limit=1, max_per_year=1)

with st.sidebar.expander("🌐 Fetch from GBIF", expanded=False):
    gbif_spp = st.text_input("Species Name (GBIF):", key="g_spp")
    col_yr1, col_yr2 = st.columns(2)
    with col_yr1: g_start = st.number_input("Start Year:", min_value=1800, max_value=2026, value=1950)
    with col_yr2: g_end = st.number_input("End Year:", min_value=1800, max_value=2026, value=2026)
    col_lim1, col_lim2 = st.columns(2)
    with col_lim1: g_limit = st.number_input("Total Records:", min_value=5, max_value=200, value=25, step=5)
    with col_lim2: g_max_yr = st.number_input("Max per Year:", min_value=1, max_value=20, value=3)
    
    if st.button("Fetch & Process GBIF", type="primary", use_container_width=True):
        with st.spinner("Querying GBIF..."):
            spp_encoded = urllib.parse.quote(gbif_spp)
            url = f"https://api.gbif.org/v1/occurrence/search?scientificName={spp_encoded}&year={g_start},{g_end}&limit={min(g_limit * 6, 300)}&hasCoordinate=true&basisOfRecord=PRESERVED_SPECIMEN"
            raw_records = []
            seen_col_nums = set()
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    for obs in res.json().get('results', []):
                        rec_url = obs.get('references', '')
                        if not rec_url and obs.get('media'): 
                            rec_url = obs.get('media')[0].get('identifier', '')
                        if not rec_url: continue
                        col_num = obs.get('recordNumber', '')
                        if col_num:
                            if col_num in seen_col_nums: continue
                            seen_col_nums.add(col_num)
                        
                        y, m, d, doy = obs.get('year'), obs.get('month'), obs.get('day'), pd.NA
                        if not y and obs.get('eventDate'):
                            try: y = int(obs['eventDate'][:4])
                            except: pass
                        if y and m and d:
                            try: doy = datetime(int(y), int(m), int(d)).timetuple().tm_yday
                            except: pass
                        
                        raw_records.append({
                            "Data_Source": "GBIF Herbarium", "Collector": obs.get('recordedBy', ''),
                            "Col_Number": col_num, "Barcode": obs.get('catalogNumber', ''),
                            "Species": obs.get('species', gbif_spp), "Year": safe_int(y) if y else pd.NA,
                            "DOY": doy, "Latitude": obs.get('decimalLatitude'),
                            "Longitude": obs.get('decimalLongitude'), "Elevation": obs.get('elevation', pd.NA),
                            "URL": rec_url, "Flowering": False, "Fruiting": False, "Vegetative": False
                        })
            except Exception as e: st.sidebar.error(f"GBIF Error: {e}")
            pipeline_enrich_and_save(pd.DataFrame(raw_records), target_limit=g_limit, max_per_year=g_max_yr)

with st.sidebar.expander("📸 Fetch from iNaturalist", expanded=False):
    inat_spp = st.text_input("Species Name (iNat):", key="i_spp")
    col_in1, col_in2 = st.columns(2)
    with col_in1: i_start = st.number_input("Start Year:", min_value=1800, max_value=2026, value=2000)
    with col_in2: i_end = st.number_input("End Year:", min_value=1800, max_value=2026, value=2026)
    col_ilim1, col_ilim2 = st.columns(2)
    with col_ilim1: i_limit = st.number_input("Total Records:", min_value=5, max_value=200, value=25, step=5)
    with col_ilim2: i_max_yr = st.number_input("Max per Year:", min_value=1, max_value=20, value=3)
        
    if st.button("Fetch & Process iNaturalist", type="primary", use_container_width=True):
        with st.spinner(f"Downloading observations..."):
            spp_encoded = urllib.parse.quote(inat_spp)
            url = f"https://api.inaturalist.org/v1/observations?taxon_name={spp_encoded}&d1={i_start}-01-01&d2={i_end}-12-31&per_page={min(i_limit * 5, 200)}&quality_grade=research"
            raw_records = []
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    for obs in res.json().get('results', []):
                        rec_url = obs.get('uri', '')
                        if not rec_url: continue
                        if obs.get('location') and obs.get('observed_on'):
                            lat_str, lon_str = obs['location'].split(',')
                            try:
                                dt = datetime.strptime(obs['observed_on'], "%Y-%m-%d")
                                obs_year, obs_doy = dt.year, dt.timetuple().tm_yday
                            except:
                                obs_year, obs_doy = pd.NA, pd.NA
                                
                            raw_records.append({
                                "Data_Source": "iNaturalist", "Collector": obs.get('user', {}).get('login', ''),
                                "Col_Number": "", "Barcode": str(obs.get('id', '')),
                                "Species": obs.get('taxon', {}).get('name', inat_spp),
                                "Year": obs_year, "DOY": obs_doy,
                                "Latitude": float(lat_str), "Longitude": float(lon_str), "Elevation": pd.NA,
                                "URL": rec_url, "Flowering": False, "Fruiting": False, "Vegetative": False
                            })
            except Exception as e: st.sidebar.error(f"iNaturalist Error: {e}")
            pipeline_enrich_and_save(pd.DataFrame(raw_records), target_limit=i_limit, max_per_year=i_max_yr)

st.sidebar.write("---")
if st.sidebar.button("🗑️ Clear Entire Database"):
    pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)
    st.rerun()

# ==========================================
#          MAIN UI: DATAFRAME
# ==========================================
st.title("🌱 Phenology & Climate Dataset Builder")

df = pd.read_csv(db_file)

# Dynamic Source Filter
available_sources = df['Data_Source'].dropna().unique().tolist()
if not available_sources:
    available_sources = ["GBIF Herbarium", "iNaturalist", "Manual Entry"]

st.markdown("### Filter Data Sources")
selected_sources = st.multiselect(
    "Select which data sources to view in the ledger and graphs:",
    options=available_sources,
    default=available_sources
)

filtered_df = df[df['Data_Source'].isin(selected_sources)]

st.write("---")
st.subheader("Data Ledger")
edited_df = st.data_editor(
    filtered_df, 
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "URL": st.column_config.LinkColumn("Record Link"),
        "Year": st.column_config.NumberColumn("Year", format="%d"),
        "DOY": st.column_config.NumberColumn("DOY", format="%d")
    }
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("💾 Save Manual Edits", type="primary"):
        edited_df.to_csv(db_file, index=False)
        st.success("Changes saved successfully!")
with col_btn2:
    st.download_button(
        label="📥 Download Dataset (CSV)",
        data=edited_df.to_csv(index=False).encode('utf-8'),
        file_name="phenology_dataset.csv",
        mime="text/csv",
    )

# ==========================================
#          GRAPHING & TRENDS DASHBOARD
# ==========================================
st.write("---")
st.subheader("📊 Dynamic Data Explorer")

if filtered_df.empty:
    st.info("The ledger is currently empty for the selected data sources.")
else:
    # Get available species for filtering
    available_species = edited_df['Species'].dropna().unique().tolist()
    
    st.markdown("### Graphing Options")
    
    # New Species Filter
    selected_species = st.multiselect(
        "Filter by Species:",
        options=available_species,
        default=available_species
    )
    
    plot_df = edited_df[edited_df['Species'].isin(selected_species)].copy()
    
    for col in ['Year', 'DOY', 'Latitude', 'Longitude', 'Elevation']:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
            
    climate_cols = [c for c in plot_df.columns if c.startswith('Y_') or c.startswith('N_')]
    for col in climate_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

    numeric_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
    plot_df = plot_df.dropna(subset=numeric_cols)
    
    if len(numeric_cols) < 2 or len(plot_df) < 2:
        st.warning("Not enough valid numeric data to plot based on your current filters.")
    else:
        default_x_ix = numeric_cols.index('Year') if 'Year' in numeric_cols else 0
        default_y_ix = numeric_cols.index('DOY') if 'DOY' in numeric_cols else 1
        
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1: selected_x = st.selectbox("Select X-Axis:", options=numeric_cols, index=default_x_ix)
        with sel_col2: selected_y = st.selectbox("Select Y-Axis:", options=numeric_cols, index=default_y_ix)
            
        fig_explorer = px.scatter(
            plot_df, 
            x=selected_x, 
            y=selected_y, 
            color='Species' if 'Species' in plot_df.columns else None,
            symbol='Data_Source' if 'Data_Source' in plot_df.columns else None,
            hover_data=['Collector', 'Year', 'DOY'] if all(c in plot_df.columns for c in ['Collector', 'Year', 'DOY']) else None,
            labels={selected_x: selected_x, selected_y: selected_y},
            template="streamlit",
            title=f"{selected_y} vs. {selected_x}"
        )
        
        # Native Numpy Calculation (No statsmodels needed)
        x_vals = plot_df[selected_x].values
        y_vals = plot_df[selected_y].values
        
        if np.var(x_vals) > 0 and np.var(y_vals) > 0:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            r_matrix = np.corrcoef(x_vals, y_vals)
            r2 = r_matrix[0, 1]**2
            
            line_x = np.array([min(x_vals), max(x_vals)])
            line_y = slope * line_x + intercept
            
            fig_explorer.add_scatter(
                x=line_x, y=line_y, mode='lines', 
                name='Filtered Trend', 
                line=dict(color='black', dash='dash')
            )
            
            sign = "+" if intercept >= 0 else "-"
            st.info(f"📈 **Filtered Trendline Equation:** y = {slope:.3f}x {sign} {abs(intercept):.3f}  |  **R²:** {r2:.3f}")
        else:
            st.warning("Not enough variance in selected variables to calculate a trendline.")
        
        fig_explorer.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        fig_explorer.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_explorer, use_container_width=True)
