import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime
import urllib.parse
import time
import os

# ==========================================
#          APP SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="Phenology & Climate Tracker", layout="wide")

# Persistent memory for the API limit warning
if "climate_limit_hit" not in st.session_state:
    st.session_state.climate_limit_hit = False

db_file = "specimen_ledger.csv"
base_headers = [
    "Data_Source", "Collector", "Col_Number", "Barcode", "Species",
    "Year", "DOY", "Latitude", "Longitude", "Elevation",
    "Flowering", "Fruiting", "Vegetative", "URL"
]

# Create local database if it doesn't exist
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
    url = "https://climatena.ca/api/data/AnyPoint"
    params = {"lat": lat, "lon": lon, "el": el, "period": period}
    try:
        res = requests.get(url, params=params, timeout=10)
        
        # Catch both 429 (Too Many Requests) and 403 (Forbidden/Limit Exceeded)
        if res.status_code in [429, 403]: 
            return {"_LIMIT_REACHED": True}
            
        elif res.status_code == 200:
            data = res.json()
            # Failsafe if the API sends a 200 OK but includes an error message
            if isinstance(data, dict) and "limit" in str(data.get("error", "")).lower():
                return {"_LIMIT_REACHED": True}
            return data
            
    except: pass
    return {}

def thin_and_cap_data(df, target_limit, max_per_year):
    # Drop rows missing coordinates or year to make sure we don't count junk
    valid_df = df.dropna(subset=['Latitude', 'Longitude', 'Year'])
    # Shuffle randomly
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    # Cap at max_per_year
    capped_df = valid_df.groupby('Year').head(max_per_year)
    # Enforce overall target_limit
    return capped_df.head(target_limit)

def pipeline_enrich_and_save(raw_df, target_limit, max_per_year=3):
    if raw_df.empty:
        st.sidebar.warning("No records found in that date range.")
        return

    st.sidebar.text(f"Applying de-clustering (Max {max_per_year}/yr)...")
    cleaned_df = thin_and_cap_data(raw_df, target_limit=target_limit, max_per_year=max_per_year)
    
    records = []
    progress_bar = st.sidebar.progress(0.0)
    status_text = st.sidebar.empty()
    
    # Check session state before we even start
    limit_reached_flag = st.session_state.climate_limit_hit 
    
    for count, (idx, row) in enumerate(cleaned_df.iterrows()):
        row_dict = row.to_dict()
        
        lat = safe_float(row_dict.get('Latitude'))
        lon = safe_float(row_dict.get('Longitude'))
        year = safe_int(row_dict.get('Year'))
        el = safe_float(row_dict.get('Elevation'))
        
        # Enforce clean Year assignment
        row_dict['Year'] = year if year is not None else pd.NA
        
        if lat is not None and lon is not None:
            # 1. Fetch Elevation
            if el is None or el == 0.0 or pd.isna(el):
                status_text.text(f"Fetching elevation... ({count+1}/{len(cleaned_df)})")
                fetched_el = get_elevation(lat, lon)
                el = fetched_el if fetched_el is not None else 0.0 
                row_dict['Elevation'] = el
            else:
                row_dict['Elevation'] = el
            
            # 2. Fetch Climate Data
            if year is not None:
                if not limit_reached_flag:
                    climate_year = min(year, 2022) # ClimateNA currently maxes at 2022
                    status_text.text(f"Fetching ClimateNA for {climate_year}... ({count+1}/{len(cleaned_df)})")
                    year_data = get_climate_data(lat, lon, el, f"Year_{climate_year}")
                    
                    if year_data.get("_LIMIT_REACHED"):
                        st.session_state.climate_limit_hit = True
                        limit_reached_flag = True
                    elif year_data: 
                        norm_data = get_climate_data(lat, lon, el, "Normal_1961_1990")
                        if norm_data.get("_LIMIT_REACHED"):
                            st.session_state.climate_limit_hit = True
                            limit_reached_flag = True
                        else:
                            for k, v in year_data.items(): row_dict[f"Y_{k}"] = v
                            for k, v in norm_data.items(): row_dict[f"N_{k}"] = v
                
        records.append(row_dict)
        
        # Safely calculate progress and ensure it never mathematically exceeds 1.0
        progress_val = min((count + 1) / len(cleaned_df), 1.0)
        progress_bar.progress(progress_val)
        
    status_text.text("Finished processing pipeline!")
    
    if records:
        final_new_df = pd.DataFrame(records)
        master_df = pd.read_csv(db_file)
        
        # Force all base headers to exist
        for col in base_headers:
            if col not in final_new_df.columns: final_new_df[col] = pd.NA
            if col not in master_df.columns: master_df[col] = pd.NA
            
        combined_df = pd.concat([master_df, final_new_df], ignore_index=True)
        
        # Sort columns so base headers are first, followed by dynamically generated climate columns
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

# Check and display the API limit warning persistently right at the top of the sidebar
if st.session_state.climate_limit_hit:
    st.sidebar.error("⚠️ **ClimateNA Limit Reached!**\n\nYou have exceeded the 50 queries per hour limit. Wait a while before querying more. New records will be added without climate fields in the meantime.")
    if st.sidebar.button("Dismiss Warning", type="primary"):
        st.session_state.climate_limit_hit = False
        st.rerun()

with st.sidebar.expander("✏️ Manual Entry", expanded=False):
    with st.form("manual_entry_form"):
        st.markdown("**Add a New Specimen Record**")
        m_spp = st.text_input("Species Name:", placeholder="e.g., Lithospermum ruderale")
        m_col = st.text_input("Collector:")
        
        c1, c2 = st.columns(2)
        with c1: m_yr = st.number_input("Year:", min_value=1800, max_value=2026, value=datetime.now().year)
        with c2: m_doy = st.number_input("DOY:", min_value=1, max_value=366, value=1)
        
        c3, c4 = st.columns(2)
        with c3: m_lat = st.number_input("Latitude:", format="%.5f", value=0.0)
        with c4: m_lon = st.number_input("Longitude:", format="%.5f", value=0.0)
        
        submit_manual = st.form_submit_button("Add & Process Record", use_container_width=True)
        
        if submit_manual:
            if not m_spp or m_lat == 0.0 or m_lon == 0.0:
                st.error("Species, Latitude, and Longitude are required!")
            else:
                new_record = [{
                    "Data_Source": "Manual Entry",
                    "Collector": m_col,
                    "Col_Number": "",
                    "Barcode": "",
                    "Species": m_spp,
                    "Year": m_yr,
                    "DOY": m_doy,
                    "Latitude": m_lat,
                    "Longitude": m_lon,
                    "Elevation": pd.NA,
                    "URL": "",
                    "Flowering": False, "Fruiting": False, "Vegetative": False
                }]
                # Push through the pipeline to auto-fetch elevation and climate!
                pipeline_enrich_and_save(pd.DataFrame(new_record), target_limit=1, max_per_year=1)

with st.sidebar.expander("🌐 Fetch from GBIF", expanded=False):
    gbif_spp = st.text_input("Species Name (GBIF):", placeholder="e.g., Lithospermum ruderale", key="g_spp")
    col_yr1, col_yr2 = st.columns(2)
    with col_yr1: g_start = st.number_input("Start Year:", min_value=1800, max_value=2026, value=1950, key="g_start")
    with col_yr2: g_end = st.number_input("End Year:", min_value=1800, max_value=2026, value=2026, key="g_end")
    
    col_lim1, col_lim2 = st.columns(2)
    with col_lim1: g_limit = st.number_input("Total Records:", min_value=5, max_value=200, value=25, step=5)
    with col_lim2: g_max_yr = st.number_input("Max per Year:", min_value=1, max_value=20, value=3, key="g_max")
    
    if st.button("Fetch & Process GBIF", type="primary", use_container_width=True):
        with st.spinner("Querying GBIF..."):
            spp_encoded = urllib.parse.quote(gbif_spp)
            fetch_limit = min(g_limit * 5, 300) # Over-fetch to ensure we have enough to survive thinning
            url = f"https://api.gbif.org/v1/occurrence/search?scientificName={spp_encoded}&year={g_start},{g_end}&limit={fetch_limit}&hasCoordinate=true&basisOfRecord=PRESERVED_SPECIMEN"
            
            raw_records = []
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    for obs in res.json().get('results', []):
                        rec_url = obs.get('references', '')
                        if not rec_url and obs.get('media'): rec_url = obs.get('media')[0].get('identifier', '')
                        
                        y = obs.get('year')
                        m = obs.get('month')
                        d = obs.get('day')
                        doy = pd.NA
                        
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
            
            pipeline_enrich_and_save(pd.DataFrame(raw_records), target_limit=g_limit, max_per_year=g_max_yr)

with st.sidebar.expander("📸 Fetch from iNaturalist", expanded=False):
    inat_spp = st.text_input("Species Name (iNat):", placeholder="e.g., Lithospermum ruderale", key="i_spp")
    col_in1, col_in2 = st.columns(2)
    with col_in1: i_start = st.number_input("Start Year:", min_value=1800, max_value=2026, value=2000, key="i_start")
    with col_in2: i_end = st.number_input("End Year:", min_value=1800, max_value=2026, value=2026, key="i_end")
    
    col_ilim1, col_ilim2 = st.columns(2)
    with col_ilim1: i_limit = st.number_input("Total Records:", min_value=5, max_value=200, value=25, step=5, key="i_lim")
    with col_ilim2: i_max_yr = st.number_input("Max per Year:", min_value=1, max_value=20, value=3, key="i_max")
        
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
                
            pipeline_enrich_and_save(pd.DataFrame(raw_records), target_limit=i_limit, max_per_year=i_max_yr)

st.sidebar.write("---")
if st.sidebar.button("🗑️ Clear Entire Database"):
    pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)
    st.rerun()

# ==========================================
#          MAIN UI: DATAFRAME
# ==========================================
st.title("🌱 Phenology & Climate Dataset Builder")
st.markdown("Use the sidebar to fetch data via API or enter data manually. The pipeline will automatically gather coordinates, pull elevation, and fetch **yearly & normal climate variables** via ClimateNA before adding them to your ledger.")

df = pd.read_csv(db_file)

st.subheader("Data Ledger")
edited_df = st.data_editor(
    df, 
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "URL": st.column_config.LinkColumn("Record Link"),
        "Year": st.column_config.NumberColumn("Year", format="%d"),
        "DOY": st.column_config.NumberColumn("DOY", format="%d")
    }
)

if st.button("💾 Save Manual Edits", type="primary"):
    edited_df.to_csv(db_file, index=False)
    st.success("Changes saved successfully!")


# ==========================================
#          GRAPHING & TRENDS DASHBOARD
# ==========================================
st.write("---")
st.subheader("📊 Dynamic Data Explorer")

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

    # Dynamically grab all numeric columns for the dropdowns
    numeric_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
    
    # Drop rows that have NaNs in the selected plotting columns to avoid trendline errors
    plot_df = plot_df.dropna(subset=numeric_cols)
    
    if len(numeric_cols) < 2 or len(plot_df) < 2:
        st.warning("Not enough valid numeric data to plot. Add more records to generate graphs.")
    else:
        st.markdown("Use the dropdowns below to explore relationships between any two variables in your dataset.")
        
        # Set sensible starting defaults (if they exist in the data)
        default_x_ix = numeric_cols.index('Year') if 'Year' in numeric_cols else 0
        default_y_ix = numeric_cols.index('DOY') if 'DOY' in numeric_cols else 1
        
        # Place dropdown selectors side-by-side
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            selected_x = st.selectbox("Select X-Axis:", options=numeric_cols, index=default_x_ix)
        with sel_col2:
            selected_y = st.selectbox("Select Y-Axis:", options=numeric_cols, index=default_y_ix)
            
        # Generate the dynamic scatter plot with a global trendline
        fig_explorer = px.scatter(
            plot_df, 
            x=selected_x, 
            y=selected_y, 
            color='Species' if 'Species' in plot_df.columns else None,
            hover_data=['Collector', 'Year', 'DOY'] if all(c in plot_df.columns for c in ['Collector', 'Year', 'DOY']) else None,
            labels={selected_x: selected_x, selected_y: selected_y},
            template="streamlit",
            title=f"{selected_y} vs. {selected_x}",
            trendline="ols",
            trendline_scope="overall" # Calculates one trendline for all data instead of per-species
        )
        
        # Extract the equation and R-squared value to place on the chart
        try:
            results = px.get_trendline_results(fig_explorer)
            if not results.empty:
                model = results.iloc[0]["px_fit_results"]
                slope = model.params.iloc[1]
                intercept = model.params.iloc[0]
                r2 = model.rsquared
                
                # Format the intercept sign nicely
                sign = "+" if intercept >= 0 else "-"
                eq_text = f"<b>y = {slope:.3f}x {sign} {abs(intercept):.3f}</b><br>R² = {r2:.3f}"
                
                # Add a text box to the top left of the graph
                fig_explorer.add_annotation(
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    text=eq_text, showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray", borderwidth=1,
                    xanchor="left", yanchor="top"
                )
        except Exception as e:
            # Silently pass if statsmodels is missing or data is perfectly vertical/horizontal
            pass 
        
        # Make the chart look clean and utilize the page width
        fig_explorer.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_explorer, use_container_width=True)
