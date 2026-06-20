import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime
import urllib.parse
import time
import os
import numpy as np
import streamlit.components.v1 as components

# ==========================================
#          APP SETUP & CONFIG
# ==========================================
st.set_page_config(page_title="Phenology & Climate Tracker", layout="wide")

db_file = "specimen_ledger.csv"
base_headers = [
    "Data_Source", "Collector", "Col_Number", "Barcode", "Species",
    "Year", "DOY", "Latitude", "Longitude", "Elevation",
    "Phenology_Scored", "Flowering", "Fruiting", "Vegetative", "URL",
    "Y_3Mo_Tmean", "N_3Mo_Tmean", "Tmean_Anomaly"
]

if not os.path.exists(db_file):
    pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)
else:
    _df = pd.read_csv(db_file)
    needs_save = False
    for col in base_headers:
        if col not in _df.columns:
            _df[col] = False if col in ["Phenology_Scored", "Flowering", "Fruiting", "Vegetative"] else pd.NA
            needs_save = True
    if needs_save: _df.to_csv(db_file, index=False)

# ==========================================
#        DATA PIPELINE FUNCTIONS
# ==========================================

def safe_float(val):
    try: return float(val) if pd.notna(val) and val != '' else None
    except: return None

def safe_int(val):
    try: return int(float(val)) if pd.notna(val) and val != '' else None
    except: return None

def parse_elev(val):
    try: return float(val) if val and str(val).strip() != "" else None
    except: return None

def calc_prior_3_months(year, doy):
    try:
        dt = datetime(year, 1, 1) + pd.Timedelta(days=doy-1)
        target_months, target_years = [], []
        for i in range(1, 4):
            m = dt.month - i
            y = year
            if m <= 0:
                m += 12
                y -= 1
            target_months.append(m)
            target_years.append(y)
        return target_years, target_months
    except: return None, None

def get_elevation(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            elevations = res.json().get('elevation')
            if elevations and len(elevations) > 0: return float(elevations[0])
    except: return None
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
    except: return {}
    return {}

def save_with_ordered_columns(df_to_save, filepath):
    new_order = [c for c in base_headers if c in df_to_save.columns]
    new_order += [c for c in df_to_save.columns if c not in new_order]
    df_to_save[new_order].to_csv(filepath, index=False)

def thin_and_cap_data(df, target_limit, max_per_year):
    valid_df = df.dropna(subset=['Latitude', 'Longitude', 'Year', 'URL']).copy()
    valid_df = valid_df[valid_df['URL'].str.strip() != '']
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    capped_df = valid_df.groupby('Year').head(max_per_year).sort_values('Year')
    unique_years = capped_df['Year'].unique()
    
    selected_years, last_y = [], -999
    for y in unique_years:
        if y - last_y >= 2: 
            selected_years.append(y)
            last_y = y
            
    spaced_df = capped_df[capped_df['Year'].isin(selected_years)]
    if len(spaced_df) < target_limit:
        remaining_df = capped_df[~capped_df['Year'].isin(selected_years)]
        spaced_df = pd.concat([spaced_df, remaining_df.head(target_limit - len(spaced_df))])
        
    return spaced_df.head(target_limit).sample(frac=1).reset_index(drop=True)

def pipeline_enrich_and_save(raw_df, target_limit, max_per_year=3):
    if raw_df.empty:
        st.sidebar.warning("No records found (or none passed the filters).")
        return

    st.sidebar.text(f"Applying de-clustering & year distribution...")
    cleaned_df = thin_and_cap_data(raw_df, target_limit=target_limit, max_per_year=max_per_year)
    
    records = []
    progress_bar = st.sidebar.progress(0.0)
    status_text = st.sidebar.empty()
    
    for count, (idx, row) in enumerate(cleaned_df.iterrows()):
        row_dict = row.to_dict()
        row_dict['Phenology_Scored'] = False
        
        lat, lon = safe_float(row_dict.get('Latitude')), safe_float(row_dict.get('Longitude'))
        year, doy = safe_int(row_dict.get('Year')), safe_int(row_dict.get('DOY'))
        el = safe_float(row_dict.get('Elevation'))
        
        if lat is not None and lon is not None:
            if el is None or el == 0.0 or pd.isna(el):
                status_text.text(f"Fetching elevation... ({count+1}/{len(cleaned_df)})")
                fetched_el = get_elevation(lat, lon)
                el = fetched_el if fetched_el is not None else 0.0 
                row_dict['Elevation'] = el
            
            if year is not None:
                climate_year = min(year, 2022) 
                status_text.text(f"Fetching ClimateNA for {climate_year}... ({count+1}/{len(cleaned_df)})")
                year_data = get_climate_data(lat, lon, el, f"Year_{climate_year}")
                if not year_data: year_data = get_climate_data(lat, lon, el, str(climate_year))
                norm_data = get_climate_data(lat, lon, el, "Normal_1961_1990")
                
                if year_data and norm_data:
                    for k, v in year_data.items(): 
                        if k not in ["ID1", "ID2", "lat", "lon", "el", "prd", "varYSM", "period"]: row_dict[f"Y_{k}"] = v
                    for k, v in norm_data.items(): 
                        if k not in ["ID1", "ID2", "lat", "lon", "el", "prd", "varYSM", "period"]: row_dict[f"N_{k}"] = v
                    
                    if doy is not None:
                        ty, tm = calc_prior_3_months(year, doy)
                        if ty and tm:
                            prev_year_data = {}
                            if min(ty) < year: prev_year_data = get_climate_data(lat, lon, el, f"Year_{min(ty)}")
                            
                            y_vals, n_vals = [], []
                            for y_t, m_t in zip(ty, tm):
                                m_str = f"{m_t:02d}"
                                n_val = norm_data.get(f"Tave{m_str}")
                                if n_val is not None: n_vals.append(float(n_val))
                                
                                target_data = year_data if y_t == year else prev_year_data
                                y_val = target_data.get(f"Tave{m_str}")
                                if y_val is not None: y_vals.append(float(y_val))
                            
                            if len(y_vals) == 3 and len(n_vals) == 3:
                                row_dict['Y_3Mo_Tmean'] = round(sum(y_vals)/3, 2)
                                row_dict['N_3Mo_Tmean'] = round(sum(n_vals)/3, 2)
                                row_dict['Tmean_Anomaly'] = round(row_dict['Y_3Mo_Tmean'] - row_dict['N_3Mo_Tmean'], 2)

        records.append(row_dict)
        progress_bar.progress(min((count + 1) / len(cleaned_df), 1.0))
        
    status_text.text("Finished processing pipeline!")
    
    if records:
        final_new_df = pd.DataFrame(records)
        master_df = pd.read_csv(db_file)
        combined_df = pd.concat([master_df, final_new_df], ignore_index=True)
        save_with_ordered_columns(combined_df, db_file)
        st.sidebar.success(f"Added {len(final_new_df)} processed records!")
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
                    "Latitude": m_lat, "Longitude": m_lon, "Elevation": m_elev if m_elev != 0.0 else pd.NA,
                    "URL": "Manual", "Flowering": False, "Fruiting": False, "Vegetative": False
                }]
                pipeline_enrich_and_save(pd.DataFrame(new_record), target_limit=1, max_per_year=1)

with st.sidebar.expander("🌐 Fetch from GBIF", expanded=False):
    gbif_spp = st.text_input("Species Name (GBIF):", key="g_spp")
    col_yr1, col_yr2 = st.columns(2)
    with col_yr1: g_start = st.number_input("Start Year:", min_value=1800, max_value=2026, value=1950, key="g_start_yr")
    with col_yr2: g_end = st.number_input("End Year:", min_value=1800, max_value=2026, value=2026, key="g_end_yr")
    col_lim1, col_lim2 = st.columns(2)
    with col_lim1: g_limit = st.number_input("Total Records:", min_value=5, max_value=200, value=25, step=5, key="g_limit_rec")
    with col_lim2: g_max_yr = st.number_input("Max per Year:", min_value=1, max_value=20, value=3, key="g_max_yr")
    g_states = st.text_input("State(s):", help="Comma-separated", key="g_states")
    g_counties = st.text_input("County(s):", help="Comma-separated", key="g_counties")
    col_e1, col_e2 = st.columns(2)
    with col_e1: g_emin = st.text_input("Min Elev (m):", key="g_emin")
    with col_e2: g_emax = st.text_input("Max Elev (m):", key="g_emax")
    
    if st.button("Fetch & Process GBIF", type="primary", use_container_width=True):
        with st.spinner("Paginating GBIF..."):
            spp_encoded = urllib.parse.quote(gbif_spp)
            e_min, e_max = parse_elev(g_emin), parse_elev(g_emax)
            base_url = f"https://api.gbif.org/v1/occurrence/search?scientificName={spp_encoded}&year={g_start},{g_end}&hasCoordinate=true&basisOfRecord=PRESERVED_SPECIMEN"
            if g_states:
                for s in g_states.split(','): base_url += f"&stateProvince={urllib.parse.quote(s.strip())}"
            if g_counties:
                for c in g_counties.split(','): base_url += f"&county={urllib.parse.quote(c.strip())}"
            if e_min is not None or e_max is not None: base_url += f"&elevation={e_min if e_min else ''},{e_max if e_max else ''}"

            raw_records, offset, end_of_records = [], 0, False
            while len(raw_records) < 3000 and not end_of_records and offset < 9000:
                try:
                    res = requests.get(base_url + f"&limit=300&offset={offset}", timeout=10)
                    if res.status_code == 200:
                        data = res.json()
                        results = data.get('results', [])
                        if not results: break
                        for obs in results:
                            rec_url = obs.get('references', '') or (obs.get('media')[0].get('identifier', '') if obs.get('media') else '')
                            if not rec_url: continue
                            y = obs.get('year') or (int(obs['eventDate'][:4]) if obs.get('eventDate') else None)
                            m, d = obs.get('month'), obs.get('day')
                            doy = datetime(int(y), int(m), int(d)).timetuple().tm_yday if y and m and d else pd.NA
                            raw_records.append({
                                "Data_Source": "GBIF Herbarium", "Collector": obs.get('recordedBy', ''),
                                "Species": obs.get('species', gbif_spp), "Year": safe_int(y) if y else pd.NA,
                                "DOY": doy, "Latitude": obs.get('decimalLatitude'), "Longitude": obs.get('decimalLongitude'), 
                                "Elevation": obs.get('elevation', pd.NA), "URL": rec_url, 
                                "Flowering": False, "Fruiting": False, "Vegetative": False
                            })
                        end_of_records = data.get('endOfRecords', True)
                        offset += 300
                    else: break
                except: break
            pipeline_enrich_and_save(pd.DataFrame(raw_records), target_limit=g_limit, max_per_year=g_max_yr)

with st.sidebar.expander("📸 Fetch from iNaturalist", expanded=False):
    inat_spp = st.text_input("Species Name (iNat):", key="i_spp")
    col_in1, col_in2 = st.columns(2)
    with col_in1: i_start = st.number_input("Start Year:", min_value=1800, max_value=2026, value=2000, key="i_start_yr")
    with col_in2: i_end = st.number_input("End Year:", min_value=1800, max_value=2026, value=2026, key="i_end_yr")
    col_ilim1, col_ilim2 = st.columns(2)
    with col_ilim1: i_limit = st.number_input("Total Records:", min_value=5, max_value=200, value=25, step=5, key="i_limit_rec")
    with col_ilim2: i_max_yr = st.number_input("Max per Year:", min_value=1, max_value=20, value=3, key="i_max_yr")
    i_states = st.text_input("State(s):", key="i_states")
    i_counties = st.text_input("County(s):", key="i_counties")
    col_ie1, col_ie2 = st.columns(2)
    with col_ie1: i_emin = st.text_input("Min Elev (m):", key="i_emin")
    with col_ie2: i_emax = st.text_input("Max Elev (m):", key="i_emax")
        
    if st.button("Fetch & Process iNat", type="primary", use_container_width=True):
        with st.spinner(f"Paginating iNaturalist..."):
            spp_encoded = urllib.parse.quote(inat_spp)
            e_min, e_max = parse_elev(i_emin), parse_elev(i_emax)
            s_list = [s.strip().lower() for s in i_states.split(',')] if i_states else []
            c_list = [c.strip().lower() for c in i_counties.split(',')] if i_counties else []
            
            raw_records, page = [], 1
            while len(raw_records) < 3000 and page <= 15:
                try:
                    url = f"https://api.inaturalist.org/v1/observations?taxon_name={spp_encoded}&d1={i_start}-01-01&d2={i_end}-12-31&per_page=200&page={page}&quality_grade=research"
                    res = requests.get(url, timeout=10)
                    if res.status_code == 200:
                        results = res.json().get('results', [])
                        if not results: break
                        for obs in results:
                            rec_url = obs.get('uri', '')
                            if not rec_url: continue
                            pg = obs.get('place_guess', '').lower()
                            if s_list and not any(s in pg for s in s_list): continue
                            if c_list and not any(c in pg for c in c_list): continue
                            
                            if obs.get('location') and obs.get('observed_on'):
                                lat_str, lon_str = obs['location'].split(',')
                                lat_flt, lon_flt = float(lat_str), float(lon_str)
                                fetched_elev = pd.NA
                                if e_min is not None or e_max is not None:
                                    el = get_elevation(lat_flt, lon_flt)
                                    if el is None or (e_min and el < e_min) or (e_max and el > e_max): continue
                                    fetched_elev = el
                                try:
                                    dt = datetime.strptime(obs['observed_on'], "%Y-%m-%d")
                                    obs_year, obs_doy = dt.year, dt.timetuple().tm_yday
                                except: obs_year, obs_doy = pd.NA, pd.NA
                                    
                                raw_records.append({
                                    "Data_Source": "iNaturalist", "Collector": obs.get('user', {}).get('login', ''),
                                    "Species": obs.get('taxon', {}).get('name', inat_spp), "Year": obs_year, "DOY": obs_doy,
                                    "Latitude": lat_flt, "Longitude": lon_flt, "Elevation": fetched_elev, "URL": rec_url, 
                                    "Flowering": False, "Fruiting": False, "Vegetative": False
                                })
                        if len(results) < 200: break 
                        page += 1
                    else: break
                except: break
            pipeline_enrich_and_save(pd.DataFrame(raw_records), target_limit=i_limit, max_per_year=i_max_yr)

st.sidebar.write("---")
if st.sidebar.button("🗑️ Clear Entire Database"):
    pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)
    st.rerun()

# ==========================================
#          MAIN UI: EXPLORER SETTINGS
# ==========================================
st.title("🌱 Phenology & Climate Dataset Builder")
df = pd.read_csv(db_file)

with st.expander("⚙️ Global Analysis & Outlier Settings (Affects Map & Graph)", expanded=True):
    if df.empty:
        st.info("Your database is empty. Fetch or add records first.")
        plot_df = pd.DataFrame()
    else:
        sources = df['Data_Source'].fillna('Unknown').unique().tolist()
        species = df['Species'].dropna().unique().tolist()
        
        c1, c2 = st.columns(2)
        with c1: sel_src = st.multiselect("Filter Data Source:", sources, default=sources)
        with c2: sel_spp = st.multiselect("Filter Species:", species, default=species)
        
        plot_df = df[df['Data_Source'].isin(sel_src) & df['Species'].isin(sel_spp)].copy()
        
        # Ensure numeric typing
        for col in ['Year', 'DOY', 'Latitude', 'Longitude', 'Elevation']:
            if col in plot_df.columns: plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        for col in [c for c in plot_df.columns if c.startswith('Y_') or c.startswith('N_')]:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

        num_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
        
        st.markdown("---")
        s_c1, s_c2, s_c3 = st.columns([1, 1, 2])
        if len(num_cols) >= 2:
            with s_c1: selected_x = st.selectbox("X-Axis (For Trendline):", num_cols, index=num_cols.index('Year') if 'Year' in num_cols else 0)
            with s_c2: selected_y = st.selectbox("Y-Axis (For Trendline & Outliers):", num_cols, index=num_cols.index('DOY') if 'DOY' in num_cols else 1)
            with s_c3:
                use_outlier_filter = st.checkbox(f"Highlight Outliers based on {selected_y}", value=False)
                if use_outlier_filter:
                    std_devs = st.slider("Flag data outside standard deviations:", 1.0, 5.0, 2.5, 0.1)
        
        # Calculate Outliers globally for the tabs
        plot_df['Is_Outlier'] = False
        if use_outlier_filter and len(plot_df.dropna(subset=[selected_y])) > 0:
            valid_y_df = plot_df.dropna(subset=[selected_y])
            mean_y, std_y = valid_y_df[selected_y].mean(), valid_y_df[selected_y].std()
            plot_df['Is_Outlier'] = np.abs(plot_df[selected_y] - mean_y) > (std_devs * std_y)
            
            outlier_count = plot_df['Is_Outlier'].sum()
            st.info(f"Identified **{outlier_count}** outliers (outside {mean_y:.1f} ± {std_devs*std_y:.1f}). They will be highlighted on the map and excluded from the graph trendline.")
        
        # Create a combined label for the Map so outliers stand out
        plot_df['Map_Label'] = plot_df['Species'] + plot_df['Is_Outlier'].apply(lambda x: ' 🔴 [OUTLIER]' if x else '')

# ==========================================
#          MAIN UI: TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["🗃️ Data Ledger", "🗺️ Map View", "📊 Trends & Outliers", "🎯 Rapid Scoring"])

with tab1:
    st.info("Edit your full CSV here. Checking the phenology boxes here also updates the 'Phenology_Scored' status.")
    edited_df = st.data_editor(
        df, num_rows="dynamic", use_container_width=True,
        column_config={
            "URL": st.column_config.LinkColumn("Record Link"),
            "Year": st.column_config.NumberColumn("Year", format="%d"),
            "DOY": st.column_config.NumberColumn("DOY", format="%d")
        }
    )
    for idx, row in edited_df.iterrows():
        if (row['Flowering'] or row['Fruiting'] or row['Vegetative']) and not row['Phenology_Scored']:
            edited_df.at[idx, 'Phenology_Scored'] = True

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("💾 Save Manual Edits", type="primary"):
            save_with_ordered_columns(edited_df, db_file)
            st.success("Database updated!")
    with col_btn2:
        st.download_button("📥 Download Full Dataset (CSV)", data=edited_df.to_csv(index=False).encode('utf-8'), file_name="phenology_dataset.csv", mime="text/csv")

with tab2:
    if plot_df.empty: st.warning("No data to map.")
    else:
        map_df = plot_df.dropna(subset=['Latitude', 'Longitude']).copy()
        if map_df.empty:
            st.warning("No coordinate data available in the filtered dataset.")
        else:
            map_df['Data_Source'] = map_df['Data_Source'].fillna('Unknown')
            fig_map = px.scatter_mapbox(
                map_df, lat="Latitude", lon="Longitude", color="Map_Label",
                hover_data=["Year", "DOY", "Data_Source", "Collector"],
                zoom=3, mapbox_style="carto-positron", title="Specimen Collection Sites"
            )
            fig_map.update_traces(marker=dict(size=9, opacity=0.8))
            fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    if plot_df.empty: st.info("No data available for graphing.")
    elif len(num_cols) < 2: st.warning("Not enough numeric columns to graph.")
    else:
        valid_plot_df = plot_df.dropna(subset=[selected_x, selected_y]).copy()
        
        # Convert boolean Is_Outlier to string for Plotly symbol mapping
        valid_plot_df['Point_Type'] = valid_plot_df['Is_Outlier'].apply(lambda x: 'Outlier' if x else 'Normal')
        
        if len(valid_plot_df) >= 2:
            fig = px.scatter(
                valid_plot_df, x=selected_x, y=selected_y, 
                color='Species' if 'Species' in valid_plot_df.columns else None,
                symbol='Point_Type', symbol_map={'Normal': 'circle', 'Outlier': 'x'},
                hover_data=['Collector', 'Year', 'DOY', 'Data_Source'], template="streamlit"
            )
            
            # Trendline Calculation (EXCLUDING Outliers)
            trend_df = valid_plot_df[valid_plot_df['Is_Outlier'] == False]
            if len(trend_df) > 1:
                x_v, y_v = trend_df[selected_x].values, trend_df[selected_y].values
                if np.var(x_v) > 0 and np.var(y_v) > 0:
                    slope, intercept = np.polyfit(x_v, y_v, 1)
                    r2 = np.corrcoef(x_v, y_v)[0, 1]**2
                    lx = np.array([min(x_v), max(x_v)])
                    fig.add_scatter(x=lx, y=slope * lx + intercept, mode='lines', name='Trend (Normals Only)', line=dict(color='black', dash='dash'))
                    st.success(f"📈 **Trendline (Excluding Outliers):** y = {slope:.3f}x {'+' if intercept>=0 else '-'} {abs(intercept):.3f}  |  **R²:** {r2:.3f}")
            
            fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("🎯 Rapid Image Scoring")
    unscored = df[(df['Phenology_Scored'] == False) & (df['URL'].notna()) & (df['URL'].str.strip() != '') & (df['URL'] != 'Manual')]
    
    if unscored.empty:
        st.success("🎉 All URL records in the database have been scored!")
    else:
        st.info(f"You have **{len(unscored)}** unscored records remaining.")
        target_idx = st.selectbox("Select Record:", unscored.index, format_func=lambda x: f"Index {x}: {df.loc[x, 'Species']} ({df.loc[x, 'Data_Source']}) - Year {df.loc[x, 'Year']}")
        
        row = df.loc[target_idx]
        st.markdown(f"### [🔗 Click Here to View Original Specimen Image/Page]({row['URL']})")
        
        # Scoring Toggles
        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1: f1 = st.checkbox("🌸 Flowering", value=bool(row['Flowering']), key=f"f1_{target_idx}")
        with c_s2: f2 = st.checkbox("🍒 Fruiting", value=bool(row['Fruiting']), key=f"f2_{target_idx}")
        with c_s3: f3 = st.checkbox("🍃 Vegetative (Leaves)", value=bool(row['Vegetative']), key=f"f3_{target_idx}")
        
        if st.button("💾 Save Phenology & Load Next", type="primary"):
            df.at[target_idx, 'Flowering'] = f1
            df.at[target_idx, 'Fruiting'] = f2
            df.at[target_idx, 'Vegetative'] = f3
            df.at[target_idx, 'Phenology_Scored'] = True
            save_with_ordered_columns(df, db_file)
            st.rerun()

        st.markdown("---")
        st.markdown("**Embedded Viewer:**")
        # Attempt to iframe the URL
        components.iframe(row['URL'], height=600, scrolling=True)
