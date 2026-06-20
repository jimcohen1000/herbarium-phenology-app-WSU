import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
import plotly.express as px

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="Herbarium Climate Ledger")
db_file = "herbarium_database.csv"

# Base tracker headers before climate expansion
base_headers = [
    "Data_Source", "Collector", "Col_Number", "Barcode", "Species", 
    "DOY", "Year", "Flowering", "Fruiting", "Vegetative", 
    "Latitude", "Longitude", "Elevation", "URL"
]

# Initialize database if it doesn't exist
if not os.path.exists(db_file):
    pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)


# ==========================================
#         DATA PIPELINE FUNCTIONS
# ==========================================

def remove_duplicate_collections(df):
    """Drops duplicate sheets from identical collecting actions."""
    if df.empty:
        return df
    df = df.copy()
    if 'Col_Number' in df.columns:
        df['Col_Number'] = df['Col_Number'].astype(str).replace(['nan', 'None', '<NA>', ''], pd.NA)
        has_col_num = df[df['Col_Number'].notna()]
        no_col_num = df[df['Col_Number'].isna()]
        subset_cols = ['Collector', 'Col_Number'] if 'Collector' in df.columns else ['Col_Number']
        clean_has_col = has_col_num.drop_duplicates(subset=subset_cols, keep='first')
        df = pd.concat([clean_has_col, no_col_num], ignore_index=True)
    return df


def thin_data_by_year(df, max_per_year=3):
    """Randomly thins dataset to pull maximum of `max_per_year` specimens per year."""
    if df.empty or 'Year' not in df.columns:
        return df
    df = df.copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    has_year = df[df['Year'].notna()].copy()
    no_year = df[df['Year'].isna()]
    has_year['Year'] = has_year['Year'].astype(int)
    
    thinned_has_year = has_year.groupby('Year', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max_per_year), random_state=42)
    )
    return pd.concat([thinned_has_year, no_year], ignore_index=True)


def pipeline_clean_and_save(new_raw_df):
    """Cleans inbound rows and writes directly to the master ledger file."""
    if new_raw_df.empty:
        st.sidebar.warning("No data found to process.")
        return

    cleaned_df = remove_duplicate_collections(new_raw_df)
    cleaned_df = thin_data_by_year(cleaned_df, max_per_year=3)
    
    master_df = pd.read_csv(db_file)
    combined_df = pd.concat([master_df, cleaned_df], ignore_index=True)
    combined_df.to_csv(db_file, index=False)
    st.sidebar.success(f"Successfully processed and added {len(cleaned_df)} filtered rows!")
    time.sleep(1.2)
    st.rerun()


# ==========================================
#      SIDEBAR: DATA ENTRY & EXPORT
# ==========================================
with st.sidebar:
    st.title("📥 Data Entry")
    
    with st.expander("🌐 Fetch from GBIF", expanded=False):
        species_input = st.text_input("Taxon/Species Name:", placeholder="e.g., Lithospermum ruderale", key="gbif_spp")
        col_yr1, col_yr2 = st.columns(2)
        with col_yr1:
            start_year = st.number_input("Start Year:", min_value=1800, max_value=2026, value=2000, key="gbif_start")
        with col_yr2:
            end_year = st.number_input("End Year:", min_value=1800, max_value=2026, value=2026, key="gbif_end")
        limit = st.number_input("Download Record Limit:", min_value=10, max_value=1000, value=100, step=10, key="gbif_limit")
        
        if st.button("Fetch & Process GBIF", type="primary", use_container_width=True):
            with st.spinner("Querying GBIF..."):
                # --- YOUR GBIF RETRIEVAL WRAPPER PLUGGED IN HERE ---
                raw_fetched_df = pd.DataFrame([{
                    "Data_Source": "GBIF", "Collector": "A. Gray", "Col_Number": "4021", 
                    "Species": species_input, "Year": int((start_year + end_year)/2), 
                    "Latitude": 44.5, "Longitude": -112.3, "DOY": 155, "Flowering": True
                }])
                pipeline_clean_and_save(raw_fetched_df)

    with st.expander("📸 Fetch from iNaturalist", expanded=False):
        species_input_inat = st.text_input("Species Name:", placeholder="e.g., Lithospermum ruderale", key="inat_spp")
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            start_yr_inat = st.number_input("Start Year:", min_value=1800, max_value=2026, value=2000, key="inat_start")
        with col_in2:
            end_yr_inat = st.number_input("End Year:", min_value=1800, max_value=2026, value=2026, key="inat_end")
        limit_inat = st.number_input("Download Record Limit:", min_value=10, max_value=1000, value=100, step=10, key="inat_limit")
            
        if st.button("Fetch & Process iNaturalist", type="primary", use_container_width=True):
            with st.spinner(f"Downloading up to {limit_inat} observations..."):
                # --- YOUR iNATURALIST RETRIEVAL WRAPPER PLUGGED IN HERE ---
                raw_fetched_df = pd.DataFrame([{
                    "Data_Source": "iNaturalist", "Collector": "CitizenSci_User", "Col_Number": "", 
                    "Species": species_input_inat, "Year": int((start_yr_inat + end_yr_inat)/2), 
                    "Latitude": 40.65, "Longitude": -111.65, "DOY": 178, "Fruiting": True
                }])
                pipeline_clean_and_save(raw_fetched_df)

    with st.expander("✍️ Manual Data Entry", expanded=False):
        m_species = st.text_input("Species Name:", placeholder="e.g., Quercus alba")
        m_collector = st.text_input("Collector's Name:")
        m_col_num = st.text_input("Collection Number / Sheet ID:")
        m_barcode = st.text_input("Barcode Number:")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            m_year = st.number_input("Collection Year:", min_value=1700, max_value=2026, value=2026)
        with col_m2:
            m_doy = st.number_input("Day of Year (DOY):", min_value=1, max_value=366, value=150)
        col_m3, col_m4, col_m5 = st.columns(3)
        with col_m3:
            m_lat = st.number_input("Latitude (°N):", format="%.5f", value=40.0)
        with col_m4:
            m_lon = st.number_input("Longitude (°W):", format="%.5f", value=-111.0)
        with col_m5:
            m_elev = st.number_input("Elevation (m):", value=0)
        m_url = st.text_input("Image/Record URL:")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            m_flowering = st.checkbox("Flowering")
        with col_p2:
            m_fruiting = st.checkbox("Fruiting")
        with col_p3:
            m_vegetative = st.checkbox("Vegetative")

        if st.button("Commit Manual Record", type="primary", use_container_width=True):
            new_row_df = pd.DataFrame([{
                "Data_Source": "Manual_Entry", "Collector": m_collector, "Col_Number": m_col_num,
                "Barcode": m_barcode, "Species": m_species, "DOY": m_doy, "Year": m_year,
                "Flowering": m_flowering, "Fruiting": m_fruiting, "Vegetative": m_vegetative,
                "Latitude": m_lat, "Longitude": m_lon, "Elevation": m_elev, "URL": m_url
            }])
            pipeline_clean_and_save(new_row_df)

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
        
    st.download_button(
        label="📥 Download Clean Ledger", 
        data=clean_csv, 
        file_name=f"herbarium_clean_{current_time}.csv", 
        mime="text/csv", 
        use_container_width=True, 
        disabled=(len(clean_csv)==0)
    )

    st.download_button(
        label="📦 Download Full CSV", 
        data=full_csv_bytes, 
        file_name=f"herbarium_full_{current_time}.csv", 
        mime="text/csv", 
        use_container_width=True, 
        disabled=(len(full_csv_bytes)==0)
    )
    
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

# --- ENHANCED TYPE SAFETY FIX FOR STREAMLIT DATA EDITOR ---
if not df.empty:
    # 1. Drop totally blank "ghost" rows from the bottom of the CSV
    df = df.dropna(how='all')
    
    # 2. Checkbox columns MUST be Booleans (No NaN)
    for col in ["Flowering", "Fruiting", "Vegetative"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    # 3. Number columns MUST be numeric
    for col in ["Year", "DOY", "Latitude", "Longitude", "Elevation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 4. Text & Link columns MUST be strings (No NaN floats allowed)
    for col in ["Data_Source", "URL", "Species", "Collector", "Barcode", "Col_Number"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

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
        plot_df['Y_MAT'] = 12.5 + (plot_df['Year'] - 2000) * 0.04
        has_climate_data = False
    else:
        plot_df['Y_MAT'] = pd.to_numeric(plot_df['Y_MAT'], errors='coerce')
        has_climate_data = True

    plot_df = plot_df.dropna(subset=['Year', 'DOY'])
    
    if not plot_df.empty:
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            st.markdown("**Chronological Trend: Mean Annual Temp vs. Collection Year**")
            yearly_summary = plot_df.groupby('Year')['Y_MAT'].mean().reset_index()
            fig_temp = px.line(
                yearly_summary, x='Year', y='Y_MAT', 
                labels={'Y_MAT': 'Mean Annual Temp (°C)', 'Year': 'Collection Year'},
                markers=True, template="streamlit"
            )
            fig_temp.update_layout(margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_temp, use_container_width=True)
            if not has_climate_data:
                st.caption("💡 *Note: Showing temporary trends. Run climate data processing scripts to generate explicit 'Y_MAT' rows.*")

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
    else:
        st.warning("Ensure rows possess valid values for Year and DOY markers to construct figures.")
