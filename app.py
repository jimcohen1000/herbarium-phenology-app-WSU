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
#      OPTION A: DATA CLEANING FUNCTIONS
# ==========================================

def remove_duplicate_collections(df):
    """Safely drops duplicate sheets from the same collection event."""
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
    """Randomly samples up to `max_per_year` rows for any given year."""
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
    """Processes incoming raw data from any source and appends it to the master ledger."""
    if new_raw_df.empty:
        st.sidebar.warning("No data found to import.")
        return

    cleaned_df = remove_duplicate_collections(new_raw_df)
    cleaned_df = thin_data_by_year(cleaned_df, max_per_year=3)
    
    master_df = pd.read_csv(db_file)
    combined_df = pd.concat([master_df, cleaned_df], ignore_index=True)
    combined_df.to_csv(db_file, index=False)
    st.sidebar.success(f"Processed & added {len(cleaned_df)} filtered rows!")
    time.sleep(1.2)
    st.rerun()


# ==========================================
#     RESTORED SIDEBAR DATA ENTRY PANEL
# ==========================================
with st.sidebar:
    st.title("📥 Data Import Panel")
    st.write("Fetch or upload raw datasets directly into the ledger pipeline.")
    
    source_type = st.radio("Choose Input Source:", ["🌐 GBIF API", "📸 iNaturalist", "📁 CSV Spreadsheets"])
    
    if source_type == "🌐 GBIF API":
        st.markdown("### GBIF Parameters")
        species_input_1 = st.text_input("Species Name:", placeholder="e.g., Quercus alba")
        limit_1 = st.number_input("Max records:", min_value=10, max_value=1000, value=100, step=10)
        
        if st.button("Fetch & Process Data", key="gbif_btn", type="primary", use_container_width=True):
            with st.spinner("Calling GBIF API..."):
                # --- PLUG IN YOUR GBIF RETRIEVAL LOGIC HERE ---
                raw_fetched_df = pd.DataFrame([{
                    "Data_Source": "GBIF", "Collector": "Jane Doe", "Col_Number": "104A", 
                    "Species": species_input_1, "Year": 2024, "Latitude": 45.1234, "Longitude": -123.4567, "DOY": 142
                }])
                pipeline_clean_and_save(raw_fetched_df)

    elif source_type == "📸 iNaturalist":
        st.markdown("### iNaturalist Parameters")
        species_input_2 = st.text_input("Species Name:", placeholder="e.g., Acer rubrum")
        
        if st.button("Fetch & Process Data", key="inat_btn", type="primary", use_container_width=True):
            with st.spinner("Calling iNaturalist API..."):
                # --- PLUG IN YOUR iNATURALIST RETRIEVAL LOGIC HERE ---
                raw_fetched_df = pd.DataFrame([{
                    "Data_Source": "iNaturalist", "Collector": "CitizenSci12", "Col_Number": "", 
                    "Species": species_input_2, "Year": 2025, "Latitude": 42.9876, "Longitude": -122.1111, "DOY": 165
                }])
                pipeline_clean_and_save(raw_fetched_df)

    elif source_type == "📁 CSV Spreadsheets":
        st.markdown("### File Uploader")
        uploaded_file = st.file_uploader("Upload raw file (.csv)", type=["csv"])
        
        if uploaded_file is not None:
            if st.button("Process Spreadsheet File", type="primary", use_container_width=True):
                try:
                    raw_uploaded_df = pd.read_csv(uploaded_file)
                    pipeline_clean_and_save(raw_uploaded_df)
                except Exception as e:
                    st.error(f"Error reading spreadsheet: {e}")


# ==========================================
#      MAIN LAYOUT & DATA LEDGER
# ==========================================
st.title("🌱 Herbarium Specimen Tracker & Climate Database")

# Load current state of the database
df = pd.read_csv(db_file)

st.subheader("📋 Formatted Database Ledger")
if not df.empty:
    df = df.sort_values(by=["Year", "DOY"], ascending=[False, False])
df = df.reset_index(drop=True)

dynamic_key = f"herbarium_ledger_{len(df.columns)}"
edited_df = st.data_editor(
    df, 
    key=dynamic_key,
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

# Action controls layout
col_save, col_dl_clean, col_dl_full = st.columns([1, 1.2, 1.2])
with col_save:
    if st.button("💾 Save Ledger Edits", type="primary", use_container_width=True):
        edited_df.to_csv(db_file, index=False)
        st.success("Database updated successfully!")
        time.sleep(1)
        st.rerun()

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
with col_dl_clean:
    clean_csv = b""
    if os.path.exists(db_file) and not df.empty:
        existing_main_cols = [c for c in base_headers if c in df.columns]
        clean_csv = edited_df[existing_main_cols].to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Download Clean Ledger", data=clean_csv, file_name=f"herbarium_clean_{current_time}.csv", mime="text/csv", use_container_width=True, disabled=(len(clean_csv)==0))

with col_dl_full:
    full_csv_bytes = b""
    if os.path.exists(db_file) and not df.empty:
        full_csv_bytes = edited_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📦 Download Full CSV (with Climate)", data=full_csv_bytes, file_name=f"herbarium_full_{current_time}.csv", mime="text/csv", use_container_width=True, disabled=(len(full_csv_bytes)==0))


# ==========================================
#     ADDED BLOCK: GRAPHING & TRENDS
# ==========================================
st.write("---")
st.subheader("📊 Climate Trend Analytics")

if df.empty:
    st.info("The ledger is currently empty. Import or upload some specimen data to view analytical trend charts.")
else:
    # Safe numerical parsing for visual mappings
    plot_df = edited_df.copy()
    plot_df['Year'] = pd.to_numeric(plot_df['Year'], errors='coerce')
    plot_df['DOY'] = pd.to_numeric(plot_df['DOY'], errors='coerce')
    plot_df['Latitude'] = pd.to_numeric(plot_df['Latitude'], errors='coerce')
    
    # Check if Climate expansion calculations exist (e.g., Y_MAT / Mean Annual Temperature column)
    if 'Y_MAT' not in plot_df.columns:
        # Mock dummy data tracking for fallback if climate expansion script hasn't run yet
        plot_df['Y_MAT'] = 12.5 + (plot_df['Year'] - 2000) * 0.04
        has_climate_data = False
    else:
        plot_df['Y_MAT'] = pd.to_numeric(plot_df['Y_MAT'], errors='coerce')
        has_climate_data = True

    plot_df = plot_df.dropna(subset=['Year', 'DOY'])
    
    if not plot_df.empty:
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            st.markdown("**Chronological Trend: Mean Annual Temp vs. Collection Year**")
            # Build aggregated annual mean temperature trendline
            yearly_summary = plot_df.groupby('Year')['Y_MAT'].mean().reset_index()
            fig_temp = px.line(
                yearly_summary, x='Year', y='Y_MAT', 
                labels={'Y_MAT': 'Mean Annual Temp (°C)', 'Year': 'Collection Year'},
                markers=True, template="streamlit"
            )
            fig_temp.update_layout(margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_temp, use_container_width=True)
            if not has_climate_data:
                st.caption("💡 *Note: Showing placeholder trends. Run your Climate API tool mapping to populate actual 'Y_MAT' columns.*")

        with g_col2:
            st.markdown("**Phenological Profile: Collection Day of Year vs. Latitude**")
            # Build scatter assessment pinpointing geographic phenology distributions
            fig_pheno = px.scatter(
                plot_df, x='Latitude', y='DOY', color='Species' if 'Species' in plot_df.columns else None,
                hover_data=['Collector', 'Year'],
                labels={'DOY': 'Day of Year (DOY)', 'Latitude': 'Latitude (°N)'},
                template="streamlit"
            )
            fig_pheno.update_layout(margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig_pheno, use_container_width=True)
    else:
        st.warning("Please ensure rows contain valid values for 'Year' and 'DOY' parameters to calculate graphics.")

# Wipe database backup tool down below
with st.expander("⚠️ Danger Zone"):
    if st.button("Wipe Entire Database", type="secondary"):
        if os.path.exists(db_file):
            pd.read_csv(db_file).to_csv(f"herbarium_backup_{current_time}.csv", index=False)
        pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)
        st.success("Database wiped! Reloading...")
        time.sleep(1)
        st.rerun()
