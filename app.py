import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime

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
    """
    Safely drops duplicate sheets from the same collection event.
    Avoids the 'NaN trap' so missing collector numbers don't wipe out data.
    """
    if df.empty:
        return df
    
    df = df.copy()
    if 'Col_Number' in df.columns:
        df['Col_Number'] = df['Col_Number'].astype(str).replace(['nan', 'None', '<NA>', ''], pd.NA)
        
        # Split data: Rows with collection numbers vs rows without
        has_col_num = df[df['Col_Number'].notna()]
        no_col_num = df[df['Col_Number'].isna()]
        
        # Deduplicate strictly on valid collector event profiles
        subset_cols = ['Collector', 'Col_Number'] if 'Collector' in df.columns else ['Col_Number']
        clean_has_col = has_col_num.drop_duplicates(subset=subset_cols, keep='first')
        
        # Re-stitch dataset back together
        df = pd.concat([clean_has_col, no_col_num], ignore_index=True)
    return df


def thin_data_by_year(df, max_per_year=3):
    """
    Stratified thinning: Caps chronological data clusters.
    Randomly samples up to `max_per_year` rows for any given year to flatten timeline bias.
    """
    if df.empty or 'Year' not in df.columns:
        return df
        
    df = df.copy()
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Isolate records with years from records without
    has_year = df[df['Year'].notna()].copy()
    no_year = df[df['Year'].isna()]
    
    has_year['Year'] = has_year['Year'].astype(int)
    
    # Group by year and randomly sample down to the cap
    thinned_has_year = has_year.groupby('Year', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max_per_year), random_state=42)
    )
    
    return pd.concat([thinned_has_year, no_year], ignore_index=True)


def pipeline_clean_and_save(new_raw_df):
    """Processes incoming raw data from any source and appends it to the master ledger."""
    if new_raw_df.empty:
        st.warning("No data found to import.")
        return

    # 1. Apply your clean filters
    cleaned_df = remove_duplicate_collections(new_raw_df)
    cleaned_df = thin_data_by_year(cleaned_df, max_per_year=3)
    
    # [OPTIONAL: YOUR CLIMATE API FETCH CODE GOES HERE]
    # If your old script fetched climate coordinates automatically upon import,
    # you can run that function right here on `cleaned_df` before combining.

    # 2. Append to master database
    master_df = pd.read_csv(db_file)
    combined_df = pd.concat([master_df, cleaned_df], ignore_index=True)
    
    # Save it back to the file
    combined_df.to_csv(db_file, index=False)
    st.success(f"Successfully processed and added {len(cleaned_df)} filtered specimens to the ledger!")
    time.sleep(1.5)
    st.rerun()


# --- MAIN APP LAYOUT ---
st.title("🌱 Herbarium Specimen Tracker & Climate Database")

# ==========================================
#     RESTORED: THREE DATA SOURCES UI
# ==========================================
st.subheader("📥 Fetch & Import Specimen Data")
tab1, tab2, tab3 = st.tabs(["🌐 Source 1: GBIF API", "📸 Source 2: iNaturalist", "📁 Source 3: CSV Upload / Manual"])

with tab1:
    st.markdown("### Import from GBIF")
    species_input_1 = st.text_input("Enter Species Name (GBIF):", placeholder="e.g., Quercus alba")
    limit_1 = st.number_input("Max records to fetch from GBIF:", min_value=10, max_value=1000, value=100, step=10)
    
    if st.button("Fetch from GBIF", type="primary"):
        with st.spinner("Querying GBIF API..."):
            # --- PASTE YOUR ORIGINAL GBIF FETCHING LOGIC HERE ---
            # Create a placeholder DataFrame mirroring your original output format:
            raw_fetched_df = pd.DataFrame(columns=base_headers) 
            
            # (Example mock data row so the button runs out of the box)
            raw_fetched_df = pd.DataFrame([{
                "Data_Source": "GBIF", "Collector": "Jane Doe", "Col_Number": "104A", 
                "Species": species_input_1, "Year": 2024, "Latitude": 45.1234, "Longitude": -123.4567
            }])
            
            # Pass results directly through the pipeline
            pipeline_clean_and_save(raw_fetched_df)

with tab2:
    st.markdown("### Import from iNaturalist")
    species_input_2 = st.text_input("Enter Species Name (iNaturalist):", placeholder="e.g., Acer rubrum")
    
    if st.button("Fetch from iNaturalist", type="primary"):
        with st.spinner("Querying iNaturalist API..."):
            # --- PASTE YOUR ORIGINAL iNATURALIST FETCHING LOGIC HERE ---
            raw_fetched_df = pd.DataFrame(columns=base_headers)
            
            # (Example mock data row missing a collector number to test the 'NaN safety trap')
            raw_fetched_df = pd.DataFrame([{
                "Data_Source": "iNaturalist", "Collector": "CitizenSci12", "Col_Number": "", 
                "Species": species_input_2, "Year": 2025, "Latitude": 42.9876, "Longitude": -122.1111
            }])
            
            pipeline_clean_and_save(raw_fetched_df)

with tab3:
    st.markdown("### Custom CSV Data Upload")
    uploaded_file = st.file_uploader("Upload a raw specimen spreadsheet (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Process & Import Uploaded File", type="primary"):
            try:
                raw_uploaded_df = pd.read_csv(uploaded_file)
                pipeline_clean_and_save(raw_uploaded_df)
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")


# ==========================================
#      TABLE & LEDGER DISPLAY SECTION
# ==========================================
st.write("---")
st.subheader("📋 Formatted Database Ledger")

# Load current state of the database
df = pd.read_csv(db_file)

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
        "Vegetative": st.column_config.CheckboxColumn("Vegetative"),
        "Species": st.column_config.TextColumn("Species") 
    }
)

# Action controls layout
col_save, col_dl_clean, col_dl_full = st.columns([1, 1.2, 1.2])

with col_save:
    if st.button("💾 Save Ledger Edits", type="primary", width="stretch"):
        edited_df.to_csv(db_file, index=False)
        st.success("Database updated successfully!")
        time.sleep(1)
        st.rerun()

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

with col_dl_clean:
    clean_csv = b""
    if os.path.exists(db_file):
        try:
            clean_df = pd.read_csv(db_file)
            existing_main_cols = [c for c in base_headers if c in clean_df.columns]
            clean_csv = clean_df[existing_main_cols].to_csv(index=False).encode('utf-8-sig')
        except Exception as e:
            st.error(f"Clean Download Error: {e}")

    st.download_button(
        label="📥 Download Clean Ledger", 
        data=clean_csv, 
        file_name=f"herbarium_clean_{current_time}.csv", 
        mime="text/csv",
        width="stretch",
        disabled=(len(clean_csv) == 0),
        help="Downloads a lightweight sheet containing only the main columns visible in your ledger."
    )

with col_dl_full:
    full_csv_bytes = b""
    if os.path.exists(db_file):
        try:
            full_df = pd.read_csv(db_file)
            full_csv_bytes = full_df.to_csv(index=False).encode('utf-8-sig')
        except Exception as e:
            st.error(f"Full Download Error: {e}")
        
    st.download_button(
        label="📦 Download Full CSV (with Climate)", 
        data=full_csv_bytes, 
        file_name=f"herbarium_full_{current_time}.csv", 
        mime="text/csv",
        width="stretch",
        disabled=(len(full_csv_bytes) == 0),
        help="Downloads the complete database including all 500+ climate environment columns."
    )
    
with st.expander("⚠️ Danger Zone"):
    st.write("Wiping the database will clear your current view, but a timestamped backup will automatically be saved.")
    if st.button("Wipe Entire Database", type="secondary"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"herbarium_backup_{timestamp}.csv"
        if os.path.exists(db_file):
            pd.read_csv(db_file).to_csv(backup_file, index=False)
        pd.DataFrame(columns=base_headers).to_csv(db_file, index=False)
        st.success(f"Database wiped! Backed up to: {backup_file}")
        time.sleep(2) 
        st.rerun()
