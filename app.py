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
    Avoids the 'NaN trap' so missing collector numbers don't wipe out iNaturalist data.
    """
    if df.empty:
        return df
    
    df = df.copy()
    # Normalize empty or string-represented missing values to true pandas NA
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


def save_with_ordered_columns(df, filepath):
    """Saves the database preserving columns layout."""
    df.to_csv(filepath, index=False)


# ==========================================
#      DATA INGESTION PIPELINE (EXAMPLE)
# ==========================================
# This represents wherever you upload, scrape, or fetch new incoming data rows.

def process_new_incoming_data(new_raw_df):
    """Applies Option A cleaning BEFORE appending data or running climate steps."""
    st.info("Applying Option A data filters...")
    
    # 1. Strip identical collection events
    cleaned_df = remove_duplicate_collections(new_raw_df)
    
    # 2. Flatten out temporal density clusters (Capped at 3 per year)
    cleaned_df = thin_data_by_year(cleaned_df, max_per_year=3)
    
    st.success(f"Filters applied! Reduced row count from {len(new_raw_df)} to {len(cleaned_df)}.")
    
    # [YOUR CLIMATE API FETCH LOGIC GOES HERE]
    # e.g., climate_df = fetch_climatena_data(cleaned_df)
    
    return cleaned_df


# --- MAIN APP LAYOUT ---
st.title("🌱 Herbarium Specimen Tracker & Climate Database")

# Load existing working database
df = pd.read_csv(db_file)

# --- TABLE SECTION ---
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
        "Vegetative": st.column_config.CheckboxColumn("Vegetative"),
        "Species": st.column_config.TextColumn("Species") 
    }
)

# Action controls layout
col_save, col_dl_clean, col_dl_full = st.columns([1, 1.2, 1.2])

with col_save:
    if st.button("💾 Save Ledger Edits", type="primary", width="stretch"):
        save_with_ordered_columns(edited_df, db_file)
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
