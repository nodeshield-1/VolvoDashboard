import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Set Streamlit page config at the very top
st.set_page_config(page_title="Volvo Driving Journal", layout="wide")

# =====================
# LOAD AND CLEAN DATA
# =====================
def load_data():
    input_file = 'volvo_driving_journal.csv'
    try:
        df = pd.read_csv(input_file, encoding='utf-16', delimiter=';')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='utf-8', delimiter=';')

    print("--- Raw Data (First 5 rows) ---")
    try:
        print(df.head().to_markdown(index=False))
    except ImportError:
        print(df.head())
        print("Please install the 'tabulate' library to see the dataframe in a better format.")
    print("--- Column Data Types ---")
    print(df.dtypes)
    print("---")

    print("--- All Columns in CSV File ---")
    print(df.columns)
    print("---")

    df.columns = df.columns.str.strip()
    
    # Convert date columns
    df['Started'] = pd.to_datetime(df['Started'], errors='coerce')
    df['Stopped'] = pd.to_datetime(df['Stopped'], errors='coerce')
    
    # Convert duration to timedelta
    df['Duration'] = pd.to_timedelta(df['Duration'], errors='coerce')
    
    # Convert numeric columns (Distance, Fuel, Battery)
    df['Distance (km)'] = pd.to_numeric(df['Distance (km)'], errors='coerce')
    df['Fuel consumption (litres)'] = df['Fuel consumption (litres)'].str.replace(',', '.', regex=False)
    df['Fuel consumption (litres)'] = pd.to_numeric(df['Fuel consumption (litres)'], errors='coerce')
    df['Battery consumption (kWh)'] = df['Battery consumption (kWh)'].str.replace(',', '.', regex=False)
    df['Battery consumption (kWh)'] = pd.to_numeric(df['Battery consumption (kWh)'], errors='coerce')
    
    # Drop unnecessary or unnamed columns
    df = df.drop(columns=['Unnamed: 14', 'Title'], errors='ignore')
    
    # Drop rows with missing necessary columns
    df = df.dropna(subset=['Started', 'Distance (km)'])
    
    return df

@st.cache_data
def geocode_address(address, _geolocator, retries=3, timeout=5):
    if pd.isna(address):
        return None, None
    for attempt in range(retries):
        try:
            location = _geolocator.geocode(address, timeout=timeout)
            if location:
                return location.latitude, location.longitude
            return None, None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt < retries - 1:
                import time
                time.sleep(timeout)
            else:
                return None, None
        except Exception:
            return None, None
    return None, None

def add_lat_lon(df):
    geolocator = Nominatim(user_agent="volvo_driving_app")
    st.info("Geocoding addresses. This might take a while...")
    start_latitudes = []
    start_longitudes = []
    end_latitudes = []
    end_longitudes = []
    for _, row in df.iterrows():
        start_lat, start_lon = geocode_address(row['Start address'], geolocator)
        start_latitudes.append(start_lat)
        start_longitudes.append(start_lon)
        end_lat, end_lon = geocode_address(row['End address'], geolocator)
        end_latitudes.append(end_lat)
        end_longitudes.append(end_lon)

    df['Latitude_Start'] = start_latitudes
    df['Longitude_Start'] = start_longitudes
    df['Latitude_End'] = end_latitudes
    df['Longitude_End'] = end_longitudes
    st.success("Geocoding complete!")
    return df

df = load_data()

# Geocode if missing
if 'Latitude_Start' not in df.columns or 'Longitude_Start' not in df.columns or 'Latitude_End' not in df.columns or 'Longitude_End' not in df.columns:
    df = add_lat_lon(df.copy())

# =====================
# ODOMETER CORRECTION
# =====================
if 'Start odometer (km)' in df.columns and 'End odometer (km)' in df.columns:
    df = df.sort_values(by='Started')
    df['Distance Calculated'] = df['End odometer (km)'] - df['Start odometer (km)']
    distance_difference = (df['Distance Calculated'] - df['Distance (km)']).abs()
    if distance_difference.max() > 1:
        st.warning(f"Significant difference found. Max difference: {distance_difference.max():.2f} km.")
        df['Cumulative Distance (km)'] = df['Distance (km)'].cumsum()
    else:
        df['Cumulative Distance (km)'] = df['Distance Calculated'].cumsum()

    if not df.empty:
        odometer_start_km = df['Start odometer (km)'].iloc[0] - df['Cumulative Distance (km)'].iloc[0]
        df['Total Distance (km)'] = odometer_start_km + df['Cumulative Distance (km)']
    else:
        df['Cumulative Distance (km)'] = 0
        df['Total Distance (km)'] = 0
else:
    st.warning("Odometer data not found. Falling back to 'Distance (km)'.")
    df = df.sort_values(by='Started')
    df['Cumulative Distance (km)'] = df['Distance (km)'].cumsum()
    df['Total Distance (km)'] = df['Cumulative Distance (km)']

location_df = df[['Latitude_Start', 'Longitude_Start', 'Latitude_End', 'Longitude_End']].copy()
location_df = location_df.dropna(how='all')

def haversine(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

df['Route Length (km)'] = df.apply(lambda row: haversine(row['Latitude_Start'], row['Longitude_Start'], row['Latitude_End'], row['Longitude_End']), axis=1)

# =====================
# FUEL CONSUMPTION DASHBOARD
# =====================
st.subheader("⛽ Fuel Consumption Dashboard")
col1, col2 = st.columns(2)
col1.metric("Total Fuel Consumed (litres)", f"{df['Fuel consumption (litres)'].sum():.2f}" if not df.empty else "0.00")
col2.metric("Avg Fuel Consumption (litres)", f"{df['Fuel consumption (litres)'].mean():.2f}" if not df.empty else "0.00")

# =====================
# BATTERY CONSUMPTION DASHBOARD
# =====================
st.subheader("🔋 Battery Consumption Dashboard")
col1, col2 = st.columns(2)
col1.metric("Total Battery Consumed (kWh)", f"{df['Battery consumption (kWh)'].sum():.2f}" if not df.empty else "0.00")
col2.metric("Avg Battery Consumption (kWh)", f"{df['Battery consumption (kWh)'].mean():.2f}" if not df.empty else "0.00")

