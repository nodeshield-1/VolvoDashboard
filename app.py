import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
from geopy.geocoders import ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Set Streamlit page config at the very top
st.set_page_config(page_title="Volvo Driving Journal", layout="wide")

# =====================
# UNIT DETECTION AND CONVERSION
# =====================
def detect_units(df):
    """Detect whether the data is in metric or imperial units."""
    # Check column names to determine unit system
    distance_col = next((col for col in df.columns if 'Distance' in col), None)
    odometer_col = next((col for col in df.columns if 'odometer' in col), None)
    fuel_col = next((col for col in df.columns if 'Fuel' in col), None)
    
    is_metric = True  # Default to metric
    unit_info = {
        'distance': 'km',
        'fuel': 'litres',
        'odometer': 'km'
    }
    
    # Check distance unit
    if distance_col and 'miles' in distance_col.lower():
        is_metric = False
        unit_info['distance'] = 'miles'
    
    # Check odometer unit
    if odometer_col and 'miles' in odometer_col.lower():
        is_metric = False
        unit_info['odometer'] = 'miles'
    
    # Check fuel unit
    if fuel_col and 'gallons' in fuel_col.lower():
        is_metric = False
        unit_info['fuel'] = 'gallons'
    
    return is_metric, unit_info

def convert_units(df, is_metric, unit_info):
    """Create standardized columns with consistent units for internal calculations."""
    # Create a copy of the dataframe to avoid modifying the original
    df_std = df.copy()
    
    # Map for column names
    distance_col = next((col for col in df.columns if 'Distance' in col and 'Calculated' not in col and 'Cumulative' not in col and 'Total' not in col and 'Route' not in col), None)
    start_odo_col = next((col for col in df.columns if 'Start odometer' in col), None)
    end_odo_col = next((col for col in df.columns if 'End odometer' in col), None)
    fuel_col = next((col for col in df.columns if 'Fuel consumption' in col), None)
    
    # Standardize column names for internal calculations
    col_map = {}
    if distance_col:
        col_map[distance_col] = 'Distance_std'
    if start_odo_col:
        col_map[start_odo_col] = 'Start odometer_std'
    if end_odo_col:
        col_map[end_odo_col] = 'End odometer_std'
    if fuel_col:
        col_map[fuel_col] = 'Fuel consumption_std'
    
    # Create standardized columns (all in metric)
    for orig_col, std_col in col_map.items():
        if orig_col in df.columns:
            df_std[std_col] = df[orig_col].copy()
            
            # Convert imperial to metric if needed
            if not is_metric:
                if 'Distance' in orig_col or 'odometer' in orig_col:
                    # Convert miles to km
                    df_std[std_col] = df_std[std_col] * 1.60934
                elif 'Fuel' in orig_col:
                    # Convert gallons to litres
                    df_std[std_col] = df_std[std_col] * 3.78541
    
    # Store original units for display purposes
    df_std['is_metric'] = is_metric
    
    return df_std

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
    
    # Detect units (metric or imperial)
    is_metric, unit_info = detect_units(df)
    
    # Get appropriate column names based on detected units
    distance_col = next((col for col in df.columns if 'Distance' in col and 'Calculated' not in col and 'Cumulative' not in col and 'Total' not in col and 'Route' not in col), 'Distance (km)')
    fuel_col = next((col for col in df.columns if 'Fuel consumption' in col), 'Fuel consumption (litres)')
    battery_col = next((col for col in df.columns if 'Battery consumption' in col), 'Battery consumption (kWh)')
    
    # Convert numeric columns (Distance, Fuel, Battery)
    df[distance_col] = pd.to_numeric(df[distance_col], errors='coerce')
    
    if fuel_col in df.columns:
        # Clean fuel consumption data - handle units in the values like '0 gal' or '0.1 gal'
        if df[fuel_col].dtype == 'object':
            # First remove the unit part (like 'gal' or 'l')
            df[fuel_col] = df[fuel_col].str.replace(' gal', '', regex=False)
            df[fuel_col] = df[fuel_col].str.replace(' l', '', regex=False)
            # Handle decimal separator (both . and , are supported)
            df[fuel_col] = df[fuel_col].str.replace(',', '.', regex=False)
        df[fuel_col] = pd.to_numeric(df[fuel_col], errors='coerce')
    
    if battery_col in df.columns:
        # Handle decimal separator (both . and , are supported)
        if df[battery_col].dtype == 'object':
            df[battery_col] = df[battery_col].str.replace(',', '.', regex=False)
        df[battery_col] = pd.to_numeric(df[battery_col], errors='coerce')
    
    # Drop unnecessary or unnamed columns
    df = df.drop(columns=[col for col in df.columns if 'Unnamed:' in col or col == 'Title'], errors='ignore')
    
    # Drop rows with missing necessary columns
    df = df.dropna(subset=['Started', distance_col])
    
    # Convert units for internal calculations
    df = convert_units(df, is_metric, unit_info)
    
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
    geolocator = ArcGIS()
    st.info("Geocoding addresses with ArcGIS. This might take a while...")
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

# Extract unit information
is_metric = True
if 'is_metric' in df.columns:
    # Get the first value if it's a Series, otherwise use the default
    is_metric = df['is_metric'].iloc[0] if not df.empty else True
distance_unit = 'km' if is_metric else 'miles'
fuel_unit = 'litres' if is_metric else 'gallons'

# =====================
# ODOMETER CORRECTION
# =====================
if 'Start odometer_std' in df.columns and 'End odometer_std' in df.columns:
    df = df.sort_values(by='Started')
    df['Distance Calculated'] = df['End odometer_std'] - df['Start odometer_std']
    distance_difference = (df['Distance Calculated'] - df['Distance_std']).abs()
    if distance_difference.max() > 1:
        st.warning(f"Significant difference found. Max difference: {distance_difference.max():.2f} {distance_unit}.")
        df['Cumulative Distance'] = df['Distance_std'].cumsum()
    else:
        df['Cumulative Distance'] = df['Distance Calculated'].cumsum()

    if not df.empty:
        odometer_start = df['Start odometer_std'].iloc[0] - df['Cumulative Distance'].iloc[0]
        df['Total Distance'] = odometer_start + df['Cumulative Distance']
    else:
        df['Cumulative Distance'] = 0
        df['Total Distance'] = 0
else:
    st.warning(f"Odometer data not found. Falling back to 'Distance'.")
    df = df.sort_values(by='Started')
    df['Cumulative Distance'] = df['Distance_std'].cumsum()
    df['Total Distance'] = df['Cumulative Distance']

location_df = df[['Latitude_Start', 'Longitude_Start', 'Latitude_End', 'Longitude_End']].copy()
location_df = location_df.dropna(how='all')

def haversine(lat1, lon1, lat2, lon2, unit='km'):
    """Calculate the great circle distance between two points on the earth.
    
    Args:
        lat1, lon1: Latitude and Longitude of point 1 (in decimal degrees)
        lat2, lon2: Latitude and Longitude of point 2 (in decimal degrees)
        unit: 'km' for kilometers or 'miles' for miles
        
    Returns:
        Distance in specified unit (km by default)
    """
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    
    # Radius of earth in km
    R = 6371.0
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    # Convert to miles if requested
    if unit.lower() == 'miles':
        distance = distance * 0.621371
        
    return distance

# Calculate route length in appropriate units
df['Route Length'] = df.apply(
    lambda row: haversine(
        row['Latitude_Start'], 
        row['Longitude_Start'], 
        row['Latitude_End'], 
        row['Longitude_End'],
        'miles' if not is_metric else 'km'
    ), 
    axis=1
)

# Get the original unit columns for display
distance_col_orig = next((col for col in df.columns if 'Distance (' in col), None)
if not distance_col_orig:
    distance_col_orig = 'Distance_std'  # Fallback to standardized column

# Find fuel consumption column
all_fuel_cols = [col for col in df.columns if 'Fuel' in col]
fuel_col_orig = next((col for col in df.columns if 'Fuel consumption (' in col), None)
if not fuel_col_orig and all_fuel_cols:
    # Try to find any column with 'Fuel' in it
    fuel_col_orig = all_fuel_cols[0]
elif not fuel_col_orig:
    fuel_col_orig = 'Fuel consumption_std' if 'Fuel consumption_std' in df.columns else 'Fuel consumption (litres)'
    
battery_col = next((col for col in df.columns if 'Battery consumption' in col), 'Battery consumption (kWh)')

# =====================
# FUEL CONSUMPTION DASHBOARD
# =====================
st.subheader(f"⛽ Fuel Consumption Dashboard ({fuel_unit})")
col1, col2 = st.columns(2)

# Handle fuel consumption metrics with error checking
total_fuel = "0.00"
avg_fuel = "0.00"

if not df.empty and fuel_col_orig in df.columns:
    # Filter out NaN values
    fuel_values = df[fuel_col_orig].dropna()
    if not fuel_values.empty:
        total_fuel = f"{fuel_values.sum():.2f}"
        avg_fuel = f"{fuel_values.mean():.2f}"

col1.metric(f"Total Fuel Consumed ({fuel_unit})", total_fuel)
col2.metric(f"Avg Fuel Consumption ({fuel_unit})", avg_fuel)

# =====================
# BATTERY CONSUMPTION DASHBOARD
# =====================
st.subheader("🔋 Battery Consumption Dashboard")
col1, col2 = st.columns(2)

# Handle battery consumption metrics with error checking
total_battery = "0.00"
avg_battery = "0.00"

if not df.empty and battery_col in df.columns:
    # Filter out NaN values
    battery_values = df[battery_col].dropna()
    if not battery_values.empty:
        total_battery = f"{battery_values.sum():.2f}"
        avg_battery = f"{battery_values.mean():.2f}"

col1.metric("Total Battery Consumed (kWh)", total_battery)
col2.metric("Avg Battery Consumption (kWh)", avg_battery)

# =====================
# DISTANCE DASHBOARD
# =====================
st.subheader(f"🚗 Distance Dashboard ({distance_unit})")
col1, col2 = st.columns(2)

# Handle distance metrics with error checking
total_distance = "0.00"
avg_distance = "0.00"

if not df.empty and distance_col_orig in df.columns:
    # Filter out NaN values
    distance_values = df[distance_col_orig].dropna()
    if not distance_values.empty:
        total_distance = f"{distance_values.sum():.2f}"
        avg_distance = f"{distance_values.mean():.2f}"

col1.metric(f"Total Distance Logged ({distance_unit})", total_distance)
col2.metric(f"Avg Distance per Trip ({distance_unit})", avg_distance)