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

# Initialize session state for unit preference
if 'use_imperial' not in st.session_state:
    st.session_state.use_imperial = False

# =====================
# UNIT CONVERSION FUNCTIONS
# =====================
def miles_to_km(miles):
    return miles * 1.60934

def km_to_miles(km):
    return km / 1.60934

def gallons_to_liters(gallons):
    return gallons * 3.78541

def liters_to_gallons(liters):
    return liters / 3.78541

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
    
    # Determine if we're using imperial or metric units
    is_imperial = 'Distance (miles)' in df.columns
    
    # Convert numeric columns based on unit system
    if is_imperial:
        # Imperial units (miles, gallons)
        df['Distance (miles)'] = pd.to_numeric(df['Distance (miles)'], errors='coerce')
        df['Distance (km)'] = df['Distance (miles)'].apply(miles_to_km)
        
        # Handle fuel consumption - extract numeric value from strings like "0.1 gal"
        if 'Fuel consumption (gallons)' in df.columns:
            df['Fuel consumption (gallons)'] = df['Fuel consumption (gallons)'].str.replace(' gal', '', regex=False)
            df['Fuel consumption (gallons)'] = pd.to_numeric(df['Fuel consumption (gallons)'], errors='coerce')
            df['Fuel consumption (litres)'] = df['Fuel consumption (gallons)'].apply(gallons_to_liters)
        
        # Set odometer column names
        odometer_start_col = 'Start odometer (miles)'
        odometer_end_col = 'End odometer (miles)'
        df['Start odometer (km)'] = df[odometer_start_col].apply(miles_to_km)
        df['End odometer (km)'] = df[odometer_end_col].apply(miles_to_km)
    else:
        # Metric units (km, litres)
        df['Distance (km)'] = pd.to_numeric(df['Distance (km)'], errors='coerce')
        df['Distance (miles)'] = df['Distance (km)'].apply(km_to_miles)
        
        df['Fuel consumption (litres)'] = df['Fuel consumption (litres)'].str.replace(',', '.', regex=False)
        df['Fuel consumption (litres)'] = pd.to_numeric(df['Fuel consumption (litres)'], errors='coerce')
        df['Fuel consumption (gallons)'] = df['Fuel consumption (litres)'].apply(liters_to_gallons)
        
        # Set odometer column names
        odometer_start_col = 'Start odometer (km)'
        odometer_end_col = 'End odometer (km)'
    
    # Handle battery consumption for both systems - same units in both
    df['Battery consumption (kWh)'] = df['Battery consumption (kWh)'].astype(str).str.replace(',', '.', regex=False)
    df['Battery consumption (kWh)'] = pd.to_numeric(df['Battery consumption (kWh)'], errors='coerce')
    
    if 'Battery regeneration (kWh)' in df.columns:
        df['Battery regeneration (kWh)'] = df['Battery regeneration (kWh)'].astype(str).str.replace(',', '.', regex=False)
        df['Battery regeneration (kWh)'] = pd.to_numeric(df['Battery regeneration (kWh)'], errors='coerce')
    
    # Drop unnecessary or unnamed columns
    df = df.drop(columns=['Unnamed: 14', 'Title'], errors='ignore')
    
    # Drop rows with missing necessary columns
    df = df.dropna(subset=['Started', 'Distance (km)'])
    
    # Store the unit system in the dataframe as metadata
    df.attrs['is_imperial'] = is_imperial
    
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
# Get unit system from dataframe metadata
is_imperial = df.attrs.get('is_imperial', False)
distance_unit = 'miles' if is_imperial else 'km'

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
        
    # Calculate imperial equivalents
    df['Cumulative Distance (miles)'] = df['Cumulative Distance (km)'].apply(km_to_miles)
    df['Total Distance (miles)'] = df['Total Distance (km)'].apply(km_to_miles)
else:
    # Handle the case where we don't have odometer data in km but might have it in miles
    if is_imperial and 'Start odometer (miles)' in df.columns and 'End odometer (miles)' in df.columns:
        df = df.sort_values(by='Started')
        df['Distance Calculated (miles)'] = df['End odometer (miles)'] - df['Start odometer (miles)']
        distance_difference_miles = (df['Distance Calculated (miles)'] - df['Distance (miles)']).abs()
        
        if distance_difference_miles.max() > 0.6:  # ~1 km in miles
            st.warning(f"Significant difference found. Max difference: {distance_difference_miles.max():.2f} miles.")
            df['Cumulative Distance (miles)'] = df['Distance (miles)'].cumsum()
        else:
            df['Cumulative Distance (miles)'] = df['Distance Calculated (miles)'].cumsum()

        if not df.empty:
            odometer_start_miles = df['Start odometer (miles)'].iloc[0] - df['Cumulative Distance (miles)'].iloc[0]
            df['Total Distance (miles)'] = odometer_start_miles + df['Cumulative Distance (miles)']
        else:
            df['Cumulative Distance (miles)'] = 0
            df['Total Distance (miles)'] = 0
            
        # Calculate metric equivalents
        df['Cumulative Distance (km)'] = df['Cumulative Distance (miles)'].apply(miles_to_km)
        df['Total Distance (km)'] = df['Total Distance (miles)'].apply(miles_to_km)
    else:
        # No odometer data found in either unit system
        st.warning(f"Odometer data not found. Falling back to 'Distance ({distance_unit})'.")
        df = df.sort_values(by='Started')
        
        # Calculate cumulative distances in both units
        df['Cumulative Distance (km)'] = df['Distance (km)'].cumsum()
        df['Cumulative Distance (miles)'] = df['Distance (miles)'].cumsum()
        
        # Set total distances equal to cumulative distances
        df['Total Distance (km)'] = df['Cumulative Distance (km)']
        df['Total Distance (miles)'] = df['Cumulative Distance (miles)']

location_df = df[['Latitude_Start', 'Longitude_Start', 'Latitude_End', 'Longitude_End']].copy()
location_df = location_df.dropna(how='all')

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Skip calculation for invalid coordinates or when start and end are the same
    if (pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2) or 
        (lat1 == lat2 and lon1 == lon2)):
        return 0.0
        
    # Convert decimal degrees to radians
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return distance

# Instead of calculating route lengths, we'll use the actual distances from the data
# This is because geocoding might not be accurate enough for precise distance calculations
# We'll only calculate the route lengths for display on maps

# Store the calculated direct-line distances for potential map visualizations
# but we won't use them for metrics display
df['Direct Distance (km)'] = df.apply(
    lambda row: haversine(row['Latitude_Start'], row['Longitude_Start'], 
                         row['Latitude_End'], row['Longitude_End']), 
    axis=1
)
df['Direct Distance (miles)'] = df['Direct Distance (km)'].apply(km_to_miles)

# =====================
# UNIT SELECTION
# =====================
st.sidebar.title("Settings")
unit_toggle = st.sidebar.checkbox("Use Imperial Units (miles, gallons)", value=st.session_state.use_imperial)
st.session_state.use_imperial = unit_toggle

# Determine which unit system to display based on user preference
if st.session_state.use_imperial:
    distance_unit = "miles"
    fuel_unit = "gallons"
else:
    distance_unit = "km"
    fuel_unit = "litres"

# =====================
# FUEL CONSUMPTION DASHBOARD
# =====================
st.subheader(f"⛽ Fuel Consumption Dashboard ({fuel_unit})")
col1, col2 = st.columns(2)

if st.session_state.use_imperial:
    total_fuel = df['Fuel consumption (gallons)'].sum()
    avg_fuel = df['Fuel consumption (gallons)'].mean()
    col1.metric(f"Total Fuel Consumed ({fuel_unit})", f"{total_fuel:.2f}" if not df.empty else "0.00")
    col2.metric(f"Avg Fuel Consumption ({fuel_unit})", f"{avg_fuel:.2f}" if not df.empty else "0.00")
else:
    total_fuel = df['Fuel consumption (litres)'].sum()
    avg_fuel = df['Fuel consumption (litres)'].mean()
    col1.metric(f"Total Fuel Consumed ({fuel_unit})", f"{total_fuel:.2f}" if not df.empty else "0.00")
    col2.metric(f"Avg Fuel Consumption ({fuel_unit})", f"{avg_fuel:.2f}" if not df.empty else "0.00")

# =====================
# BATTERY CONSUMPTION DASHBOARD
# =====================
st.subheader("🔋 Battery Consumption Dashboard")
col1, col2 = st.columns(2)
col1.metric("Total Battery Consumed (kWh)", f"{df['Battery consumption (kWh)'].sum():.2f}" if not df.empty else "0.00")
col2.metric("Avg Battery Consumption (kWh)", f"{df['Battery consumption (kWh)'].mean():.2f}" if not df.empty else "0.00")

# =====================
# DISTANCE METRICS
# =====================
st.subheader(f"🛣️ Distance Metrics ({distance_unit})")
col1, col2, col3 = st.columns(3)

if st.session_state.use_imperial:
    total_dist = df['Total Distance (miles)'].max() if not df.empty else 0
    total_traveled = df['Distance (miles)'].sum() if not df.empty else 0
    
    # Use the actual distance values from the data rather than calculated route lengths
    # This assumes the distance in the CSV is more accurate than the calculated haversine distance
    avg_trip = df['Distance (miles)'].mean() if not df.empty else 0
    
    col1.metric(f"Total Distance ({distance_unit})", f"{total_dist:.1f}")
    col2.metric(f"Total Distance Traveled ({distance_unit})", f"{total_traveled:.1f}")
    col3.metric(f"Average Trip Distance ({distance_unit})", f"{avg_trip:.1f}")
else:
    total_dist = df['Total Distance (km)'].max() if not df.empty else 0
    total_traveled = df['Distance (km)'].sum() if not df.empty else 0
    
    # Use the actual distance values from the data rather than calculated route lengths
    avg_trip = df['Distance (km)'].mean() if not df.empty else 0
    
    col1.metric(f"Total Distance ({distance_unit})", f"{total_dist:.1f}")
    col2.metric(f"Total Distance Traveled ({distance_unit})", f"{total_traveled:.1f}")
    col3.metric(f"Average Trip Distance ({distance_unit})", f"{avg_trip:.1f}")

