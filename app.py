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
from streamlit_folium import st_folium

# Set Streamlit page config at the very top
st.set_page_config(page_title="Volvo Driving Journal", layout="wide")

# Header and introduction
st.title("Volvo Driving Journal Dashboard")
st.markdown("""
This dashboard visualizes your driving data from the Volvo Driving Journal export. 
It shows metrics for fuel consumption, battery usage, and distances traveled.

**To see map visualizations:** Click the 'Load Map Data' button in the sidebar.
""")

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
    # Allow user to select the input file
    input_file = st.sidebar.selectbox(
        "Select data file",
        ["volvo_driving_journal.csv", "test_sample.csv"],
        index=0
    )
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
    
    # Add a progress bar
    progress_bar = st.progress(0)
    total_rows = len(df)
    
    # Ask for location context to improve geocoding accuracy
    location_context = st.text_input(
        "Enter your city and state/country for better geocoding accuracy (e.g., 'Boston, MA, USA')",
        value="New York, NY, USA"
    )
    
    start_latitudes = []
    start_longitudes = []
    end_latitudes = []
    end_longitudes = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        # Add location context to addresses for better geocoding
        start_address = f"{row['Start address']}, {location_context}" if pd.notna(row['Start address']) else None
        end_address = f"{row['End address']}, {location_context}" if pd.notna(row['End address']) else None
        
        # Geocode with improved addresses
        start_lat, start_lon = geocode_address(start_address, geolocator)
        start_latitudes.append(start_lat)
        start_longitudes.append(start_lon)
        
        end_lat, end_lon = geocode_address(end_address, geolocator)
        end_latitudes.append(end_lat)
        end_longitudes.append(end_lon)
        
        # Update progress
        progress_bar.progress((i + 1) / total_rows)

    df['Latitude_Start'] = start_latitudes
    df['Longitude_Start'] = start_longitudes
    df['Latitude_End'] = end_latitudes
    df['Longitude_End'] = end_longitudes
    
    # Add the location context used to the dataframe for reference
    df.attrs['location_context'] = location_context
    
    st.success("Geocoding complete!")
    return df

df = load_data()

# Initialize geocoding state
if 'geocoding_done' not in st.session_state:
    st.session_state.geocoding_done = False

# Make geocoding optional via a button
if ('Latitude_Start' not in df.columns or 'Longitude_Start' not in df.columns or 
    'Latitude_End' not in df.columns or 'Longitude_End' not in df.columns):
    if not st.session_state.geocoding_done:
        geocode_button = st.sidebar.button("Load Map Data (Geocode Addresses)")
        if geocode_button:
            # Update the dataframe with geocoded data and store in session state
            geocoded_df = add_lat_lon(df.copy())
            
            # Store state to session so it persists on rerun
            st.session_state.geocoding_done = True
            
            # Also store geocoded data to session to preserve it
            st.session_state.geocoded_df = geocoded_df
            
            # Force a rerun to update the UI
            st.rerun()
    else:
        # Try to recover geocoded data from session state
        if 'geocoded_df' in st.session_state:
            df = st.session_state.geocoded_df
        st.sidebar.success("Map data loaded successfully!")
else:
    st.session_state.geocoding_done = True

# =====================
# ODOMETER CORRECTION
# =====================
# Get unit system from dataframe metadata
is_imperial = df.attrs.get('is_imperial', False)
distance_unit = 'miles' if st.session_state.use_imperial else 'km'

if 'Start odometer (km)' in df.columns and 'End odometer (km)' in df.columns:
    df = df.sort_values(by='Started')
    df['Distance Calculated'] = df['End odometer (km)'] - df['Start odometer (km)']
    distance_difference = (df['Distance Calculated'] - df['Distance (km)']).abs()
    
    if distance_difference.max() > 1:
        # Display the warning with the proper unit based on user preference
        if st.session_state.use_imperial:
            # Convert the max difference to miles for display
            max_diff_miles = distance_difference.max() * 0.621371
            st.warning(f"Significant difference found between odometer readings and recorded distances. Max difference: {max_diff_miles:.2f} miles.")
        else:
            st.warning(f"Significant difference found between odometer readings and recorded distances. Max difference: {distance_difference.max():.2f} km.")
            
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

# Only create location dataframe if geocoding has been done
if ('Latitude_Start' in df.columns and 'Longitude_Start' in df.columns and 
    'Latitude_End' in df.columns and 'Longitude_End' in df.columns):
    location_df = df[['Latitude_Start', 'Longitude_Start', 'Latitude_End', 'Longitude_End']].copy()
    location_df = location_df.dropna(how='all')
else:
    # Create an empty location dataframe with the right columns
    location_df = pd.DataFrame(columns=['Latitude_Start', 'Longitude_Start', 'Latitude_End', 'Longitude_End'])

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

# Only calculate direct-line distances if geocoding has been done
if st.session_state.geocoding_done:
    # Store the calculated direct-line distances for potential map visualizations
    # but we won't use them for metrics display
    df['Direct Distance (km)'] = df.apply(
        lambda row: haversine(row['Latitude_Start'], row['Longitude_Start'], 
                             row['Latitude_End'], row['Longitude_End']), 
        axis=1
    )
    df['Direct Distance (miles)'] = df['Direct Distance (km)'].apply(km_to_miles)
else:
    # Add placeholder columns to avoid errors elsewhere in the code
    df['Direct Distance (km)'] = 0
    df['Direct Distance (miles)'] = 0

# =====================
# UNIT SELECTION
# =====================
st.sidebar.title("Settings")

# Only update session state if the value actually changes
prev_imperial_setting = st.session_state.use_imperial
unit_toggle = st.sidebar.checkbox("Use Imperial Units (miles, gallons)", value=prev_imperial_setting)

# Check if toggle state has changed, and if so, force a rerun
if unit_toggle != prev_imperial_setting:
    st.session_state.use_imperial = unit_toggle
    st.rerun()  # Force rerun to update all calculations and displays

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

# Add debugging section with a toggle
debug_toggle = st.sidebar.checkbox("Show Debug Info", value=False)
if debug_toggle:
    st.subheader("Debug Information")
    st.write(f"DataFrame shape: {df.shape}")
    st.write(f"DataFrame columns: {df.columns.tolist()}")
    st.write(f"Missing values in key columns: {df[['Started', 'Distance (km)']].isna().sum().to_dict()}")
    st.write(f"Geocoding state: {st.session_state.geocoding_done}")
    st.write(f"Unit system (imperial): {st.session_state.use_imperial}")
    
    # Check if geocoding columns exist and display stats
    geo_cols = [col for col in df.columns if col.startswith('Latitude') or col.startswith('Longitude')]
    if geo_cols:
        st.write(f"Geocoding columns: {geo_cols}")
        st.write(f"Non-null geocoding values: {df[geo_cols].count().to_dict()}")
        if 'location_context' in df.attrs:
            st.write(f"Location context used for geocoding: {df.attrs['location_context']}")
    
    st.write("Data Sample (first 3 rows):")
    st.dataframe(df.head(3))
    
    # If there are geocoded locations, show the first few to verify
    if geo_cols and not df[geo_cols].dropna().empty:
        st.write("First few geocoded addresses:")
        for i, row in df.head(3).iterrows():
            if pd.notna(row.get('Latitude_Start')) and pd.notna(row.get('Longitude_Start')):
                st.write(f"Start: {row['Start address']} → ({row['Latitude_Start']:.5f}, {row['Longitude_Start']:.5f})")
            if pd.notna(row.get('Latitude_End')) and pd.notna(row.get('Longitude_End')):
                st.write(f"End: {row['End address']} → ({row['Latitude_End']:.5f}, {row['Longitude_End']:.5f})")

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

# =====================
# MAP VISUALIZATION
# =====================
if st.session_state.geocoding_done:
    # Make sure geocoding columns actually exist
    if ('Latitude_Start' in df.columns and 'Longitude_Start' in df.columns and 
        'Latitude_End' in df.columns and 'Longitude_End' in df.columns):
        
        st.subheader("🗺️ Trip Map Visualization")
        
        # Create a map centered at the mean of start and end coordinates
        valid_starts = df[['Latitude_Start', 'Longitude_Start']].dropna()
        valid_ends = df[['Latitude_End', 'Longitude_End']].dropna()
        
        if not valid_starts.empty and not valid_ends.empty:
            # Calculate center point for the map
            center_lat = (valid_starts['Latitude_Start'].mean() + valid_ends['Latitude_End'].mean()) / 2
            center_lon = (valid_starts['Longitude_Start'].mean() + valid_ends['Longitude_End'].mean()) / 2
            
            # Create a map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            
            # Add markers for all start and end points
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add start points (green)
            for idx, row in valid_starts.iterrows():
                folium.Marker(
                    location=[row['Latitude_Start'], row['Longitude_Start']],
                    popup=f"Start: {df.at[idx, 'Start address']}",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(marker_cluster)
                
            # Add end points (red)
            for idx, row in valid_ends.iterrows():
                folium.Marker(
                    location=[row['Latitude_End'], row['Longitude_End']],
                    popup=f"End: {df.at[idx, 'End address']}",
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(marker_cluster)
                
            # Add lines connecting start to end points
            for idx, row in df.dropna(subset=['Latitude_Start', 'Longitude_Start', 'Latitude_End', 'Longitude_End']).iterrows():
                folium.PolyLine(
                    locations=[
                        [row['Latitude_Start'], row['Longitude_Start']],
                        [row['Latitude_End'], row['Longitude_End']]
                    ],
                    color='blue',
                    weight=2,
                    opacity=0.7,
                    popup=f"Trip: {row['Distance (km)']:.1f} km / {row['Distance (miles)']:.1f} miles"
                ).add_to(m)
                
            # Display the map
            st_data = st_folium(m, width=1200, height=600)
            
            # Also add a heatmap in a separate tab
            st.subheader("🔥 Trip Density Heatmap")
            
            # Create heatmap data
            heat_data = []
            for idx, row in df.dropna(subset=['Latitude_Start', 'Longitude_Start']).iterrows():
                heat_data.append([row['Latitude_Start'], row['Longitude_Start'], 1])
            for idx, row in df.dropna(subset=['Latitude_End', 'Longitude_End']).iterrows():
                heat_data.append([row['Latitude_End'], row['Longitude_End'], 1])
                
            # Create another map for the heatmap
            m_heat = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            
            # Add heatmap layer
            HeatMap(heat_data).add_to(m_heat)
            
            # Display the heatmap
            st_folium(m_heat, width=1200, height=600)
        else:
            st.warning("Geocoding completed but no valid coordinates were found. The map cannot be displayed.")