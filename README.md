# 🚗 Volvo Driving Journal Dashboard

Welcome!  
This project is an interactive dashboard built with [Streamlit](https://streamlit.io/) to visualize and explore trip data exported from a **Volvo Driving Journal**. It's designed to give you deep insights into your vehicle's usage, from **fuel and battery consumption** to **trip distances** and even **mapped routes** based on real-world geolocation.

---

## ✨ Features

- **📊 Fuel & Battery Dashboards**  
  View total and average consumption at a glance.

- **🗺️ Geolocation & Mapping**  
  Automatically geocode start and end addresses using OpenStreetMap's Nominatim API, and visualize routes on a map.

- **🧮 Odometer Validation**  
  Cross-check trip distances against odometer readings and compute cumulative totals.

- **📂 Intelligent Data Cleaning**  
  Handles inconsistent data, missing values, and localization issues (e.g., decimal commas).

- **📈 Visual Insights**  
  Potential for rich visualizations using Seaborn and Matplotlib.

---

**Prepare your data**
Make sure you have your volvo_driving_journal.csv file in the same directory.
Ensure it uses ; as delimiter and is encoded in UTF-16 or UTF-8.


**Install dependencies**
pip install -r requirements.txt


**Run the dashboard**
streamlit run app.py


**Tech Stack**
 - Python 3
 - Streamlit – UI layer
 - Pandas – data wrangling
 - Folium – interactive maps
 - Geopy – geocoding
 - Matplotlib & Seaborn – visualizations


💡 Future Ideas
Add trip filters (e.g., by date, location, or duration)

Export cleaned & enriched data

Heatmap of frequent routes or stops

Charts for fuel efficiency over time

🔐 Notes on Privacy
The app processes location and odometer data locally and does not store or share any data externally. If you wish to anonymize data before using the dashboard, it's easy to do with a simple script or spreadsheet.


🤝 Contributing
Pull requests are welcome! If you have ideas, improvements, or would like to collaborate, feel free to open an issue or PR.





