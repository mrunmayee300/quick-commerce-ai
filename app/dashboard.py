import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

# Page Config (Enterprise Mode)
st.set_page_config(
    page_title="Quick Commerce AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔹 Custom Dark Theme
dark_theme_css = """
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3, p, label, span { color: #e0e4eb !important; }
    .stMetric { background-color: #1a1d24; border-radius: 10px; padding: 10px; }
    .plot-container { background-color: #11131a !important; }
    .css-1v3fvcr { background-color: #0e1117 !important; }
</style>
"""
st.markdown(dark_theme_css, unsafe_allow_html=True)

# Title
st.title("⚡ Quick Commerce AI Platform")
st.caption("Enterprise Intelligence | Demand | Delivery | Inventory")

# =================================================================
# 1️⃣ Demand Forecast
# =================================================================
st.header("📈 Real-time Demand Forecasting")

sales_48 = st.text_area("Enter last 48 hours sales (comma-separated)")

if st.button("Predict Demand"):
    try:
        sales_list = list(map(float, sales_48.split(",")))
        if len(sales_list) != 48:
            st.error("Please enter exactly 48 comma-separated values.")
        else:
            response = requests.post(
                f"{API_URL}/forecast",
                json={"sales_last_48": sales_list}
            )

            if response.status_code == 200:
                result = response.json()
                forecast = result["forecast_next_4_hours"]

                df = pd.DataFrame({
                    "Hour": ["H+1", "H+2", "H+3", "H+4"],
                    "Forecast": forecast
                })

                st.success("Prediction generated successfully!")

                fig = px.bar(
                    df,
                    x="Hour",
                    y="Forecast",
                    title="Predicted Demand for Next 4 Hours",
                    text="Forecast"
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("API Error: Could not compute forecast")

    except:
        st.error("Invalid input. Please enter valid numbers separated by commas.")

# =================================================================
# 2️⃣ ETA & SLA
# =================================================================
st.header("🚚 Delivery Time & SLA Compliance")

col1, col2, col3 = st.columns(3)
distance = col1.number_input("Distance to Customer (km)", min_value=0.0, step=0.1)
hour = col2.number_input("Current Hour (0-23)", min_value=0, max_value=23)
weather = col3.selectbox("Weather Condition", ["Clear", "Rain"])

is_peak = 1 if hour in [8, 9, 18, 19, 20] else 0
weather_flag = 1 if weather == "Rain" else 0

if st.button("Predict ETA & SLA"):
    response = requests.post(
        f"{API_URL}/eta-sla",
        params={
            "distance_km": float(distance),
            "hour": int(hour),
            "is_peak": int(is_peak),
            "weather": int(weather_flag)
        }
    )

    if response.status_code == 200:
        result = response.json()
        eta = round(result["eta_minutes"], 2)
        sla = result["sla_violation"]

        st.metric("Estimated ETA (Minutes)", eta)

        if sla == 1:
            st.error("⛔ SLA will be violated (Late Delivery)")
        else:
            st.success("🟢 SLA will be met (On-time Delivery)")
    else:
        st.error("API Failed to respond!")

# =================================================================
# 3️⃣ Smart Inventory Reorder
# =================================================================
st.header("🛒 Automated Inventory Reorder System")

col4, col5 = st.columns(2)
stock_qty = col4.number_input("Current Stock Quantity", min_value=0)
future_fc = col5.text_input("Predicted 4-hour demand (comma-separated)")

if st.button("Get Reorder Recommendation"):
    try:
        fc_list = list(map(float, future_fc.split(",")))

        if len(fc_list) != 4:
            st.error("Please enter exactly 4 demand numbers.")
        else:
            response = requests.post(
                f"{API_URL}/reorder",
                json={"stock": int(stock_qty), "forecast": fc_list}
            )

            if response.status_code == 200:
                qty = response.json()["recommended_reorder_qty"]

                if qty > 0:
                    st.warning(f"⚠️ Reorder needed: Order **{qty} units**")
                else:
                    st.success("✔ Stock level sufficient. No reorder required.")
            else:
                st.error("API failed to respond!")

    except:
        st.error("Invalid demand input! Provide valid numeric values.")


# ======================
# KPI Insights Section
# ======================
st.header("📊 Operational KPI Insights")

if 'eta' in locals():
    stockout_risk = max(0, 1 - (stock_qty / sum(fc_list))) if 'fc_list' in locals() else 0.2
    delay_risk = min(1, eta / 60)
    volatility = np.std(sales_list) if 'sales_list' in locals() else 0.3

    colA, colB, colC = st.columns(3)

    colA.metric("📦 Stockout Risk", f"{stockout_risk:.2f}")
    colB.metric("⏱ Delivery Delay Risk", f"{delay_risk:.2f}")
    colC.metric("📉 Demand Volatility", f"{volatility:.2f}")

    st.caption("KPIs are simulated using available inputs. Live data improves signal accuracy.")
else:
    st.info("Run predictions above to view operational KPIs.")

# =================================================================
# 4️⃣ Delivery Region Intelligence Map
# =================================================================
st.header("📍 Delivery Region Intelligence")

city_coords = {
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Delhi": (28.7041, 77.1025),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882)
}

city = st.selectbox("Select City", list(city_coords.keys()))
warehouse_lat, warehouse_lon = city_coords[city]

if 'eta' in locals():
    # Generate random customer locations based on selected distance
    num_customers = 8
    distances_km = np.random.uniform(0.5, distance, size=num_customers)

    # Lat/lon simulation (tiny shifts)
    customer_lats = warehouse_lat + np.random.uniform(-0.02, 0.02, num_customers)
    customer_lons = warehouse_lon + np.random.uniform(-0.02, 0.02, num_customers)

    sla_colors = ['green' if eta < 25 else 'red'] * num_customers

    map_df = pd.DataFrame({
        "lat": customer_lats,
        "lon": customer_lons,
        "ETA (min)": [round(eta, 2)] * num_customers,
        "SLA Risk": ["OK" if eta < 25 else "Violation"] * num_customers,
        "color": sla_colors
    })

    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat", lon="lon",
        color="SLA Risk",
        size_max=12,
        zoom=10,
        height=400,
        hover_data=["ETA (min)"]
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0, "t":0, "l":0, "b":0},
        showlegend=True
    )

    # Mark warehouse separately
    fig_map.add_scattermapbox(
        lat=[warehouse_lat], lon=[warehouse_lon],
        mode="markers",
        marker=dict(size=15, color="blue"),
        name="Warehouse"
    )

    st.plotly_chart(fig_map, use_container_width=True)

else:
    st.info("Run ETA prediction first to visualize delivery region")
