# Quick Commerce AI – Real-Time Demand Forecasting, ETA Prediction and Smart Inventory System

Quick Commerce AI is a machine learning based system designed for hyperlocal delivery platforms. It predicts future product demand, estimates delivery time, and recommends optimal inventory reorder quantities. The system simulates real quick-commerce operational workflows such as demand spikes, weather delays, and SLA risks.

---

## Features

- Real-time demand forecasting using past sales history
- Delivery ETA prediction based on distance, hour of the day, and weather
- SLA violation risk classification for operational efficiency
- Inventory reorder recommendation using safety stock logic
- Interactive web dashboard for decision making

---

## Tech Stack

### Machine Learning
- PyTorch (LSTM forecasting)
- XGBoost (ETA and SLA models)
- Scikit-learn (feature engineering)

### Backend
- FastAPI
- Uvicorn
- REST APIs for model inference

### Frontend
- Streamlit Dashboard
- Plotly visualizations

### Deployment / Tools
- Render for FastAPI backend hosting
- Streamlit Cloud for frontend hosting
- GitHub for version control

---

## System Workflow

1. Sales, delivery and stock inputs are provided from the UI.
2. Dashboard sends request to FastAPI backend.
3. Backend loads trained ML models and processes input.
4. API returns demand forecast, delivery time estimate and reorder suggestions.
5. Dashboard displays the results for operational decisions.
