from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import joblib
from app.inventory_rules import reorder_strategy

app = FastAPI(title="Quick Commerce AI API")

# Input model for forecast API
class ForecastInput(BaseModel):
    sales_last_48: list

# Input model for reorder API
class ReorderInput(BaseModel):
    stock: int
    forecast: list

# Load LSTM Forecast Model
class LSTMForecast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, 64, batch_first=True)
        self.fc = torch.nn.Linear(64, 4)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

lstm_model = LSTMForecast()
lstm_model.load_state_dict(torch.load("models/demand_forecast.pt"))
lstm_model.eval()

# Load ETA + SLA models
eta_model = joblib.load("models/eta_model.pkl")
sla_model = joblib.load("models/sla_model.pkl")


@app.get("/")
def root():
    return {"message": "Quick Commerce AI Backend Running!"}


@app.post("/forecast")
def forecast(input_data: ForecastInput):
    x = torch.tensor([input_data.sales_last_48]).float()
    pred = lstm_model(x).detach().numpy().flatten().tolist()
    return {"forecast_next_4_hours": pred}


@app.post("/eta-sla")
def eta_sla(distance_km: float, hour: int, is_peak: int, weather: int):
    features = np.array([[distance_km, hour, is_peak, weather]])
    eta = eta_model.predict(features)[0]
    sla = sla_model.predict(features)[0]
    return {"eta_minutes": float(eta), "sla_violation": int(sla)}


@app.post("/reorder")
def reorder(input_data: ReorderInput):
    qty = reorder_strategy(input_data.stock, input_data.forecast)
    return {"recommended_reorder_qty": int(qty)}
