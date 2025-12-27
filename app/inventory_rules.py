import numpy as np

lead_time_hours = 2
holding_cost_per_unit = 0.5
stockout_cost_per_unit = 4.0

def reorder_strategy(stock, forecast):
    demand_lead = np.sum(forecast[:lead_time_hours])
    reorder_qty = max(demand_lead - stock + 20, 0)
    return reorder_qty
