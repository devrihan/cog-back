
import numpy as np
import pandas as pd

# ----------- LOAD BALANCER ANALYZER ----------- #
def analyze_orders(orders, stations: int):
    """
    Analyze packing orders before running optimizer.
    Returns summary stats for visualization, including imbalance %.
    """
    
    # --- FIX: Handle empty or invalid input gracefully ---
    if not orders or stations <= 0:
        return {
            "totalOrders": 0,
            "totalTime": 0,
            "avgLoadPerStation": 0,
            "maxOrderTime": 0,
            "stationLoadSummary": [],
            "imbalancePercent": 0,
            "insight": "No order data provided or invalid number of stations."
        }

    df = pd.DataFrame(orders)

    # Ensure packingTime column exists and is numeric
    if "packingTime" not in df.columns:
        return {
            "totalOrders": 0, "totalTime": 0, "avgLoadPerStation": 0, "maxOrderTime": 0,
            "stationLoadSummary": [], "imbalancePercent": 0,
            "insight": "CSV must include a 'packingTime' column."
        }

    df["packingTime"] = pd.to_numeric(df["packingTime"], errors='coerce').fillna(0)

    total_time = df["packingTime"].sum()
    avg_load = total_time / stations
    max_order = df["packingTime"].max()

    # Random distribution baseline (before optimization)
    random_loads = np.random.choice(range(1, stations + 1), size=len(df))
    df["assignedStation"] = random_loads

    # station_summary = (
    #     df.groupby("assignedStation")["packingTime"].sum().reset_index()
    #     .rename(columns={"packingTime": "totalTime", "assignedStation": "station"})
    #     .to_dict(orient="records")
    # )
    station_summary = (
        df.groupby("assignedStation")
        .agg(totalTime=("packingTime", "sum"), orders=("id", lambda x: list(x)))
        .reset_index()
        .rename(columns={"assignedStation": "station"})
        .to_dict(orient="records")
    )
    assignments = df[["id", "packingTime", "assignedStation"]].to_dict(orient="records")

    # Compute imbalance %
    station_totals = [s["totalTime"] for s in station_summary]
    if station_totals:
        imbalance = round(
            (max(station_totals) - min(station_totals)) / avg_load * 100, 2
        )
    else:
        imbalance = 0.0

    

    return {
        "totalOrders": len(orders),
        "totalTime": float(total_time),
        "avgLoadPerStation": float(avg_load),
        "maxOrderTime": float(max_order),
        "stationLoadSummary": station_summary,
        "assignments": assignments,
        "imbalancePercent": imbalance,
        "insight": f"Expected bottleneck: station(s) above {avg_load:.2f} mins avg load."
    }



# ----------- INVENTORY ANALYZER ----------- #
# def analyze_inventory(skuData):
#     """
#     Analyze inventory before optimization.
#     Returns demand, stock stats, forecast accuracy.
#     """
#     if not skuData:
#         return {
#             "totalForecastDemand": 0, "totalActualDemand": 0, "totalStock": 0,
#             "forecastAccuracy": 0, "likelyShortages": [], "likelyOverstocks": [],
#             "insight": "No SKU data provided."
#         }

#     df = pd.DataFrame([sku.dict() for sku in skuData])

#     total_demand = df["forecastDemand"].sum()
#     total_actual = df["actualDemand"].sum()
#     total_stock = df["stock"].sum()

#     # Forecast accuracy = 1 - (|forecast - actual| / forecast)
#     df["forecastError"] = abs(df["forecastDemand"] - df["actualDemand"])
    
#     # Avoid division by zero if total_demand is 0
#     if total_demand > 0:
#         forecast_accuracy = round(100 * (1 - df["forecastError"].sum() / total_demand), 2)
#     else:
#         forecast_accuracy = 100.0

#     shortages = df[df["stock"] < df["actualDemand"]].to_dict(orient="records")
#     overstocks = df[df["stock"] > df["actualDemand"]].to_dict(orient="records")

#     return {
#         "totalForecastDemand": int(total_demand),
#         "totalActualDemand": int(total_actual),
#         "totalStock": int(total_stock),
#         "forecastAccuracy": forecast_accuracy,
#         "likelyShortages": shortages,
#         "likelyOverstocks": overstocks,
#         "insight": f"Forecast accuracy is {forecast_accuracy}%. Risk of shortages in {len(shortages)} SKUs."
#     }

import numpy as np
import pandas as pd

# ... (keep the existing analyze_orders function)


# ----------- INVENTORY ANALYZER ----------- #
def analyze_inventory(skuData):
    """
    Analyze inventory before optimization.
    Returns demand, stock stats, forecast accuracy, and supplier insights.
    """
    if not skuData:
        return {
            "totalForecastDemand": 0, "totalActualDemand": 0, "totalStock": 0,
            "forecastAccuracy": 0, "likelyShortages": [], "likelyOverstocks": [],
            "supplierReliability": [],
            "insight": "No SKU data provided."
        }

    df = pd.DataFrame([sku.dict() for sku in skuData])

    total_demand = df["forecastDemand"].sum()
    total_actual = df["actualDemand"].sum()
    total_stock = df["stock"].sum()

    df["forecastError"] = abs(df["forecastDemand"] - df["actualDemand"])
    
    if total_demand > 0:
        forecast_accuracy = round(100 * (1 - df["forecastError"].sum() / total_demand), 2)
    else:
        forecast_accuracy = 100.0

    shortages = df[df["stock"] < df["actualDemand"]].to_dict(orient="records")
    overstocks = df[df["stock"] > df["actualDemand"]].to_dict(orient="records")
    
    # New analysis for supplier reliability
    supplier_reliability = df.groupby('supplierReliability').agg(
        num_skus=('sku', 'count'),
        avg_stock=('stock', 'mean')
    ).reset_index().to_dict(orient='records')


    return {
        "totalForecastDemand": int(total_demand),
        "totalActualDemand": int(total_actual),
        "totalStock": int(total_stock),
        "forecastAccuracy": forecast_accuracy,
        "likelyShortages": shortages,
        "likelyOverstocks": overstocks,
        "supplierReliability": supplier_reliability,
        "insight": f"Forecast accuracy is {forecast_accuracy}%. Supplier reliability analysis is available."
    }