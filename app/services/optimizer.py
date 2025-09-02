# import numpy as np

# # ----------- LOAD BALANCER OPTIMIZER ----------- #
# import numpy as np

# def optimize_orders(orders, stations: int):
#     """
#     LPT (Longest Processing Time first) load balancing algorithm.
#     Assign orders to stations by sorting jobs descending and
#     always giving next largest job to least loaded station.
#     """
#     # Sort orders by packingTime (largest first)
#     sorted_orders = sorted(orders, key=lambda x: x["packingTime"], reverse=True)

#     station_loads = [0] * stations
#     assignments = []

#     for order in sorted_orders:
#         # Find the station with minimum load
#         min_station = int(np.argmin(station_loads))
#         station_loads[min_station] += order["packingTime"]
#         assignments.append({
#             "orderId": order["id"],
#             "station": min_station + 1
#         })

#     station_summary = [
#         {"station": i+1, "totalTime": station_loads[i]} 
#         for i in range(stations)
#     ]

#     imbalance = round(
#         (max(station_loads) - min(station_loads)) / (sum(station_loads)/stations) * 100,
#         2
#     )

#     return {
#         "assignments": assignments,
#         "stationLoadSummary": station_summary,
#         "imbalancePercent": imbalance,
#         "insight": f"LPT scheduling reduced load imbalance to {imbalance}%."
#     }


# import pulp
# import numpy as np


# # ----------- INVENTORY OPTIMIZER ----------- #
# def optimize_inventory(skuData, capacity: int, surgeFactor: float = 1.0, disruption: bool = False):
#     """
#     Optimizes inventory allocation using a linear programming model,
#     considering supplier reliability, MOQ, surges, and disruptions.
#     """
#     # 1. Create the model
#     model = pulp.LpProblem("Advanced_Inventory_Optimization", pulp.LpMinimize)

#     # 2. Define Decision Variables
#     alloc_vars = {
#         sku.sku: pulp.LpVariable(f"alloc_{sku.sku}", lowBound=0, cat='Integer')
#         for sku in skuData
#     }

#     # Adjust for disruptions by reducing effective stock based on reliability
#     if disruption:
#         effective_stock = {sku.sku: sku.stock * sku.supplierReliability for sku in skuData}
#     else:
#         effective_stock = {sku.sku: sku.stock for sku in skuData}


#     # 3. Define the Objective Function
#     # Adjust demand for festival surges
#     effective_demand = {sku.sku: min(sku.forecastDemand, sku.actualDemand) * surgeFactor for sku in skuData}
#     model += pulp.lpSum(
#         [effective_demand[sku.sku] - alloc_vars[sku.sku] for sku in skuData]
#     ), "Total_Shortage"

#     # 4. Define Constraints
#     model += pulp.lpSum([alloc_vars[sku.sku] for sku in skuData]) <= capacity

#     for sku in skuData:
#         # Respect the effective stock level
#         model += alloc_vars[sku.sku] <= effective_stock[sku.sku]
#         # Can't allocate more than the demand
#         model += alloc_vars[sku.sku] <= effective_demand[sku.sku]
#         # Add MOQ constraint
#         is_ordered = pulp.LpVariable(f"is_ordered_{sku.sku}", cat='Binary')
#         model += alloc_vars[sku.sku] >= sku.moq * is_ordered
#         # If we order, we must allocate at least MOQ, but not more than stock
#         model += alloc_vars[sku.sku] <= effective_stock[sku.sku] * is_ordered


#     # 5. Solve the model
#     model.solve()

#     if model.status != pulp.LpStatusOptimal:
#         return {
#             "allocationPlan": [],
#             "shortages": [],
#             "excess": [],
#             "status": pulp.LpStatus[model.status]
#         }

#     # 6. Extract results
#     allocationPlan = []
#     shortages = []
#     excess = []

#     for sku in skuData:
#         allocated_val = int(alloc_vars[sku.sku].varValue)
#         allocationPlan.append({"sku": sku.sku, "allocated": allocated_val})

#         demand = effective_demand[sku.sku]
#         if allocated_val < demand:
#             shortages.append({"sku": sku.sku, "shortage": demand - allocated_val})

#         if allocated_val < effective_stock[sku.sku]:
#             excess.append({"sku": sku.sku, "excess": effective_stock[sku.sku] - allocated_val})

#     return {
#         "allocationPlan": allocationPlan,
#         "shortages": shortages,
#         "excess": excess,
#         "status": pulp.LpStatus[model.status]
#     }

import numpy as np
import pulp
import pandas as pd

# ----------- LOAD BALANCER OPTIMIZER (FROM USER) ----------- #
def optimize_orders(orders, stations: int):
    """
    LPT (Longest Processing Time first) load balancing algorithm.
    Assign orders to stations by sorting jobs descending and
    always giving next largest job to least loaded station.
    """
    # Sort orders by packingTime (largest first)
    sorted_orders = sorted(orders, key=lambda x: x["packingTime"], reverse=True)

    station_loads = [0] * stations
    assignments = []

    for order in sorted_orders:
        # Find the station with minimum load
        min_station = int(np.argmin(station_loads))
        station_loads[min_station] += order["packingTime"]
        assignments.append({
            "orderId": order["id"],
            "station": min_station + 1
        })

    station_summary = [
        {"station": i+1, "totalTime": station_loads[i]} 
        for i in range(stations)
    ]

    imbalance = round(
        (max(station_loads) - min(station_loads)) / (sum(station_loads)/stations) * 100,
        2
    )

    return {
        "assignments": assignments,
        "stationLoadSummary": station_summary,
        "imbalancePercent": imbalance,
        "insight": f"LPT scheduling reduced load imbalance to {imbalance}%."
    }

# ----------- NEW: REPLENISHMENT METRICS CALCULATOR ----------- #
def calculate_replenishment_metrics(df: pd.DataFrame):
    """
    Calculates traditional inventory metrics like EOQ, Safety Stock, and Reorder Point.
    This provides a policy-based view alongside the allocation optimization.
    Assumes demand is annual and lead time is in days.
    """
    # Use forecastDemand as the basis for demand calculations
    demand = df['forecastDemand']

    # --- EOQ Calculation ---
    # Avoid division by zero if holding_cost is 0 or not present
    holding_cost = df.get('holding_cost', 1e-6).replace(0, 1e-6)
    ordering_cost = df.get('ordering_cost', 0)
    df['eoq'] = np.sqrt((2 * demand * ordering_cost) / holding_cost).round()

    # --- Safety Stock and Reorder Point ---
    Z_SCORE = 1.65 # 95% service level
    lead_time = df.get('lead_time', 0)
    daily_demand = demand / 365
    std_dev_daily_demand = daily_demand * 0.20 # Assume 20% demand variability if not provided

    df['safety_stock'] = (Z_SCORE * std_dev_daily_demand * np.sqrt(lead_time)).round()
    df['reorder_point'] = ((daily_demand * lead_time) + df['safety_stock']).round()

    # --- Total Cost Calculation (based on EOQ policy) ---
    annual_holding_cost = (df['eoq'] / 2) * holding_cost
    annual_ordering_cost = (demand / df['eoq'].replace(0, 1e-6)) * ordering_cost
    df['total_cost'] = (annual_holding_cost + annual_ordering_cost).round(2)

    # Clean up results
    df.fillna(0, inplace=True)
    df['safety_stock'] = df['safety_stock'].apply(lambda x: max(0, x))
    df['reorder_point'] = df['reorder_point'].apply(lambda x: max(0, x))
    
    return df[['sku', 'eoq', 'safety_stock', 'reorder_point', 'total_cost']].to_dict(orient='records')


# ----------- INVENTORY OPTIMIZER (FROM USER) ----------- #
def optimize_inventory(skuData, capacity: int, surgeFactor: float = 1.0, disruption: bool = False):
    """
    Optimizes inventory allocation using a linear programming model,
    considering supplier reliability, MOQ, surges, and disruptions.
    """
    # 1. Create the model
    model = pulp.LpProblem("Advanced_Inventory_Optimization", pulp.LpMinimize)

    # 2. Define Decision Variables
    alloc_vars = {
        sku.sku: pulp.LpVariable(f"alloc_{sku.sku}", lowBound=0, cat='Integer')
        for sku in skuData
    }

    # Adjust for disruptions by reducing effective stock based on reliability
    if disruption:
        effective_stock = {sku.sku: sku.stock * sku.supplierReliability for sku in skuData}
    else:
        effective_stock = {sku.sku: sku.stock for sku in skuData}


    # 3. Define the Objective Function
    # Adjust demand for festival surges
    effective_demand = {sku.sku: min(sku.forecastDemand, sku.actualDemand) * surgeFactor for sku in skuData}
    model += pulp.lpSum(
        [effective_demand[sku.sku] - alloc_vars[sku.sku] for sku in skuData]
    ), "Total_Shortage"

    # 4. Define Constraints
    model += pulp.lpSum([alloc_vars[sku.sku] for sku in skuData]) <= capacity

    for sku in skuData:
        # Respect the effective stock level
        model += alloc_vars[sku.sku] <= effective_stock[sku.sku]
        # Can't allocate more than the demand
        model += alloc_vars[sku.sku] <= effective_demand[sku.sku]
        # Add MOQ constraint
        is_ordered = pulp.LpVariable(f"is_ordered_{sku.sku}", cat='Binary')
        model += alloc_vars[sku.sku] >= sku.moq * is_ordered
        # If we order, we must allocate at least MOQ, but not more than stock
        model += alloc_vars[sku.sku] <= effective_stock[sku.sku] * is_ordered


    # 5. Solve the model
    model.solve(pulp.PULP_CBC_CMD(msg=0)) # Suppress solver logs in production

    if model.status != pulp.LpStatusOptimal:
        return {
            "allocationPlan": [], "shortages": [], "excess": [],
            "status": pulp.LpStatus[model.status]
        }

    # 6. Extract results
    allocationPlan = []
    shortages = []
    excess = []

    for sku in skuData:
        allocated_val = int(alloc_vars[sku.sku].varValue)
        allocationPlan.append({"sku": sku.sku, "allocated": allocated_val})

        demand = effective_demand[sku.sku]
        if allocated_val < demand:
            shortages.append({"sku": sku.sku, "shortage": demand - allocated_val})

        if allocated_val < effective_stock[sku.sku]:
            excess.append({"sku": sku.sku, "excess": effective_stock[sku.sku] - allocated_val})

    return {
        "allocationPlan": allocationPlan, "shortages": shortages,
        "excess": excess, "status": pulp.LpStatus[model.status]
    }
