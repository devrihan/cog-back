import numpy as np

# ----------- LOAD BALANCER OPTIMIZER ----------- #
import numpy as np

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



# ----------- INVENTORY OPTIMIZER ----------- #
# import pulp

# def optimize_inventory(skuData, capacity: int):
#     """
#     Optimizes inventory allocation using a linear programming model.
#     """
#     # 1. Create the model
#     model = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)

#     # 2. Define Decision Variables
#     alloc_vars = {
#         sku.sku: pulp.LpVariable(f"alloc_{sku.sku}", lowBound=0, cat='Integer')
#         for sku in skuData
#     }

#     # 3. Define the Objective Function
#     effective_demand = {sku.sku: min(sku.forecastDemand, sku.actualDemand) for sku in skuData}
#     model += pulp.lpSum(
#         [effective_demand[sku.sku] - alloc_vars[sku.sku] for sku in skuData]
#     ), "Total_Shortage"

#     # 4. Define Constraints
#     model += pulp.lpSum([alloc_vars[sku.sku] for sku in skuData]) <= capacity
#     for sku in skuData:
#         model += alloc_vars[sku.sku] <= sku.stock
#         model += alloc_vars[sku.sku] <= effective_demand[sku.sku]

#     # 5. Solve the model
#     model.solve()


#     # Check if the solution is optimal before proceeding
#     if model.status != pulp.LpStatusOptimal:
#         return {
#             "allocationPlan": [],
#             "shortages": [],
#             "excess": [],
#             "status": pulp.LpStatus[model.status]
#         }

#     # 6. Extract results and calculate shortage/excess
#     allocationPlan = []
#     shortages = []
#     excess = []

#     for sku in skuData:
#         # Get the optimal allocated value from the solver
#         allocated_val = int(alloc_vars[sku.sku].varValue)
#         allocationPlan.append({"sku": sku.sku, "allocated": allocated_val})

#         # Calculate shortage
#         demand = effective_demand[sku.sku]
#         if allocated_val < demand:
#             shortages.append({"sku": sku.sku, "shortage": demand - allocated_val})

#         # Calculate excess
#         if allocated_val < sku.stock:
#             excess.append({"sku": sku.sku, "excess": sku.stock - allocated_val})
            
#     # 7. Return the complete data structure
#     return {
#         "allocationPlan": allocationPlan,
#         "shortages": shortages,
#         "excess": excess,
#         "status": pulp.LpStatus[model.status]
#     }



import pulp
import numpy as np

# ... (keep the existing optimize_orders function)

# ----------- INVENTORY OPTIMIZER ----------- #
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
    model.solve()

    if model.status != pulp.LpStatusOptimal:
        return {
            "allocationPlan": [],
            "shortages": [],
            "excess": [],
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
        "allocationPlan": allocationPlan,
        "shortages": shortages,
        "excess": excess,
        "status": pulp.LpStatus[model.status]
    }