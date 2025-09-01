# from fastapi import APIRouter
# from pydantic import BaseModel
# from typing import List

# from app.services.analyzer import analyze_inventory
# from app.services.optimizer import optimize_inventory

# router = APIRouter()

# # -------------------- MODELS -------------------- #
# class SKU(BaseModel):
#     sku: str
#     forecastDemand: int
#     actualDemand: int
#     stock: int

# class InventoryRequest(BaseModel):
#     skuData: List[SKU]
#     capacity: int


# # -------------------- ROUTES -------------------- #

# @router.post("/analyze-inventory")
# def analyze_inventory_route(req: InventoryRequest):
#     """Analyze inventory before running optimizer"""
#     return analyze_inventory(req.skuData)


# @router.post("/optimize-inventory")
# def run_inventory_optimizer(req: InventoryRequest):
#     """Run inventory optimization"""
#     return optimize_inventory(req.skuData, req.capacity)





from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
import google.generativeai as genai
import os

from app.services.analyzer import analyze_inventory
from app.services.optimizer import optimize_inventory

router = APIRouter()

# -------------------- MODELS -------------------- #
class SKU(BaseModel):
    sku: str
    forecastDemand: int
    actualDemand: int
    stock: int
    supplierReliability: float # New field
    moq: int # New field

class InventoryRequest(BaseModel):
    skuData: List[SKU]
    capacity: int
    surgeFactor: float = 1.0 # Optional field for festival surge
    disruption: bool = False # Optional field for disruptions

class AIInsightRequest(BaseModel):
    question: str
    analysis_data: Dict[str, Any]
    optimization_data: Dict[str, Any]

# -------------------- ROUTES -------------------- #

@router.post("/analyze-inventory")
def analyze_inventory_route(req: InventoryRequest):
    """Analyze inventory before running optimizer"""
    return analyze_inventory(req.skuData)


@router.post("/optimize-inventory")
def run_inventory_optimizer(req: InventoryRequest):
    """Run inventory optimization"""
    return optimize_inventory(req.skuData, req.capacity, req.surgeFactor, req.disruption)

@router.post("/inventory-insights")
def get_ai_insights(req: AIInsightRequest):
    """Get AI-powered insights for inventory optimization"""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set on the server.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        You are a supply chain analyst. Based on the following data, please answer the user's question.

        User Question: {req.question}

        Initial Analysis Data (before optimization):
        {req.analysis_data}

        Optimized Results Data:
        {req.optimization_data}

        Please provide a concise and helpful answer.
        """

        response = model.generate_content(prompt)

        return {"insight": response.text}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching AI insights: {e}")