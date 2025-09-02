from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from app.routers import load_balancer, inventory_optimizer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smart Supply Chain Optimizer")

# --- FIX: Updated CORS Configuration ---
# Define the specific origins that are allowed to connect.
# This is more secure than allowing everyone ("*").
origins = [
    "http://localhost:5173",  # Your local development frontend
    "http://localhost:3000",  # Another common local port
    # Add your deployed frontend URL here when you have it
    "https://cog-front.vercel.app",
    "https://cog-front.vercel.app/load-balancer",
    "https://cogni-haka.vercel.app/load-balancer",
    "https://cogni-haka.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Routers
app.include_router(load_balancer.router, prefix="/api")
app.include_router(inventory_optimizer.router, prefix="/api")