from fastapi import FastAPI
from app.routers import predict, upload
import os

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(upload.router,  prefix="/upload",  tags=["upload"])

@app.get("/health")
def health():
    model_path = "app/models/churn_pipeline.pkl"
    model_loaded = os.path.exists(model_path)
    return {"status": "healthy" if model_loaded else "unhealthy", "model_loaded": model_loaded}
