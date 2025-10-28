# Lightweight production image for FastAPI + scikit-learn
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model (optional in CI/CD). For reproducible builds you may comment this out
# and bake a pre-trained model in app/models/*.pkl
RUN python app/services/train_model.py --data-path data/train.csv --output app/models/churn_pipeline.pkl

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
