# Customer Churn Prediction (FastAPI + scikit-learn)

A production-ready ML service that predicts **customer churn** from telecom usage data.
Built with **FastAPI**, **scikit-learn**, and **Docker**.

## Features
- `/predict` – JSON body with one or more customer records → churn probability
- `/upload` – Upload a CSV → batch predictions
- Auto-generated docs at `/docs`
- Example dataset in `data/` and a trained pipeline saved to `app/models/churn_pipeline.pkl`

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (Optional) Retrain pipeline
python app/services/train_model.py --data-path data/train.csv --output app/models/churn_pipeline.pkl

uvicorn app.main:app --reload
# Visit http://127.0.0.1:8000/docs
```

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Example requests

### Predict (single)
```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{
    "records": [{
      "state": "OH",
      "account_length": 103,
      "area_code": "area_code_408",
      "international_plan": "no",
      "voice_mail_plan": "yes",
      "number_vmail_messages": 29,
      "total_day_minutes": 294.7,
      "total_day_calls": 95,
      "total_day_charge": 50.10,
      "total_eve_minutes": 200.1,
      "total_eve_calls": 105,
      "total_eve_charge": 17.01,
      "total_night_minutes": 300.3,
      "total_night_calls": 127,
      "total_night_charge": 13.51,
      "total_intl_minutes": 13.7,
      "total_intl_calls": 6,
      "total_intl_charge": 3.70,
      "number_customer_service_calls": 1
    }]
  }'
```

### Upload CSV
```bash
curl -X POST "http://127.0.0.1:8000/upload"   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "file=@data/sample_upload.csv"
```

## Notes
- The pipeline uses a `ColumnTransformer` to one-hot encode categorical features and scale numeric features, then a `LogisticRegression` classifier.
- You can swap in any estimator (e.g., `RandomForestClassifier`) inside the pipeline.
