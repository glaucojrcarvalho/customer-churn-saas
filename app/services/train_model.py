import argparse, os, joblib, pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from app.services.preprocessing import CATEGORICAL, split_X_y, ensure_column_order

def build_pipeline(categorical):
    numeric = None  # infer later
    # Pipeline will infer numeric columns by excluding categorical at fit-time
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(with_mean=False), [])  # set later
        ],
        remainder="passthrough",
        sparse_threshold=0.0,
    )

    # Logistic Regression with simple regularization
    clf = LogisticRegression(max_iter=200, n_jobs=None)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe

def main(data_path: str, output: str):
    df = pd.read_csv(data_path)
    X, y = split_X_y(df.copy())
    X = ensure_column_order(X)

    # Determine numeric columns dynamically
    numeric_candidates = [c for c in X.columns if c not in CATEGORICAL]
    pipe = build_pipeline(CATEGORICAL)
    # Update ColumnTransformer numeric indices
    ct: ColumnTransformer = pipe.named_steps["pre"]
    # find numeric transformer index and set columns
    new_transformers = []
    for name, trans, cols in ct.transformers:
        if name == "num":
            new_transformers.append((name, trans, numeric_candidates))
        else:
            new_transformers.append((name, trans, cols))
    ct.transformers = new_transformers

    pipe.fit(X, y)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    joblib.dump(pipe, output)
    print(f"Saved pipeline to {output}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output", default="app/models/churn_pipeline.pkl")
    args = ap.parse_args()
    main(args.data_path, args.output)
