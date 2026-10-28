from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
kaggle_path = ROOT / "data" / "data.csv"
df = pd.read_csv(kaggle_path if kaggle_path.exists() else ROOT / "data" / "breast_cancer.csv")
rename_map = {
    "radius_mean": "mean radius", "texture_mean": "mean texture",
    "perimeter_mean": "mean perimeter", "area_mean": "mean area",
    "smoothness_mean": "mean smoothness", "compactness_mean": "mean compactness",
    "concavity_mean": "mean concavity", "concave points_mean": "mean concave points",
    "symmetry_mean": "mean symmetry", "fractal_dimension_mean": "mean fractal dimension",
    "radius_se": "radius error", "texture_se": "texture error",
    "perimeter_se": "perimeter error", "area_se": "area error",
    "smoothness_se": "smoothness error", "compactness_se": "compactness error",
    "concavity_se": "concavity error", "concave points_se": "concave points error",
    "symmetry_se": "symmetry error", "fractal_dimension_se": "fractal dimension error",
    "radius_worst": "worst radius", "texture_worst": "worst texture",
    "perimeter_worst": "worst perimeter", "area_worst": "worst area",
    "smoothness_worst": "worst smoothness", "compactness_worst": "worst compactness",
    "concavity_worst": "worst concavity", "concave points_worst": "worst concave points",
    "symmetry_worst": "worst symmetry", "fractal_dimension_worst": "worst fractal dimension",
}
df = df.rename(columns=rename_map)
df = df.drop(columns=[c for c in ["id", "Unnamed: 32", "target"] if c in df.columns], errors="ignore")
features = [c for c in df.columns if c != "diagnosis"]
X = df[features].apply(pd.to_numeric, errors="coerce")
y = (df["diagnosis"] == "M").astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
])
model.fit(X_train, y_train)
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
joblib.dump({"model": model, "feature_names": features,
             "target_labels": {"0": "benign", "1": "malignant"}},
            ROOT / "models" / "model.joblib")
print("Saved models/model.joblib")
