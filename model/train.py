from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import pandas as pd
import pathlib
import numpy as np
import os

# Crear carpeta model si no existe
os.makedirs("model", exist_ok=True)

print("📂 Cargando dataset...")
df = pd.read_csv(pathlib.Path("data/pesticides.csv"))

# Features (Area, Year) y target (Value)
X = df[["Area", "Year"]]
y = df["Value"]

# Preprocesamiento: convertir Area en variables dummy
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Area"]),
        ("num", "passthrough", ["Year"])
    ]
)

# Modelo
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Entrenando modelo...")
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Modelo entrenado")
print(f"   RMSE en test: {rmse:.2f}")
print(f"   R² en test:   {r2:.2f}")

# Guardar modelo
print("💾 Guardando modelo...")
dump(model, pathlib.Path("model/pesticides-model-v1.joblib"))
print("🚀 Modelo guardado en model/pesticides-model-v1.joblib")

