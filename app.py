from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title='Pesticides Use Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

# Cargar modelo entrenado
model = load(pathlib.Path('model/pesticides-model-v1.joblib'))

# Definir la entrada de datos
class InputData(BaseModel):
    Area: str = "Mexico"   # País
    Year: int = 2020       # Año

# Definir la salida
class OutputData(BaseModel):
    prediction: float = 0.0

@app.post('/predict', response_model=OutputData)
def predict(data: InputData):
    # Convertimos los datos a un DataFrame con los nombres de columna que el modelo espera
    model_input = pd.DataFrame([data.dict()])

    # Hacemos la predicción
    result = model.predict(model_input)

    return {'prediction': float(result[0])}

