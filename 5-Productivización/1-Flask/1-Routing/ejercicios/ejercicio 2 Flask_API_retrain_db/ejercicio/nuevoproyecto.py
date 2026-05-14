import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from sklearn.linear_model import LinearRegression

app = FastAPI(title="API de Predicción de Ventas")

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'Advertising.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'advertising.model')

# --- Modelos de Datos (Pydantic) ---
class AdvertisingData(BaseModel):
    tv: float
    radio: float
    newspaper: float
    sales: float

class PredictionInput(BaseModel):
    tv: float
    radio: float
    newspaper: float

# --- Funciones de Utilidad ---
def insert_data_csv(data: AdvertisingData):
    new_row = pd.DataFrame([[data.tv, data.radio, data.newspaper, data.sales]], 
                          columns=['TV', 'radio', 'newspaper', 'sales'])
    
    header = not os.path.exists(CSV_PATH)
    new_row.to_csv(CSV_PATH, mode='a', index=False, header=header)

def train_model_logic():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("El archivo CSV no existe.")
    
    df = pd.read_csv(CSV_PATH)
    X = df[['TV', 'radio', 'newspaper']]
    y = df['sales']
    
    model = LinearRegression()
    model.fit(X, y)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

# --- Endpoints ---

@app.get("/")
async def home():
    return {"message": "API para predecir ventas. Visita /docs para la documentación interactiva."}

@app.get("/datos")
async def devolver_datos():
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="El archivo CSV aún no existe")
    
    df = pd.read_csv(CSV_PATH)
    return df.to_dict(orient='records')

@app.get("/predecir")
async def predecir(tv: float, radio: float, newspaper: float):
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError()
            
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        prediction = model.predict([[tv, radio, newspaper]])
        return {"prediction": float(prediction[0])}
    
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Modelo no disponible. Reentrene primero.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingresar", status_code=201)
async def ingest(tv: float, radio: float, newspaper: float, sales: float):
    try:
        nuevo_registro = AdvertisingData(tv=tv, radio=radio, newspaper=newspaper, sales=sales)
        insert_data_csv(nuevo_registro)
        return {"status": "Registro insertado correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reentrenar")
async def reentrena():
    try:
        train_model_logic()
        return {"status": "Modelo actualizado con los datos del CSV"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)