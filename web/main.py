from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from catboost import CatBoostRegressor
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# ===  НАСТРОЙКА ПУТЕЙ  ===
BASE_DIR = Path(__file__).resolve().parent.parent  # корень проекта 

MODEL_PATH = BASE_DIR / "models" / "kvartis_model.cbm"
SCALER_PATH = BASE_DIR / "models" / "kvartis_scaler.pkl"


# ===  ЗАГРУЗКА МОДЕЛИ И СКАЛЕРА  ===
model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))

scaler = joblib.load(str(SCALER_PATH))

app = FastAPI(title="Kvartis Price Predictor")
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "inputs": {}
        }
    )


@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    city: str = Form(...),
    rooms: int = Form(...),
    m2: float = Form(...),
    kitchen_m2: float = Form(...),
    repair: str = Form(...),
    district: str = Form(...),
    year: int = Form(...),
    floor: int = Form(...),
    all_floor: int = Form(...)
):


    current_year = 2026  
    building_age = current_year - year


    data = {
        'city': [city],
        'rooms': [rooms],
        'm2': [m2],
        'kitchen_m2': [kitchen_m2],
        'repair': [repair],
        'district': [district],
        'building_age': [building_age], 
        'floor': [floor],
        'all_floor': [all_floor]
    }
    x = pd.DataFrame(data)

    # ===  Масштабирование числовых признаков  ===

    numeric = ['rooms', 'm2', 'kitchen_m2', 'building_age', 'floor', 'all_floor']
    x[numeric] = scaler.transform(x[numeric])

    # ===  Предсказание  ===
    pred_log = model.predict(x)
    pred_price = np.expm1(pred_log)[0]

    # Вывод в консоль для отладки
    print(x.to_string(index=False))
    print("цена ≈", f"{pred_price:,.0f}", "₽")

    result = f"Цена ≈ {pred_price:,.0f} ₽"


    inputs = {
        "city": city,
        "rooms": rooms,
        "m2": m2,
        "kitchen_m2": kitchen_m2,
        "repair": repair,
        "district": district,
        "year": year,
        "floor": floor,
        "all_floor": all_floor
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "inputs": inputs
        }
    )
