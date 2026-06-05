from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from catboost import CatBoostRegressor
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import csv

# ===  НАСТРОЙКА ПУТЕЙ  ===
BASE_DIR = Path(__file__).resolve().parent.parent  # корень проекта (Kvartis-bot)

MODEL_PATH = BASE_DIR / "models" / "kvartis_model.cbm"
SCALER_PATH = BASE_DIR / "models" / "kvartis_scaler.pkl"
CSV_PATH = BASE_DIR / "web" / "wdata.csv"

# ===  ЗАГРУЗКА МОДЕЛИ  ===
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
    repair: str = Form(...),
    floor: int = Form(...),
    all_floor: int = Form(...)
):
    # ===  Перезаписываем wdata.csv  ===
    row = [city, rooms, m2, repair, floor, all_floor]
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)          


    df = pd.read_csv(CSV_PATH, header=None)
    df.columns = ['city', 'rooms', 'm2', 'repair', 'floor', 'all_floor']
    row = df.iloc[0]
    x = pd.DataFrame([row])

    numeric = ['rooms', 'm2', 'floor', 'all_floor']
    x[numeric] = scaler.transform(x[numeric])

    pred_log = model.predict(x)
    pred_price = np.expm1(pred_log)[0]

# консоль
    print(x.to_string(index=False))
    print("цена ≈", f"{pred_price:,.0f}", "₽")

    result = f"цена ≈ {pred_price:,.0f} ₽"

    inputs = {
        "city": city,
        "rooms": rooms,
        "m2": m2,
        "repair": repair,
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

