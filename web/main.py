from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import csv

# Загружаем модель и скейлер ОДИН РАЗ при старте приложения
model = CatBoostRegressor()
model.load_model('Kvartis/models/kvartis_model.cbm')
scaler = joblib.load('Kvartis/models/kvartis_scaler.pkl')

app = FastAPI(title="Kvartis Price Predictor")
templates = Jinja2Templates(directory="templates")

CSV_PATH = 'C:/holl/python/Kvartis/web/wdata.csv'


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
    # ===  Перезаписываем wdata.csv (УДАЛЯЕМ старое, пишем только новую строку без заголовка) ===
    row = [city, rooms, m2, repair, floor, all_floor]
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)          # строго одна строка, без заголовка — как было


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

